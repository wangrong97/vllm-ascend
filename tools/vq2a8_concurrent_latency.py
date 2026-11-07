# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run bounded concurrent SSE requests with cancellable client processes."""

import multiprocessing
import sys
from contextlib import redirect_stderr
from multiprocessing.connection import wait

from tools import benchmark_vq2a8_serving as bench


class ProgressOutput:
    """Send complete progress lines to the parent, which owns all log files."""

    def __init__(self, connection):
        self.connection = connection
        self.pending = ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.connection.send(("progress", line))
        return len(text)

    def flush(self):
        pass


def request_worker(connection, args):
    """Keep transport timing in the worker and report results to one writer."""
    try:
        connection.send(("ready", None))
        with redirect_stderr(ProgressOutput(connection)):
            while (task := connection.recv()) is not None:
                phase, index, count = task
                try:
                    result = bench.run_request(args, phase, index, count)
                except (Exception, KeyboardInterrupt) as error:
                    connection.send(("error", f"{type(error).__name__}: {error}"))
                    return
                connection.send(("result", result))
    finally:
        connection.close()


def receive(connection):
    try:
        return connection.recv()
    except (EOFError, OSError) as error:
        raise RuntimeError("Concurrent request worker exited without a result") from error


def stop_workers(workers):
    # Terminate active HTTP clients on failure or Ctrl+C; waiting for their
    # socket timeout could otherwise take hours for a long-context request.
    for _, process in workers:
        if process.is_alive():
            process.terminate()
    for connection, process in workers:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        connection.close()
        process.close()


def run_phase(args, concurrency, phase, count, on_result):
    """Fill at most C request slots, saving each completion before refilling.

    Workers are spawned before requests start; process creation is outside
    each request's SSE timing. A failure cancels in-flight client processes
    and prevents further submissions. The server may still finish their work.
    """
    if not count:
        return
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for _ in range(min(concurrency, count)):
            parent, child = context.Pipe()
            process = context.Process(target=request_worker, args=(child, args))
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            workers.append((parent, process))
        for connection, _ in workers:
            if receive(connection)[0] != "ready":
                raise RuntimeError("Concurrent request worker did not become ready")
        active = {}
        next_index = 1
        for connection, _ in workers:
            connection.send((phase, next_index, count))
            active[connection] = next_index
            next_index += 1
        while active:
            finished = []
            for connection in wait(active):
                kind, value = receive(connection)
                if kind == "progress":
                    print(value, file=sys.stderr, flush=True)
                elif kind == "error":
                    raise RuntimeError(f"{phase}={active[connection]}/{count}: {value}")
                elif kind == "result":
                    on_result(active.pop(connection), value)
                    finished.append(connection)
                else:
                    raise RuntimeError(f"Unexpected worker message: {kind}")
            for connection in finished:
                if next_index <= count:
                    connection.send((phase, next_index, count))
                    active[connection] = next_index
                    next_index += 1
    finally:
        stop_workers(workers)
