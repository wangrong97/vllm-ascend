# vLLM Ascend Plugin documents

Live doc: <https://docs.vllm.ai/projects/ascend>

## VQ2A8 文档

VQ2A8 相关说明统一放在 [vq2a8/](vq2a8/) 目录：

- [GS32 VQ2/MXFP4 混合量化推理](vq2a8/vq2a8_gs32.md)
- [GS32 底层运行时](vq2a8/vq2a8_gs32_runtime.md)
- [NPU CI 回归](vq2a8/vq2a8_npu_ci.md)
- [性能测试](vq2a8/vq2a8_performance.md)
- [精度评测](vq2a8/vq2a8_accuracy.md)
- [混合专家权重交付规范建议](vq2a8/vq2_mixed_gs32_export_spec_20261016.md)

## Build the docs

The documentation uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Prerequisites

Run all commands in this guide from the repository root. Using a virtual
environment is recommended, but not required.

```bash
# Install documentation dependencies.
python -m pip install -r docs/requirements-docs.txt
```

### Build and serve (English)

```bash
# Serve docs locally with live reload.
make -f docs/Makefile serve
# Open http://127.0.0.1:8000/projects/ascend/en/latest/

# Or build to site/.
make -f docs/Makefile build

# Serve the built static files.
python -m http.server -d site/
# Open http://127.0.0.1:8000/
```

### Build and serve (Chinese)

Chinese docs are generated from `.po` translation files in
`docs/source/locale/zh_CN/LC_MESSAGES/`. The Chinese serve and build targets
generate the translated Markdown files automatically.

```bash
# Serve Chinese docs locally.
make -f docs/Makefile serve-zh

# Or build to site/zh/.
make -f docs/Makefile build-zh

# Or only generate Chinese Markdown files without serving or building.
make -f docs/Makefile gen-zh
```

### Migration from Sphinx

If you are migrating markdown files from the old Sphinx/MyST format, conversion follows these rules:

- MyST toctree → removed (nav is in `mkdocs.yml`)
- MyST admonitions → MkDocs admonition syntax
- MyST tab-set/tab-item → MkDocs Material tabbed syntax
- MyST code-block → standard fenced code blocks
- MyST variable substitution `|var|` → `{{ var }}` macro syntax

### Version variables

Version variables (e.g., `{{ vllm_ascend_version }}`) are defined in the
`extra` section of `mkdocs.yml` and substituted at build time by the
`mkdocs-macros` plugin.

To update versions, edit the `extra` section in `mkdocs.yml`.
