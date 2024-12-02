# pytorchcocotools


[![Build](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/pytorchcocotools/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)


Unofficial APIs for the MS-COCO dataset using PyTorch.
Uses the [repository](https://github.com/ppwwyyxx/cocoapi) of the `pycocotools` packages as reference.

The file format is explained in the [official documentation](https://cocodataset.org/#format-data) and nicely summarized [here](https://www.youtube.com/watch?v=h6s61a_pqfM).

## Features

- Strongly typed COCO data format [represetation](./src/pytorchcocotools/internal/structure/)
- Drop-in compatible `COCO` and `COCOeval` classes
- (Almost) drop-in compatible `mask` methods
- Pure `torch` implementation
- `torchvision` data set using the latest transformation API
- fully unit tested and documented

## Installation

With `pip`:
```bash
python -m pip install pytorchcocotools
```

With [`poetry`](https://python-poetry.org/):
```bash
poetry add pytorchcocotools
```

## How to use it

```python
import pytorchcocotools

...
```

## Docs

```bash
poetry run mkdocs build -f ./docs/mkdocs.yml -d ./_build/
```


## Update template

```bash
copier update --trust
```

## Credits

This project was generated with [![🚀 A generic python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
