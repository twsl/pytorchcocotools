# pytorchcocotools: Unofficial APIs for the MS-COCO dataset using PyTorch

Welcome to pytorchcocotools's documentation!

<!--- BADGES: START --->
[![Build](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![vulnerability: safety](https://img.shields.io/badge/vulnerability-safety-yellow.svg)](https://github.com/pyupio/safety)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/pytorchcocotools/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)


<!--- BADGES: END --->

Unofficial APIs for the MS-COCO dataset using PyTorch

The file format is explained in the [official documentation](https://cocodataset.org/#format-data) and nicely summarized [here](https://www.youtube.com/watch?v=h6s61a_pqfM).

## Features

- Strongly typed COCO data format [represetation](./src/pytorchcocotools/internal/structure/)
- Drop-in compatible [`COCO`](./coco.md) and [`COCOeval`](./cocoeval.md) classes
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

## Usage

```python
import pytorchcocotools

...
```

See a more complete example in the [notebooks](notebooks) folder.

## API

Check the [API reference](api/pytorchcocotools/) for more details.
