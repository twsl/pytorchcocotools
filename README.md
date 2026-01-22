# pytorchcocotools

[![Build](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/pytorchcocotools/actions/workflows/docs.yaml)
![GitHub Release](https://img.shields.io/github/v/release/twsl/pytorchcocotools?include_prereleases)
[![PyPI - Package Version](https://img.shields.io/pypi/v/pytorchcocotools?logo=pypi&style=flat&color=orange)](https://pypi.org/project/pytorchcocotools/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorchcocotools?logo=pypi&style=flat&color=blue)](https://pypi.org/project/pytorchcocotools/)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/pytorchcocotools/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Unofficial APIs for the MS-COCO dataset using PyTorch.
Uses the official [repository](https://github.com/ppwwyyxx/cocoapi) of the `pycocotools` packages as reference.

The file format is explained in the [official documentation](https://cocodataset.org/#format-data) and nicely summarized [here](https://www.youtube.com/watch?v=h6s61a_pqfM).

> [!WARNING]
> This is work in progress, feel free to contribute!

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

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add pytorchcocotools
```

## How to use it

Pretty much all you need to do is to change the import statement from `pycocotools` to `pytorchcocotools`:

```diff
-import pycocotools
+import pytorchcocotools
```

So all imports look like this:

```python
from pytorchcocotools.coco import COCO
from pytorchcocotools.cocoeval import COCOeval

...
```

## API

> [!WARNING]
> While the API is mostly compatible with the original `pycocotools` package, there are some differences.
> For some methods you need to adapt the handling of the return type. See the examples below.

> [!NOTE]
> All methods are documented with detailed type hints.

### `mask`

All methods now have a optional `device` and `requires_grad` parameters that allows to specify the device on which the masks should be created and whether gradients are required. This is useful for acceleration.

> [!IMPORTANT] > `decode`, `encode`, `toBbox` and `frPyObjects` now always return the batch/channel dimension implementation as opposed to a single return element if only a single element was passed.
> This was done to make the API more consistent by providing single, defined return types, but is open for further discussion.

### `COCO`

One major difference is that the `COCO` class now uses a strongly typed data structure to represent the COCO data format. This makes it easier to work with the data and to understand the structure of the data, but also might cause problems with data sources that do not strictly follow the data format.

### `COCOeval`

Strongly typed as well. Includes also minor fixes, e.g. the `__str__` now also returns the `stats`.

### Other additions

- `pytorchcocotools.utils.coco.download.CocoDownloader`: While `gsutils rsync` is the officially recommended way to download the data, this allows you to trigger the download from Python.
- The `logger` property in both all classes from the `logging` module replaces the `print` command, so you can fully customize it.
- `pytorchcocotools.torch.dataset.CocoDetection`: A drop-in replacement for the dataset from `torchvision`, now strongly typed using the new `transforms.v2` api.

## Docs

```bash
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```

## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```

## Credits

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
