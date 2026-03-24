# Profiling Tests for Internal Mask API

This directory contains comprehensive profiling and benchmarking tests for all methods in the `pytorchcocotools.internal.mask_api` package.

## Test Files

- **test_bb_iou.py** - Bounding box IoU computation (`bbIou`)
- **test_bb_nms.py** - Bounding box NMS (`bbNms`)
- **test_fr_bbox.py** - RLE from bounding boxes (`rleFrBbox`)
- **test_fr_poly.py** - RLE from polygons (`rleFrPoly`)
- **test_fr_string.py** - RLE from compressed string (`rleFrString`)
- **test_nms.py** - RLE-based NMS (`rleNms`)
- **test_to_string.py** - RLE to compressed string (`rleToString`)

## Running Tests

### Run All Profiling Tests

```bash
pytest tests/profiling/ -v
```

### Run Benchmark Tests Only

```bash
pytest tests/profiling/ --benchmark-only
```

### Run PyTorch Profiler Tests Only

```bash
pytest tests/profiling/ -m profiling -v -s
```

Note: Use `-s` to see the profiler output tables printed to stdout.

### Run Specific Test Group

```bash
# Benchmark tests
pytest tests/profiling/test_bb_iou.py --benchmark-only -v

# Profiler tests
pytest tests/profiling/test_bb_iou.py -m profiling -v -s
```

### Run on Specific Device

```bash
# CPU benchmarks
pytest tests/profiling/ -k "cpu" --benchmark-only

# CPU profiling
pytest tests/profiling/ -k "cpu" -m profiling -v -s

# CUDA (if available)
pytest tests/profiling/ -k "cuda" --benchmark-only
pytest tests/profiling/ -k "cuda" -m profiling -v -s
```

### Generate Benchmark Comparison

```bash
pytest tests/profiling/ --benchmark-only --benchmark-compare
```

## Test Structure

Each test file contains:

1. **Test Cases** - Parametrized test cases using `pytest-cases`
2. **PyTorch Benchmarks** (`test_*_pt`) - Benchmark PyTorch implementation with pytest-benchmark
3. **PyTorch Profiler** (`test_*_pt_profiling`) - Detailed profiling with `torch.profiler` including CPU/GPU metrics, memory usage, and stack traces
4. **Correctness Tests** (`test_*_correctness`) - Non-benchmark tests for correctness verification

All tests are parametrized by the `device` fixture, which tests on both CPU and CUDA (if available).

## Profiling with PyTorch Profiler

The profiling tests (`test_*_pt_profiling`) use `torch.profiler` to provide detailed performance analysis including:

- **CPU Time**: Time spent on CPU operations
- **CUDA Time**: Time spent on GPU operations (CUDA devices only)
- **Memory Usage**: CPU and GPU memory allocation/deallocation
- **Call Stack**: Stack traces for performance bottlenecks
- **Operator Details**: Per-operation performance metrics

### Profiler Output

Each profiling test runs the function 10 times after a 5-iteration warmup and prints a table showing:

- Operation names
- CPU time total and self time
- Number of calls
- Memory usage (if applicable)

Example output:

```
=== bbIou Profiling Results (device=cpu) ===
-------------------------------------------------------  ------------  ------------  ...
Name                                                     Self CPU %      Self CPU   ...
-------------------------------------------------------  ------------  ------------  ...
aten::box_iou                                            45.23%        12.345ms    ...
aten::mul                                                15.67%        4.234ms     ...
...
```

### Advanced Profiling

For production profiling or more detailed analysis, use the dedicated profiling script:

```bash
python scripts/profile_methods.py
```
