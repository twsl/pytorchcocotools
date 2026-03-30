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

### Run Line Profiler Tests Only

```bash
pytest tests/profiling/ -m line_profiling -v -s
```

Note: Use `-s` to see the profiler output tables printed to stdout.

### Run Specific Test Group

```bash
# Benchmark tests
pytest tests/profiling/test_bb_iou.py --benchmark-only -v

# Profiler tests
pytest tests/profiling/test_bb_iou.py -m profiling -v -s

# Line profiler tests
pytest tests/profiling/test_bb_iou.py -m line_profiling -v -s
```

### Run on Specific Device

```bash
# CPU benchmarks
pytest tests/profiling/ -k "cpu" --benchmark-only

# CPU profiling (torch.profiler)
pytest tests/profiling/ -k "cpu" -m profiling -v -s

# CPU profiling (line_profiler)
pytest tests/profiling/ -k "cpu" -m line_profiling -v -s

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
4. **Line Profiler** (`test_*_pt_line_profiling`) - Line-by-line profiling with `line_profiler` showing per-line timing and hit counts
5. **Correctness Tests** (`test_*_correctness`) - Non-benchmark tests for correctness verification

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

## Profiling with Line Profiler

The line profiling tests (`test_*_pt_line_profiling`) use `line_profiler` to provide line-by-line execution timing for each function. This is useful for identifying exactly which lines within a function are the most expensive.

Each line profiling test runs the function 10 times after a 5-iteration warmup and prints a table showing:

- **Line #**: Source line number
- **Hits**: Number of times the line was executed
- **Time**: Total time spent on the line (in microseconds)
- **Per Hit**: Average time per execution
- **% Time**: Percentage of total function time
- **Line Contents**: The source code

Example output:

```
=== bbIou Line Profiling Results (device=cpu) ===
Timer unit: 1e-06 s

Total time: 0.001234 s
File: .../bb_iou.py
Function: bbIou at line 10

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    10                                           def bbIou(dt, gt, iscrowd):
    11        10         50.0      5.0      4.1      ...
    12        10        800.0     80.0     64.8      ...
```

### Advanced Profiling

For production profiling or more detailed analysis, use the dedicated profiling script:

```bash
python scripts/profile_methods.py
```
