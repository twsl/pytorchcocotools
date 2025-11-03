# pytest-benchmark Custom Grouping

This repository now includes custom pytest-benchmark grouping that automatically clusters tests ending with `_np` (NumPy) and `_pt` (PyTorch) together, making it easy to compare implementations side-by-side.

## Features

- **Automatic Clustering**: Tests with `_np` and `_pt` suffixes are grouped together by their base name
- **Device Agnostic**: The "device" parameter (cpu/cuda) is filtered out during grouping
- **Flexible**: Works with all standard pytest-benchmark grouping options
- **Backward Compatible**: Tests not matching the pattern use default grouping

## Usage

Run benchmarks with grouping options:

```bash
# Group by benchmark group name
pytest --benchmark-only --benchmark-group-by=group

# Group by benchmark group and function name (recommended for _np/_pt comparison)
pytest --benchmark-only --benchmark-group-by=group,name

# Group by function name only
pytest --benchmark-only --benchmark-group-by=func
```

## Example Output

**Before (without custom grouping):**
```
benchmark 'area': 8 tests
- test_area_np[cpu-test1]
- test_area_np[cpu-test2]
- test_area_np[cuda-test1]
- test_area_np[cuda-test2]
- test_area_pt[cpu-test1]
- test_area_pt[cpu-test2]
- test_area_pt[cuda-test1]
- test_area_pt[cuda-test2]
```

**After (with custom grouping, using `--benchmark-group-by=group,name`):**
```
benchmark 'area test_area': 8 tests
- test_area_np[cpu-test1]
- test_area_np[cuda-test1]
- test_area_np[cpu-test2]
- test_area_np[cuda-test2]
- test_area_pt[cpu-test1]
- test_area_pt[cuda-test1]
- test_area_pt[cpu-test2]
- test_area_pt[cuda-test2]
```

Notice that both `_np` and `_pt` variants are now grouped under `test_area`, and the device parameter doesn't affect grouping.

## Implementation Details

The custom grouping is implemented via the `pytest_benchmark_group_stats` hook in `tests/conftest.py`:

1. Detects test names ending with `_np` or `_pt`
2. Extracts the base function name (e.g., `test_evaluate` from `test_evaluate_np`)
3. Groups tests by base name, ignoring the suffix
4. Filters out "device" parameters when using `param` grouping
5. Falls back to default grouping for non-matching tests

## Test Coverage

Comprehensive tests are provided in `tests/test_benchmark_grouping.py` covering:
- _np/_pt clustering behavior
- Device parameter filtering
- Different grouping options
- Fallback for non-matching patterns
