---
name: pytest-benchmark-tracking
description: Track and analyze Python function performance improvements across multiple benchmark runs using pytest-park and pytest-benchmark. Use when optimizing code performance, detecting regressions, validating improvements, or comparing different optimization strategies.
license: Apache-2.0
compatibility: Requires Python 3.8+, pytest, pytest-benchmark, pytest-park
metadata:
    author: twsl
    version: "1.0"
    domain: performance-testing
    repository: github.com/twsl/pytest-park
---

# Benchmarking Function Improvements with pytest-park

## Overview

This skill teaches AI assistants how to guide users in tracking and analyzing function performance improvements across multiple benchmark runs using pytest-park. The skill covers the complete workflow from initial setup through comparison and interpretation of results.

**Target Audience**: Python developers optimizing code performance
**Prerequisites**: pytest, pytest-benchmark installed
**When to Use**: Performance optimization workflows, regression detection, improvement validation

---

## Core Concepts

### 1. Benchmark Runs

A **run** is a collection of benchmark measurements captured at a specific point in time:

- Stored as pytest-benchmark JSON files (`--benchmark-save` output)
- Each run has an identifier (run_id, tag, or datetime)
- Runs are typically organized in a folder hierarchy
- Example tags: `baseline`, `optimized`, `v1.0`, `v2.0`

### 2. Benchmark Cases

Individual performance measurements within a run:

- Each test function generates one or more benchmark cases
- Cases include statistics: mean, median, min, max, stddev, rounds, iterations
- Cases can have parameters (e.g., `device=cpu`) and custom metadata

### 3. Name Normalization

pytest-park parses test names to enable variant comparison:

```
test_func1_original[cpu] → base_name: test_func1, params: cpu, postfix: _original
test_func1_new[cpu]      → base_name: test_func1, params: cpu, postfix: _new
```

This enables comparing different implementations of the same logical function.

### 4. Grouping Strategies

Benchmarks can be grouped for analysis using precedence order:

1. **custom:<key>** - Custom metadata from `extra_info["custom_groups"]["<key>"]`
2. **group** - Base method name (e.g., `test_func1`)
3. **name**, **method**, **func** - Base method name (e.g., `test_func1`)
4. **fullname**, **fullfunc** - Full path with base method name
5. **param** - All parameters combined (can ignore specific parameters via `ignore_params`)
6. **param:<name>** - Specific parameter (e.g., `param:device`)
7. **postfix**, **benchmark_postfix** - The parsed postfix (e.g., `_original`)

### 5. Performance Metrics

- **Delta percentage**: `(candidate - reference) / reference × 100`
    - Negative = improvement (faster)
    - Positive = regression (slower)
- **Speedup ratio**: `reference_mean / candidate_mean`
    - `>1.0` = improvement
    - `<1.0` = regression
- **Threshold**: Changes within ±1e-9 considered "unchanged"

---

## Prerequisites and Setup

### Installation

```bash
# With pip
pip install pytest-park pytest-benchmark

# With uv (recommended)
uv add --group test pytest-park pytest-benchmark
```

### Directory Structure

```
project/
├── benchmarks/          # Store benchmark JSON files here
├── tests/
│   ├── conftest.py     # Configure pytest-park integration
│   └── test_*.py       # Benchmark tests
└── src/                # Your code to optimize
```

---

## Complete Workflow: Tracking Function Improvements

### Step 1: Write Baseline Benchmark Tests

Create benchmark tests with descriptive naming conventions:

```python
# tests/test_performance.py
import time
from pytest_benchmark.fixture import BenchmarkFixture

def process_data_original(data: list[int]) -> int:
    """Original implementation - baseline."""
    time.sleep(0.1)
    return sum(data)

def process_data_optimized(data: list[int]) -> int:
    """Optimized implementation."""
    time.sleep(0.05)
    return sum(data)

def test_process_data_original(benchmark: BenchmarkFixture):
    data = list(range(1000))
    result = benchmark(process_data_original, data)
    assert result == 499500

def test_process_data_optimized(benchmark: BenchmarkFixture):
    data = list(range(1000))
    result = benchmark(process_data_optimized, data)
    assert result == 499500
```

**Best Practices**:

- Use consistent postfixes: `_original`, `_baseline`, `_old` vs `_optimized`, `_new`, `_fast`
- Keep test logic identical except for the function being benchmarked
- Use descriptive names that indicate what's being measured

### Step 2: Configure pytest-park Integration (Optional but Recommended)

Add pytest-park grouping to your `conftest.py`:

```python
# tests/conftest.py
from typing import Any
from pytest_park.pytest_benchmark import default_pytest_benchmark_group_stats

def pytest_benchmark_group_stats(config: Any, benchmarks: list[Any], group_by: str) -> list[tuple[str, list[Any]]]:
    """Enable pytest-park name parsing and grouping."""
    return default_pytest_benchmark_group_stats(
        config,
        benchmarks,
        group_by,
        original_postfix="original",
        reference_postfix="optimized",
        group_values_by_postfix={
            "original": "baseline",
            "optimized": "improved",
        },
        ignore_params=["device"],  # Optional: ignore specific parameters in grouping
    )
```

This configuration:

- Normalizes names by removing postfixes
- Groups `test_func_original` and `test_func_optimized` together
- Creates friendly labels for visualization
- Can ignore specific parameters (like `device`) when grouping by `param`

### Step 3: Add Custom Metadata for Tracking Optimization Techniques (Optional)

Tag benchmarks with metadata about which optimization technique was used:

```python
def test_process_data_optimized(benchmark: BenchmarkFixture):
    benchmark.extra_info["custom_groups"] = {
        "technique": "vectorization",
        "difficulty": "medium"
    }
    data = list(range(1000))
    result = benchmark(process_data_optimized, data)
    assert result == 499500
```

This enables grouping by technique: `--group-by custom:technique`

### Step 4: Capture Baseline Run

```bash
pytest tests/ --benchmark-only --benchmark-save=baseline --benchmark-save-data
```

**Explanation**:

- `--benchmark-only` - Skip non-benchmark tests
- `--benchmark-save=baseline` - Tag this run as "baseline"
- `--benchmark-save-data` - Save full data (required for pytest-park)

**Output**: Creates `.benchmarks/<platform>/<timestamp>_baseline.json`

### Step 5: Make Code Improvements

Optimize your functions, then keep or update the `_optimized` test variants:

```python
def process_data_optimized(data: list[int]) -> int:
    """Even more optimized implementation."""
    time.sleep(0.02)  # Now 5x faster!
    return sum(data)
```

### Step 6: Capture Optimized Run

```bash
pytest tests/ --benchmark-only --benchmark-save=optimized_v1 --benchmark-save-data
```

### Step 7: Compare Runs

#### Quick Comparison (Latest vs Previous)

```bash
pytest-park analyze ./.benchmarks
```

By default, this will compare and print details for all methods without prompting.

**Output**:

```
        Benchmark Analysis (Candidate: 2026-02-18T23:49:28.042400+00:00)
┏━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┓
┃       ┃       ┃   Avg ┃   Avg ┃   Med ┃   Med ┃   Avg ┃   Avg ┃  Med ┃   Med ┃
┃       ┃       ┃    vs ┃    vs ┃    vs ┃    vs ┃    vs ┃    vs ┃   vs ┃    vs ┃
┃       ┃       ┃  Orig ┃  Orig ┃  Orig ┃  Orig ┃  Prev ┃  Prev ┃ Prev ┃  Prev ┃
┃ Group ┃ Meth… ┃ (Tim… ┃   (%) ┃ (Tim… ┃   (%) ┃ (Tim… ┃   (%) ┃ (Ti… ┃   (%) ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━┩
│ cust… │ test… │   N/A │   N/A │   N/A │   N/A │ +0.0… │ +0.0… │ +0.… │ +0.0… │
│ cust… │ test… │   N/A │   N/A │   N/A │   N/A │ +0.0… │ +0.0… │ +0.… │ +0.0… │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┘
```

#### Explicit Run Selection

```bash
pytest-park analyze ./.benchmarks --reference baseline --candidate optimized_v1
```

#### Group by Base Method

```bash
pytest-park analyze ./.benchmarks --reference baseline --candidate optimized_v1 --group-by group
```

#### Group by Optimization Technique

```bash
pytest-park analyze ./.benchmarks --reference baseline --candidate optimized_v1 --group-by custom:technique
```

#### Exclude Parameters

```bash
pytest-park analyze ./.benchmarks --exclude-param device
```

### Step 8: Interactive Dashboard

For exploratory analysis:

```bash
pytest-park serve ./.benchmarks --reference baseline --port 8080
```

**Features**:

- Real-time run selection and comparison
- Interactive charts (history, delta distribution, top movers)
- Method-level drill-down
- Filterable views

Access at: `http://127.0.0.1:8080`

---

## CLI Command Reference

### `pytest-park version`

Print the installed version.

```bash
pytest-park version
```

### `pytest-park` (no arguments)

Launch interactive mode. Presents a numbered menu for `analyze`, `serve`, and `version`, then prompts for required arguments.

```bash
pytest-park
```

### `pytest-park analyze <folder>`

Compare two runs and print a rich summary table.

```bash
pytest-park analyze ./.benchmarks --group-by group --group-by param:device --exclude-param device
```

**Options**:

- `--reference <id|tag>` - Reference run (defaults to second-latest when both flags are omitted; when only `--candidate` is given, defaults to the run immediately preceding that candidate)
- `--candidate <id|tag>` - Candidate run (defaults to latest when only `--reference` is given)
- `--group-by <token>` - Grouping strategy (repeatable)
- `--distinct-param <key>` - Treat parameter as a separate dimension instead of collapsing it (repeatable)
- `--exclude-param <key>` - Parameter key to exclude from comparison (repeatable)
- `--original-postfix <str>` - Configure name parsing for candidate/original method names
- `--reference-postfix <str>` - Configure name parsing for reference method names
- `--profiler-folder <path>` - Optional folder containing profiler JSON artifacts to attach

**No arguments**: Compares latest run (candidate) against second-latest run (reference)

### `pytest-park serve <folder>`

Launch interactive NiceGUI dashboard.

```bash
pytest-park serve ./.benchmarks --reference baseline --host 127.0.0.1 --port 8080
```

**Options**:

- `--reference <id|tag>` - Default reference run
- `--group-by <token>` - Initial grouping (repeatable)
- `--distinct-param <key>` - Treat parameter as a separate dimension (repeatable)
- `--host <address>` - Server host (default: 127.0.0.1)
- `--port <number>` - Server port (default: 8080)
- `--original-postfix <str>` - Configure name parsing for candidate/original method names
- `--reference-postfix <str>` - Configure name parsing for reference method names
- `--profiler-folder <path>` - Optional folder containing profiler JSON artifacts to attach

---

## Best Practices

### Naming Conventions

✅ **Good**:

```python
test_parse_json_original()
test_parse_json_optimized()
test_parse_json_baseline()
test_parse_json_fast()
```

❌ **Avoid**:

```python
test_parse_json_1()  # Non-descriptive
test_parse_json_2()  # Unclear what changed
```

### Tagging Runs

✅ **Good tags**:

- `baseline` - Initial implementation
- `optimized_v1`, `optimized_v2` - Iteration versions
- `vectorized` - Specific technique applied
- `release_1.0` - Version correlation

❌ **Avoid**:

- Generic timestamps only
- Non-descriptive tags like `run1`, `run2`

### Parametrization

Use pytest parameters for different scenarios:

```python
@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    return request.param

def test_compute_optimized(benchmark, device):
    benchmark.extra_info["custom_groups"] = {"device_type": device}
    result = benchmark(compute, device=device)
```

Then group by parameter: `--group-by param:device`

### Custom Metadata

Store rich metadata for analysis:

```python
def test_optimized(benchmark):
    benchmark.extra_info["custom_groups"] = {
        "technique": "caching",           # What optimization
        "complexity": "O(1)",             # Algorithmic complexity
        "author": "alice",                # Who implemented
        "difficulty": "easy"              # Implementation difficulty
    }
    result = benchmark(optimized_func)
```

Group by any field: `--group-by custom:technique`, `--group-by custom:author`

### Iteration and Stability

For reliable benchmarks:

```python
# More rounds = more reliable statistics
benchmark.pedantic(func, rounds=10, iterations=100)

# Or let pytest-benchmark auto-calibrate
benchmark(func)  # Auto-calibrates rounds/iterations
```

---

## Interpreting Results

### Delta Percentages

```
avg_delta=-50.0%    → 50% faster (2x speedup)
avg_delta=-80.0%    → 80% faster (5x speedup)
avg_delta=+100.0%   → 100% slower (0.5x speedup - regression!)
avg_delta=-0.01%    → Negligible change
```

**Formula**: `(candidate_mean - reference_mean) / reference_mean × 100`

### Speedup Ratios

```
avg_speedup=2.0     → 2x faster
avg_speedup=5.0     → 5x faster
avg_speedup=0.5     → 2x slower (regression!)
avg_speedup=1.001   → Negligible change
```

**Formula**: `reference_mean / candidate_mean`

### Improvement/Regression Counts

```
improved=5, regressed=1, unchanged=0
```

- **improved**: candidate_mean < reference_mean (faster)
- **regressed**: candidate_mean > reference_mean (slower)
- **unchanged**: absolute difference ≤ 1e-9 (effectively identical)

### Example Output Interpretation

```
Compared run optimized_v2 against reference baseline
Accumulated: count=10, avg_delta=-65.5%, median_delta=-70.2%, avg_speedup=2.9, improved=9, regressed=1, unchanged=0

- group=test_parse: count=4, avg_delta=-75.0%, improved=4, regressed=0
- group=test_compute: count=4, avg_delta=-60.0%, improved=4, regressed=0
- group=test_validate: count=2, avg_delta=-40.0%, improved=1, regressed=1
```

**Interpretation**:

- Overall: 65.5% faster on average, ~3x speedup
- 9 out of 10 benchmarks improved
- 1 regression detected in the `test_validate` group
- `test_parse` showed the best improvement (75% faster)
- **Action**: Investigate the `test_validate` regression

---

## Common Pitfalls and Troubleshooting

### Issue: "No runs found"

**Cause**: Benchmark folder is empty or pytest-benchmark didn't save files.

**Solution**:

```bash
# Ensure --benchmark-save-data is used
pytest tests/ --benchmark-only --benchmark-save=myrun --benchmark-save-data

# Check the .benchmarks folder
ls -la .benchmarks/<platform>/
```

### Issue: "Cannot compare - no matching cases"

**Cause**: Test names changed between runs, or postfix configuration doesn't match.

**Solution**:

```bash
# Use correct postfix configuration
pytest-park analyze ./.benchmarks --original-postfix _original --reference-postfix _optimized

# Or check test names are consistent by inspecting the benchmarks folder
ls -la .benchmarks/
```

### Issue: Delta shows 0% but code changed

**Cause**:

- Cache effects (OS or Python-level caching)
- Test is too fast (below timer resolution)
- Not actually calling the optimized variant

**Solution**:

```bash
# Increase work size to make timing differences visible
# Use more rounds/iterations
benchmark.pedantic(func, rounds=100, iterations=1000)

# Verify correct function is called
print(f"Testing: {func.__name__}")
```

### Issue: Highly variable results between runs

**Cause**: System load, background processes, or insufficient rounds.

**Solution**:

```bash
# Close other applications
# Use more rounds for stability
benchmark.pedantic(func, rounds=50, iterations=100)

# Check system load
top  # Ensure CPU is mostly idle
```

### Issue: Can't group by custom metadata

**Cause**: `extra_info["custom_groups"]` not set, or key name typo.

**Solution**:

```python
# Ensure custom_groups is a dictionary
benchmark.extra_info["custom_groups"] = {
    "technique": "vectorization"  # Not "tecnique" typo!
}
```

Then: `pytest-park analyze ./.benchmarks --group-by custom:technique`

---

## Decision Tree for AI Assistants

When a user asks about benchmarking function improvements:

```
1. Does the user have pytest-benchmark installed?
   └─ No → Guide them through installation
   └─ Yes → Continue

2. Do they have existing benchmark tests?
   └─ No → Help them write baseline tests with naming conventions
   └─ Yes → Continue

3. Have they captured any benchmark runs yet?
   └─ No → Guide them to run: pytest --benchmark-only --benchmark-save=<tag> --benchmark-save-data
   └─ Yes → Continue

4. What do they want to do?
   ├─ Compare two specific runs → Use: pytest-park analyze --reference <A> --candidate <B>
   ├─ Compare latest vs previous → Use: pytest-park analyze (no args)
   ├─ Explore interactively → Use: pytest-park serve
   ├─ Group by technique/metadata → Use: pytest-park analyze --group-by custom:<key>
   ├─ Group by parameters → Use: pytest-park analyze --group-by param:<name>
   └─ Exclude a parameter → Use: pytest-park analyze --exclude-param <name>

5. Are results unclear or unexpected?
   ├─ Check naming conventions match postfix configuration
   ├─ Verify benchmark data was saved with --benchmark-save-data
   ├─ Inspect .benchmarks/ folder directly (ls -la .benchmarks/)
   └─ Check for system variability or insufficient rounds
```

---

## Summary

pytest-park enables systematic tracking of function performance improvements across time by:

1. **Organizing** pytest-benchmark results with intelligent name parsing
2. **Comparing** runs with flexible grouping and filtering
3. **Analyzing** improvements, regressions, and trends
4. **Visualizing** results through CLI summaries and interactive dashboards

**Success Pattern**:

1. Write baseline tests with clear naming
2. Capture baseline run with meaningful tag
3. Make code improvements
4. Capture optimized run with meaningful tag
5. Compare runs and interpret results
6. Iterate and track progress over time
