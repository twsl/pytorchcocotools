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

This skill teaches AI assistants how to help users track and analyze function performance improvements using pytest-park. The primary workflow runs entirely inside normal `pytest` invocations — no separate commands needed. The CLI and dashboard are supplementary tools for deeper historical analysis.

**Target Audience**: Python developers optimizing code performance
**Prerequisites**: pytest, pytest-benchmark installed
**When to Use**: Performance optimization, regression detection, improvement validation

---

## Primary Workflow: Inside the Test Suite

### Step 1 — Installation

```bash
# pip
pip install pytest-park pytest-benchmark

# uv (recommended)
uv add --group test pytest-park pytest-benchmark
```

### Step 2 — Write benchmark tests

```python
# tests/test_performance.py
def process_data_original(data):
    return sum(data)

def process_data_optimized(data):
    return sum(data)  # faster implementation

def test_process_data_original(benchmark):
    benchmark(process_data_original, list(range(1000)))

def test_process_data_optimized(benchmark):
    benchmark(process_data_optimized, list(range(1000)))
```

Naming convention: use consistent postfixes like `_original`/`_optimized` or `_orig`/`_ref` so pytest-park can pair variants together.

### Step 3 — Run tests

```bash
pytest
```

After the normal pytest-benchmark tables, a `pytest-park` section is printed automatically. It compares the current run against the latest saved benchmark artifact in pytest-benchmark storage. No extra flags are required.

> The plugin is registered automatically via the `pytest11` entry point when `pytest-park` is installed — no `conftest.py` changes are required.

### Step 4 — Build a history with `--benchmark-save` (optional but recommended)

```bash
# Rolling "compare against latest" workflow
pytest --benchmark-autosave

# Save a named baseline for a stable reference point
pytest --benchmark-save baseline

# Compare against a specific saved run by number or id prefix
pytest --benchmark-compare=0001
pytest --benchmark-compare=8d530304

# Save a new candidate and compare against a known baseline in one command
pytest --benchmark-save candidate-v2 --benchmark-compare=0001
```

pytest-park reuses the baseline resolved by pytest-benchmark from its configured storage. `--benchmark-storage` is respected as usual.

> **VS Code Test Explorer**: if the run looks like a single-shot execution (benchmark timing disabled or reduced), pytest-park prints a warning so the output is not mistaken for a real comparison.

---

## Optional Enhancements Inside the Test Suite

### Name normalization with postfixes

If test names encode variant postfixes (e.g. `test_func_orig`, `test_func_ref`), add the `pytest_benchmark_group_stats` hook to pair and label them in the comparison table:

```python
# tests/conftest.py
from pytest_park.pytest_benchmark import default_pytest_benchmark_group_stats

def pytest_benchmark_group_stats(config, benchmarks, group_by):
    return default_pytest_benchmark_group_stats(
        config,
        benchmarks,
        group_by,
        original_postfix="_orig",
        reference_postfix="_ref",
        group_values_by_postfix={
            "_orig": "original",
            "_ref": "reference",
        },
    )
```

This stores parsed parts in `extra_info["pytest_park_name_parts"]` (`base_name`, `parameters`, `postfix`) and groups paired variants under the same row.

To expose postfix options as pytest flags:

```python
def pytest_addoption(parser):
    parser.addoption("--benchmark-original-postfix", action="store", default="")
    parser.addoption("--benchmark-reference-postfix", action="store", default="")
```

### Custom grouping metadata

Store arbitrary metadata on a benchmark for richer grouping in the CLI:

```python
def test_compute_optimized(benchmark):
    benchmark.extra_info["custom_groups"] = {
        "technique": "vectorization",
        "scenario": "large-batch",
    }
    benchmark(compute)
```

Group by any key with `--group-by custom:technique` in the CLI.

---

## Interpreting Results

### Delta percentage

```
avg_delta=-50.0%   → 50% faster (2× speedup)
avg_delta=-80.0%   → 80% faster (5× speedup)
avg_delta=+100.0%  → 100% slower (regression!)
avg_delta=-0.01%   → Negligible change
```

Formula: `(candidate_mean - reference_mean) / reference_mean × 100`. Negative = faster.

### Speedup ratio

```
avg_speedup=2.0    → 2× faster
avg_speedup=0.5    → 2× slower (regression!)
```

Formula: `reference_mean / candidate_mean`.

### Counts

- `improved`: candidate faster than reference
- `regressed`: candidate slower than reference
- `unchanged`: absolute difference ≤ 1e-9

---

## CLI — Deeper Analysis Across Saved Artifacts

Use the CLI only when routine `pytest` output is not enough: comparing specific named runs, applying advanced grouping, including profiler data, or reviewing a longer history.

```bash
# Compare latest run against second-latest (default)
pytest-park analyze ./.benchmarks

# Compare named runs
pytest-park analyze ./.benchmarks --reference baseline --candidate candidate-v2

# When only --candidate is given, the preceding run is used as reference
pytest-park analyze ./.benchmarks --candidate candidate-v2

# Group by benchmark group and a specific parameter
pytest-park analyze ./.benchmarks --group-by group --group-by param:device

# Group by custom metadata key
pytest-park analyze ./.benchmarks --group-by custom:technique

# Exclude a parameter from comparison
pytest-park analyze ./.benchmarks --exclude-param device

# Treat a parameter as a separate dimension instead of collapsing it
pytest-park analyze ./.benchmarks --group-by group --distinct-param device

# Normalize method names by stripping postfixes
pytest-park analyze ./.benchmarks --original-postfix _orig --reference-postfix _ref

# Include profiler artifacts
pytest-park analyze ./.benchmarks --profiler-folder ./.profiler --group-by group

# Print installed version
pytest-park version
```

### Grouping reference

Default precedence when no `--group-by` is given: `custom > benchmark_group > marks > params`.

| Token          | Alias(es)         | Resolves to                            |
| -------------- | ----------------- | -------------------------------------- |
| `custom:<key>` | —                 | `extra_info["custom_groups"]["<key>"]` |
| `custom`       | `custom_group`    | All custom group keys combined         |
| `group`        | `benchmark_group` | Benchmark group label                  |
| `marks`        | `mark`            | Comma-joined pytest marks              |
| `params`       | —                 | All parameter key=value pairs          |
| `param:<name>` | —                 | Value of a specific parameter          |
| `name`         | `method`          | Normalized method name                 |
| `fullname`     | `nodeid`          | Full test node path                    |

Multiple `--group-by` tokens are combined; the label is joined with `|`.

Additional tokens available only in `pytest_benchmark_group_stats` (not the CLI):

| Token                           | Resolves to                                               |
| ------------------------------- | --------------------------------------------------------- |
| `func` / `fullfunc`             | Node path or base name                                    |
| `postfix` / `benchmark_postfix` | Parsed postfix label mapped via `group_values_by_postfix` |

### Artifact folder expectations

- Input files are pytest-benchmark JSON files (`--benchmark-save` output) stored anywhere under the folder.
- Default comparison: latest run as candidate, second-latest as reference.
- When only `--candidate` is given, the run immediately preceding it is used as reference.
- Run identity uses `metadata.run_id`, `metadata.tag`, or fallback datetime identifiers.

---

## Interactive Dashboard

For visual, exploratory analysis across many saved runs:

```bash
pytest-park serve ./.benchmarks --reference baseline --host 127.0.0.1 --port 8080

# With profiler data
pytest-park serve ./.benchmarks --profiler-folder ./.profiler --port 8080
```

Access at `http://127.0.0.1:8080`. Features include run selection, history charts, delta distribution, and method-level drill-down.

To launch a guided interactive CLI session:

```bash
pytest-park
```

---

## Troubleshooting

### "No runs found"

Benchmark folder is empty or data was not saved. Use `--benchmark-save-data` when saving:

```bash
pytest --benchmark-save=myrun --benchmark-save-data
ls -la .benchmarks/
```

### "Cannot compare — no matching cases"

Test names changed between runs, or postfix configuration does not match:

```bash
pytest-park analyze ./.benchmarks --original-postfix _orig --reference-postfix _ref
```

### Delta shows 0% but code changed

Common causes: cache effects, test is too fast (below timer resolution), or wrong function being called. Use more rounds:

```python
benchmark.pedantic(func, rounds=100, iterations=1000)
```

### Highly variable results

Close background applications and increase rounds:

```python
benchmark.pedantic(func, rounds=50, iterations=100)
```

### VS Code Test Explorer shows only a single execution

pytest-park prints a warning automatically. For a proper comparison, run from the terminal:

```bash
pytest --benchmark-compare
```

### Can't group by custom metadata

Ensure `extra_info["custom_groups"]` is set correctly (check for typos in the key):

```python
benchmark.extra_info["custom_groups"] = {"technique": "vectorization"}
```

---

## Decision Tree for AI Assistants

```
1. Is pytest-benchmark installed?
   └─ No → pip install pytest-park pytest-benchmark

2. What does the user need?
   ├─ Inline feedback while coding
   │   └─ Run: pytest  (plugin auto-loads after installation)
   ├─ Save runs for future comparison
   │   └─ pytest --benchmark-autosave  or  --benchmark-save <name>
   ├─ Compare two specific saved runs
   │   └─ pytest-park analyze --reference <A> --candidate <B>
   ├─ Compare latest vs previous artifact
   │   └─ pytest-park analyze (no args)
   ├─ Group / filter results
   │   └─ pytest-park analyze --group-by <token> --exclude-param <key>
   └─ Visual exploration
       └─ pytest-park serve ./.benchmarks --port 8080

3. Results unclear or unexpected?
   ├─ Check postfix configuration matches test names
   ├─ Verify data was saved with --benchmark-save-data
   ├─ Inspect .benchmarks/ (ls -la .benchmarks/)
   └─ Check for system variability / insufficient rounds
```

4. Iterate and track progress over time
