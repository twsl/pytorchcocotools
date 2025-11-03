from collections import defaultdict
import operator
import re

import pytest
from pytest import FixtureRequest
import torch


@pytest.fixture(
    scope="session",
    params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def device(request: FixtureRequest) -> str:
    """Return the device to use for the tests (parametrized)."""
    return request.param


def pytest_benchmark_group_stats(config, benchmarks, group_by):
    """Custom grouping for pytest-benchmark.

    Groups benchmarks by clustering methods ending with _np and _pt together,
    while ignoring the 'device' parameter in the grouping logic.
    """
    groups = defaultdict(list)

    for bench in benchmarks:
        key = []

        # Extract the test function name without parameters
        fullname = bench["fullname"]
        name = bench["name"]

        # Split the name to get the base function name and parameters
        # e.g., "test_evaluate_np[eval_keypoints-test_1-cpu]" -> "test_evaluate", "_np", "eval_keypoints-test_1-cpu"
        match = re.match(r"^(.+?)(_np|_pt)(\[.*\])?$", name.split("[")[0])

        if match:
            # If the test name ends with _np or _pt, extract base name
            base_name = match.group(1)
            # suffix = match.group(2)  # _np or _pt - not used but kept for clarity

            # Build grouping key based on group_by parameter
            for grouping in group_by.split(","):
                if grouping == "group":
                    key.append(bench["group"])
                elif grouping == "name":
                    # Use base name (without _np/_pt suffix) for clustering
                    key.append(base_name)
                elif grouping == "func":
                    # Use base function name
                    key.append(base_name)
                elif grouping == "fullname":
                    # Use full path but with base name
                    fullname_base = fullname.split("::")[0] + "::" + base_name
                    if "[" in fullname:
                        fullname_base += fullname[fullname.index("["):]
                    key.append(fullname_base)
                elif grouping == "fullfunc":
                    # Use full path with function name only
                    fullname_base = fullname.split("::")[0] + "::" + base_name
                    key.append(fullname_base)
                elif grouping == "param":
                    # Get parameter string but exclude device parameter
                    param_str = bench["param"]
                    # Remove device parameter if present
                    param_parts = param_str.split("-")
                    filtered_parts = [p for p in param_parts if p not in ["cpu", "cuda"]]
                    key.append("-".join(filtered_parts) if filtered_parts else param_str)
                elif grouping.startswith("param:"):
                    param_name = grouping[len("param:"):]
                    # Ignore 'device' parameter
                    if param_name != "device" and param_name in bench["params"]:
                        key.append(f"{param_name}={bench['params'][param_name]}")
                else:
                    raise NotImplementedError(f"Unsupported grouping {group_by!r}.")
        else:
            # Fallback to default grouping for tests not matching the pattern
            for grouping in group_by.split(","):
                if grouping == "group":
                    key.append(bench["group"])
                elif grouping == "name" or grouping == "func":
                    key.append(name.split("[")[0])
                elif grouping == "fullname":
                    key.append(fullname)
                elif grouping == "fullfunc":
                    key.append(fullname.split("[")[0])
                elif grouping == "param":
                    param_str = bench["param"]
                    # Remove device parameter if present
                    param_parts = param_str.split("-")
                    filtered_parts = [p for p in param_parts if p not in ["cpu", "cuda"]]
                    key.append("-".join(filtered_parts) if filtered_parts else param_str)
                elif grouping.startswith("param:"):
                    param_name = grouping[len("param:"):]
                    # Ignore 'device' parameter
                    if param_name != "device" and param_name in bench["params"]:
                        key.append(f"{param_name}={bench['params'][param_name]}")
                else:
                    raise NotImplementedError(f"Unsupported grouping {group_by!r}.")

        # Convert key to string
        group_key = " ".join(str(p) for p in key if p) or None
        groups[group_key].append(bench)

    # Sort benchmarks within each group
    for grouped_benchmarks in groups.values():
        grouped_benchmarks.sort(key=operator.itemgetter("fullname" if "full" in group_by else "name"))

    return sorted(groups.items(), key=lambda pair: pair[0] or "")
