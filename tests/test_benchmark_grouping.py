"""Test custom pytest-benchmark grouping behavior."""
import sys
from pathlib import Path

# Add tests directory to path to import conftest
sys.path.insert(0, str(Path(__file__).parent))
from conftest import pytest_benchmark_group_stats  # noqa: E402


def test_benchmark_grouping_np_pt_clustering():
    """Test that _np and _pt methods are clustered together."""
    # Create mock benchmarks
    benchmarks = [
        {
            'fullname': 'tests/test_file.py::test_func_np[param1-param2]',
            'name': 'test_func_np[param1-param2]',
            'group': 'test_group',
            'param': 'param1-param2',
            'params': {'param1': 'value1', 'param2': 'value2'}
        },
        {
            'fullname': 'tests/test_file.py::test_func_pt[param1-param2]',
            'name': 'test_func_pt[param1-param2]',
            'group': 'test_group',
            'param': 'param1-param2',
            'params': {'param1': 'value1', 'param2': 'value2'}
        }
    ]
    
    # Test grouping by 'group'
    config = None
    result = pytest_benchmark_group_stats(config, benchmarks, 'group')
    
    # Both benchmarks should be in the same group
    assert len(result) == 1
    group_key, group_benchmarks = result[0]
    assert group_key == 'test_group'
    assert len(group_benchmarks) == 2


def test_benchmark_grouping_with_device_parameter():
    """Test that device parameter is ignored in grouping."""
    # Create mock benchmarks with device parameter
    benchmarks = [
        {
            'fullname': 'tests/test_file.py::test_func_np[cpu-param1]',
            'name': 'test_func_np[cpu-param1]',
            'group': 'test_group',
            'param': 'cpu-param1',
            'params': {'device': 'cpu', 'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_func_np[cuda-param1]',
            'name': 'test_func_np[cuda-param1]',
            'group': 'test_group',
            'param': 'cuda-param1',
            'params': {'device': 'cuda', 'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_func_pt[cpu-param1]',
            'name': 'test_func_pt[cpu-param1]',
            'group': 'test_group',
            'param': 'cpu-param1',
            'params': {'device': 'cpu', 'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_func_pt[cuda-param1]',
            'name': 'test_func_pt[cuda-param1]',
            'group': 'test_group',
            'param': 'cuda-param1',
            'params': {'device': 'cuda', 'param1': 'value1'}
        }
    ]
    
    # Test grouping by 'group,name'
    config = None
    result = pytest_benchmark_group_stats(config, benchmarks, 'group,name')
    
    # All benchmarks should be in the same group (ignoring device)
    assert len(result) == 1
    group_key, group_benchmarks = result[0]
    assert group_key == 'test_group test_func'
    assert len(group_benchmarks) == 4


def test_benchmark_grouping_by_name():
    """Test that _np and _pt are grouped by base name."""
    # Create mock benchmarks
    benchmarks = [
        {
            'fullname': 'tests/test_file.py::test_evaluate_np[param1]',
            'name': 'test_evaluate_np[param1]',
            'group': 'evaluate',
            'param': 'param1',
            'params': {'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_evaluate_pt[param1]',
            'name': 'test_evaluate_pt[param1]',
            'group': 'evaluate',
            'param': 'param1',
            'params': {'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_compute_np[param1]',
            'name': 'test_compute_np[param1]',
            'group': 'compute',
            'param': 'param1',
            'params': {'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_compute_pt[param1]',
            'name': 'test_compute_pt[param1]',
            'group': 'compute',
            'param': 'param1',
            'params': {'param1': 'value1'}
        }
    ]
    
    # Test grouping by 'name'
    config = None
    result = pytest_benchmark_group_stats(config, benchmarks, 'name')
    
    # Should have 2 groups: test_evaluate and test_compute
    assert len(result) == 2
    
    # Check first group
    group1_key, group1_benchmarks = result[0]
    assert group1_key == 'test_compute'
    assert len(group1_benchmarks) == 2
    
    # Check second group
    group2_key, group2_benchmarks = result[1]
    assert group2_key == 'test_evaluate'
    assert len(group2_benchmarks) == 2


def test_benchmark_grouping_fallback_for_non_matching():
    """Test fallback grouping for tests not matching _np/_pt pattern."""
    # Create mock benchmarks without _np/_pt suffix
    benchmarks = [
        {
            'fullname': 'tests/test_file.py::test_other[param1]',
            'name': 'test_other[param1]',
            'group': 'other',
            'param': 'param1',
            'params': {'param1': 'value1'}
        },
        {
            'fullname': 'tests/test_file.py::test_another[param2]',
            'name': 'test_another[param2]',
            'group': 'other',
            'param': 'param2',
            'params': {'param2': 'value2'}
        }
    ]
    
    # Test grouping by 'group'
    config = None
    result = pytest_benchmark_group_stats(config, benchmarks, 'group')
    
    # Both should be in the same group
    assert len(result) == 1
    group_key, group_benchmarks = result[0]
    assert group_key == 'other'
    assert len(group_benchmarks) == 2
