"""Modal GPU Test Runner.

Run tests on Modal's GPU infrastructure with pytest interface similar to local execution.

Usage:
    modal run .modal/test_runner.py --test-path tests/ --pytest-args "-v"
    modal run .modal/test_runner.py --test-path tests/datasets/ --pytest-args "-v -k synthetic"

Environment Variables:
    MODAL_GPU: GPU type to use (default: L4, options: L4, T4, A10G, A100, etc.)

Based on https://github.com/Borda/affordable-GPU-CI/blob/main/.modal/test_runner.py
"""

import os
from pathlib import Path
import sys

import modal

# Create Modal app with a fixed, descriptive name
app = modal.App("ci-gpu-tests")

WORKING_DIR = "/root/project"

# Get GPU type from environment or default to L4
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")
PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.11")

# Note: copy=True is required when running build commands after add_local_dir
image = (
    modal.Image.from_registry(  # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
        "nvcr.io/nvidia/pytorch:26.02-py3",
        add_python=PYTHON_VERSION,
    )
    .env(
        {
            "UV_SYSTEM_PYTHON": "1",
            "UV_DYNAMIC_VERSIONING_BYPASS": "1.0.0",
            "UV_PROJECT_ENVIRONMENT": "/usr/local",
        }
    )
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        "./src",
        remote_path=f"{WORKING_DIR}/src",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir(
        "./tests",
        remote_path=f"{WORKING_DIR}/tests",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_file(
        "pyproject.toml",
        remote_path=f"{WORKING_DIR}/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "uv.lock",
        remote_path=f"{WORKING_DIR}/uv.lock",
        copy=True,
    )
    .add_local_file(
        "README.md",
        remote_path=f"{WORKING_DIR}/README.md",
        copy=True,
    )
    # Test assets (if needed by tests, otherwise can be removed)
    .add_local_file(
        "./data/example.json",
        remote_path=f"{WORKING_DIR}/data/example.json",
        copy=True,
    )
    .add_local_dir(
        "./data/coco",
        remote_path=f"{WORKING_DIR}/data/coco",
        copy=True,
    )
    .workdir(WORKING_DIR)
    # Install dependencies during image build for faster execution
    .run_commands("uv sync --all-groups --all-extras --frozen")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,  # GPU type from environment variable
    timeout=3600,  # Hard 1 hour timeout safety limit
)
def run_tests(test_path: str = "tests/", pytest_args: str = "-v") -> dict[str, object]:
    """Run pytest on Modal GPU infrastructure."""
    import os
    import subprocess

    # Change to project directory
    os.chdir(WORKING_DIR)

    # Verify GPU availability
    try:
        import torch

        gpu_info = f"\n{'=' * 80}\n"
        "GPU ENVIRONMENT CHECK\n{'=' * 80}\n"
        "🎮 GPU Available: {torch.cuda.is_available()}\n"

        if torch.cuda.is_available():
            gpu_info += (
                f"🎮 GPU Device: {torch.cuda.get_device_name(0)}\n"
                f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n"
            )
        gpu_info += f"{'=' * 80}\n"
        print(gpu_info)
    except Exception as e:
        print(f"\n{'=' * 80}\nGPU ENVIRONMENT CHECK\n{'=' * 80}\n⚠️  GPU check failed: {e}\n{'=' * 80}\n")

    # Build pytest command
    pytest_cmd = ["pytest", test_path]

    # Add user-provided pytest arguments
    if pytest_args:
        # Split args properly, handling quoted strings
        import shlex

        pytest_cmd.extend(shlex.split(pytest_args))

    # Disable colored output for clean logs (especially for CI)
    pytest_cmd.append("--color=no")

    print(
        f"{'=' * 80}\n"
        f"RUNNING TESTS\n"
        f"{'=' * 80}\n"
        f"Command: {' '.join(pytest_cmd)}\n"
        f"Working directory: {Path.cwd()}\n"
        f"{'=' * 80}\n"
    )

    # Create output directory for pytest logs
    output_dir = Path(WORKING_DIR) / "test-outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "pytest-output.log"

    # Run pytest and stream output to both console and file
    try:
        output_lines: list[str] = []
        with Path(output_file).open("w") as log_file:
            process = subprocess.Popen(  # noqa: S603
                pytest_cmd,
                cwd=WORKING_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output line by line in real time while persisting it.
            if process.stdout is None:
                raise RuntimeError("Failed to capture pytest output stream.")  # noqa: TRY301
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                output_lines.append(line)
            process.wait()

        print(
            f"\n{'=' * 80}\n"
            f"TEST EXECUTION COMPLETE\n"
            f"{'=' * 80}\n"
            f"Exit code: {process.returncode}\n"
            f"📄 Test output saved to: {output_file}\n"
            f"{'=' * 80}\n"
        )

        return {
            "returncode": process.returncode,
            "success": process.returncode == 0,
            "pytest_output": "".join(output_lines),
            "output_file": str(output_file),
        }

    except Exception as e:
        print(f"\n{'=' * 80}\nERROR DURING TEST EXECUTION\n{'=' * 80}\nError: {e}\n{'=' * 80}\n")

        return {
            "returncode": 1,
            "success": False,
            "error": str(e),
            "pytest_output": "",
        }


@app.local_entrypoint()
def main(
    test_path: str = "tests/",
    pytest_args: str = "-v",
) -> None:
    """Local entrypoint to run tests on Modal GPU."""
    print(
        f"\n{'=' * 80}\n"
        f"GPU TEST RUNNER\n"
        f"{'=' * 80}\n"
        f"📁 Test Path: {test_path}\n"
        f"⚙️  Pytest Args: {pytest_args}\n"
        f"🎮 GPU: {GPU_TYPE}\n"
        f"⏱️  Timeout: 1 hour\n"
        f"{'=' * 80}\n"
    )

    # Run tests remotely and collect output after completion
    result = run_tests.remote(test_path=test_path, pytest_args=pytest_args)

    # Save output to local file
    local_output_dir = Path("test-outputs")
    local_output_dir.mkdir(exist_ok=True)
    local_output_file = local_output_dir / "pytest-output.log"

    if result.get("pytest_output"):
        with Path(local_output_file).open("w") as f:
            f.write(str(result["pytest_output"]))
        print(f"📄 Test output saved to: {local_output_file}")

    final_status = (
        f"\n{'=' * 80}\nFINAL RESULTS\n{'=' * 80}\nReturn Code: {result['returncode']}\nSuccess: {result['success']}\n"
    )

    if not result["success"]:
        if "error" in result:
            final_status += f"Error: {result['error']}\n"
        final_status += f"{'=' * 80}\n"
        print(final_status)
        sys.exit(str(result["returncode"]))

    final_status += f"{'=' * 80}\n\n✅ All tests passed!"
    print(final_status)
