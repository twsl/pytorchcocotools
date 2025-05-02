from collections.abc import Callable, Generator
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

from line_profiler import LineProfiler
import torch
from torch.autograd import ProfilerActivity  # pyright: ignore[reportPrivateImportUsage]
from torch.profiler import profile as TorchProfiler  # noqa: N812

ExperimentalConfigType = object | None


class CombinedProfiler:
    """A class that combines PyTorch profiler and line_profiler capabilities."""

    def __init__(
        self,
        functions_to_profile: list[Callable],
        activities: list[ProfilerActivity] | None = None,
        schedule_instance: Callable | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = True,
        with_flops: bool = False,
        with_modules: bool = True,
        experimental_config: ExperimentalConfigType = None,
        output_dir: Path | str | None = None,
        on_trace_ready: Callable | None = None,
        print_summary: bool = True,
        summary_sort_by: str = "cpu_time_total",
        summary_row_limit: int = 10,
        print_line_stats: bool = True,
        output_unit: float = 1e-6,
        stripzeros: bool = False,
    ):
        """Initialize the combined profiler.

        Args:
            functions_to_profile: List of functions to profile line-by-line (required).
            activities: List of activities to profile (e.g., [ProfilerActivity.CPU, ProfilerActivity.CUDA]).
            schedule_instance: A schedule instance that controls the profiler state.
            on_trace_ready: Callback function to handle the trace results.
            record_shapes: Whether to record tensor shapes.
            profile_memory: Whether to profile memory usage.
            with_stack: Whether to record call stacks.
            with_flops: Whether to estimate FLOPs.
            with_modules: Whether to record module hierarchy.
            experimental_config: Experimental profiler configuration.
            output_dir: Directory to save profiling results (e.g., for TensorBoard). Can be str or Path.
            output_filename: Filename for exported JSON trace if output_dir is not specified for TensorBoard.
            print_summary: Whether to print a summary table to the console.
            summary_sort_by: Key to sort the summary table by (e.g., "cpu_time_total", "cuda_time_total").
            summary_row_limit: Number of rows to show in the summary table.
            print_line_stats: Whether to print the line-by-line profiling statistics to stdout.
            output_unit: Unit for timing measurements in the output (default: 1e-6 for microseconds).
            stripzeros: Whether to remove lines with zero hits from the output.
        """
        self.logger = logging.getLogger(__name__)
        self.functions_to_profile = functions_to_profile
        self.activities = self._initialize_activities(activities)
        self.schedule_instance = schedule_instance
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.experimental_config = experimental_config
        self.output_dir = Path(output_dir) if output_dir else Path("../logs")
        self.on_trace_ready = self._initialize_trace(on_trace_ready)
        self.print_summary = print_summary
        self.summary_sort_by = summary_sort_by
        self.summary_row_limit = summary_row_limit
        self.print_line_stats = print_line_stats
        self.output_unit = output_unit
        self.stripzeros = stripzeros
        self.line_profiler = self._initialize_line_profiler()
        self.profiler_args = self._initialize_profiler_args()

    def _initialize_line_profiler(self) -> LineProfiler:
        """Initialize the line profiler with the provided functions."""
        lp = LineProfiler()
        for func in self.functions_to_profile:
            lp.add_function(func)
        return lp

    def _initialize_activities(self, activities) -> list[ProfilerActivity]:
        """Initialize the list of activities to profile."""
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            return activities
        return activities

    def _initialize_trace(self, on_trace_ready: Callable | None) -> Callable[..., None]:
        """Initialize the output directory and trace handler."""
        if on_trace_ready is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            on_trace_ready = torch.profiler.tensorboard_trace_handler(str(self.output_dir.resolve()))
            self.logger.info(f"Profiling results will be saved for TensorBoard in: {self.output_dir.resolve()}")

        return on_trace_ready

    def _initialize_profiler_args(self) -> dict:
        """Initialize arguments for torch.profiler.profile."""
        profiler_args = {
            "activities": self.activities,
            "schedule": self.schedule_instance,
            "on_trace_ready": self.on_trace_ready,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
            "with_flops": self.with_flops,
            "with_modules": self.with_modules,
        }

        if self.experimental_config is not None:
            if isinstance(self.experimental_config, object):
                profiler_args["experimental_config"] = self.experimental_config
            else:
                self.logger.warning("experimental_config provided but type is unknown or invalid, ignoring.")

        return profiler_args

    def _finalize_profiling(self, prof: TorchProfiler) -> None:
        """Finalize profiling by printing summaries and exporting traces."""
        if self.print_summary:
            self.logger.info(f"PyTorch Profiler Summary (sorted by {self.summary_sort_by}):")
            try:
                summary = prof.key_averages().table(sort_by=self.summary_sort_by, row_limit=self.summary_row_limit)
                print(summary)
            except Exception:
                self.logger.exception("Failed to generate or print profiler summary")

        if self.line_profiler and self.print_line_stats:
            self.line_profiler.print_stats(output_unit=self.output_unit, stripzeros=self.stripzeros)

    @contextmanager
    def profile(self) -> Generator[TorchProfiler, Any, None]:
        """Context manager for profiling PyTorch code execution and line-by-line profiling."""
        with TorchProfiler(**self.profiler_args) as prof:
            self.line_profiler.enable_by_count()
            try:
                yield prof
            finally:
                self.line_profiler.disable_by_count()

        self._finalize_profiling(prof)
