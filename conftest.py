# Pre-import the real `lightning` package before pytest adds `tests/` to sys.path,
# which would cause `tests/lightning/` to shadow it.
import lightning  # noqa: F401
