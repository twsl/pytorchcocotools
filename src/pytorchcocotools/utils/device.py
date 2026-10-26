import torch


def get_float_dtype(device: torch.device | str | int | None) -> torch.dtype:
    """Return float32 on MPS and float64 on other devices."""
    if device is not None and torch.device(device).type == "mps":
        return torch.float32
    return torch.float64
