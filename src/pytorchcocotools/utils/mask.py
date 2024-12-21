from jaxtyping import Num
from torch import Tensor


def reorder_mask_tensor_orig(
    x: Num[Tensor, "N H W"] | Num[Tensor, "B N H W"],
) -> Num[Tensor, "H W N"] | Num[Tensor, "B H W N"]:
    """Reorders a PyTorch mask tensor so that the format matches og implementation.

    - (N, H, W) becomes (H, W, N)
    - (B, N, H, W) becomes (B, H, W, N)

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The reordered tensor.
    """
    # Check the number of dimensions in x
    dims = x.dim()

    if dims == 3:
        # Shape is (N, H, W) -> (H, W, N)
        return x.permute(1, 2, 0)
    elif dims == 4:
        # Shape is (B, N, H, W) -> (B, H, W, N)
        return x.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported tensor shape with {dims} dimensions.")


def reorder_mask_tensor_pytorch(x: Tensor) -> Tensor:
    """Reorders a PyTorch mask tensor so that the format matches og implementation.

    - (H, W, N) becomes (N, H, W)
    - (B, H, W, N) becomes (B, N, H, W)

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The reordered tensor.
    """
    # Check the number of dimensions in x
    dims = x.dim()

    if dims == 3:
        # Shape is (H, W, N) -> (N, H, W)
        return x.permute(2, 0, 1)
    elif dims == 4:
        # Shape is (B, H, W, N) -> (B, N, H, W)
        return x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported tensor shape with {dims} dimensions.")
