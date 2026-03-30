import torch
from torchvision import tv_tensors as tv
from torchvision.transforms.v2 import functional as f

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


@torch.inference_mode()
def rleFrBbox(  # noqa: N802
    bb: tv.BoundingBoxes,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool = False,
) -> RLEs:
    """Convert bounding boxes to encoded masks.

    Computes column-major RLE directly from bbox coordinates without polygon
    rasterization.  All arithmetic is vectorised over the batch; the final
    loop only slices pre-computed tensors to build RLE objects.

    Args:
        bb: The bounding boxes.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The RLE encoded masks.
    """
    bb = tv.wrap(f.convert_bounding_box_format(bb, new_format=tv.BoundingBoxFormat.XYWH), like=bb)
    h, w = bb.canvas_size
    n = bb.shape[0]
    if n == 0:
        return []
    src_device = bb.device

    bb_int = bb.round().long()
    x_col, y_row, w_box, h_box = bb_int[:, 0], bb_int[:, 1], bb_int[:, 2], bb_int[:, 3]

    # --- vectorised arithmetic (no per-box Python work) ---
    bg0 = x_col * h + y_row
    bg_inner = h - h_box
    bg_last = (h - y_row - h_box) + (w - x_col - w_box) * h
    total_fg = h_box * w_box
    valid = (w_box > 0) & (h_box > 0)
    full_h = valid & (bg_inner == 0)

    max_w = int(w_box.clamp(min=1).max().item())
    max_runs = 2 * max_w + 1
    hw = h * w

    # Unified [n, max_runs] run matrix via broadcasting
    j = torch.arange(max_runs, device=src_device)  # [R]
    last_j = (2 * w_box).unsqueeze(1)  # [n, 1]
    padded = torch.where(
        j == 0,
        bg0.unsqueeze(1),
        torch.where(
            j % 2 == 1,
            h_box.unsqueeze(1),
            torch.where(j == last_j, bg_last.unsqueeze(1), bg_inner.unsqueeze(1)),
        ),
    ).to(torch.int32)  # [n, max_runs]

    # Per-row valid lengths (normal case)
    run_lens = 2 * w_box + 1

    # Full-height rows: merge fg runs → [bg0, total_fg, bg_last?]
    if full_h.any():
        padded[full_h, 1] = total_fg[full_h].to(torch.int32)
        padded[full_h, 2] = bg_last[full_h].to(torch.int32)
        run_lens = torch.where(full_h, torch.where(bg_last > 0, 3, 2), run_lens)

    # Invalid rows: [h*w]
    inv = ~valid
    if inv.any():
        padded[inv, 0] = hw
        run_lens = torch.where(inv, 1, run_lens)

    # Branchless loop: one slice + constructor per row
    run_lens_l = run_lens.tolist()
    results: RLEs = []
    for i in range(n):
        results.append(RLE(h=h, w=w, cnts=padded[i, : run_lens_l[i]]))
    return results
