import torch
from torch import Tensor

from pytorchcocotools.internal.entities import RLE, RLEs, TorchDevice


def rleMerge(  # noqa: N802
    rles: RLEs,
    intersect: bool,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RLE:
    """Compute union or intersection of encoded masks.

    Args:
        rles: The masks to merge.
        intersect: Whether to compute the intersection.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The merged mask.
    """
    n = len(rles)
    if not rles or n == 0:
        return RLE(0, 0, Tensor())  # Return an empty RLE if empty
    if n == 1:
        return rles[0]  # Return the RLE if only one is provided

    # Rs[0].cnts.device
    # h, w = Rs[0].h, Rs[0].w

    # se_inds = torch.cat([torch.cumsum(r.cnts[:-1], 0).reshape(-1, 2) for r in Rs])
    # seq_lens = (se_inds[:, 1] - se_inds[:, 0]).unsqueeze(-1)
    # max_len = torch.max(seq_lens)
    # d = torch.arange(0, max_len, device=device).unsqueeze(0)  # +1 maxlen?
    # d = d.expand(seq_lens.size(0), d.size(1))
    # d_mask = d <= seq_lens

    # inds = d + se_inds[:, 0].unsqueeze(1)
    # selected_inds = inds[d_mask]
    # unq_inds, counts = torch.unique(selected_inds, sorted=True, return_counts=True)
    # id_mask = counts >= (n if intersect else 1)
    # merged_ids = unq_inds[id_mask]

    # zero = torch.zeros((1,), device=device)
    # rle_max = torch.tensor([h * w], device=device)
    # diffs = torch.diff(merged_ids, prepend=zero, append=rle_max)

    # one_indices = torch.nonzero(diffs == 1).view(-1)
    # one_ind_diffs = torch.diff(one_indices)

    # start_indices = one_indices[:-1][torch.diff(one_indices) > 1]
    # end_indices = one_indices[1:][torch.diff(one_indices) > 1] - 1

    # # Handle the case when the last element is 1
    # if diffs[-1] == 1:
    #     start_indices = torch.cat([start_indices, one_indices[-1].unsqueeze(0)])
    #     end_indices = torch.cat([end_indices, one_indices[-1].unsqueeze(0)])

    # run_ends = torch.nonzero(diffs != 1, as_tuple=True)[0] + 1

    # # Create tensors to store run lengths and values
    # run_lengths = torch.cat([run_ends.new_tensor([run_ends[0]]), run_ends[1:] - run_ends[:-1]])
    # run_values = merged_ids[run_ends]

    # [skip,used,skip,used,skip]

    # return run_values, run_lengths

    h, w = rles[0].h, rles[0].w
    # m = len(rles[0].cnts)
    cnts = rles[0].cnts.clone()
    for i in range(1, n):
        B = rles[i]  # noqa: N806
        if B.h != h or B.w != w:
            return RLE(0, 0, torch.tensor([]))  # Return an empty RLE if dimensions don't match

        A = RLE(h, w, cnts)  # noqa: N806
        ca = A.cnts[0].clone()
        cb = B.cnts[0].clone()
        v = False
        va = False
        vb = False
        a = 1
        b = 1
        cc = 0
        ct = 1
        cnts_out = []
        while ct > 0:
            c = torch.min(ca, cb).clone()
            cc += c
            ct = 0
            ca -= c
            if not ca and a < len(A.cnts):  # m
                ca = A.cnts[a].clone()
                a += 1
                va = not va
            cb -= c
            if not cb and b < len(B.cnts):  # m
                cb = B.cnts[b].clone()
                b += 1
                vb = not vb
            ct += ca
            ct += cb
            vp = v
            v = va and vb if intersect else va or vb
            if v != vp or ct == 0:
                cnts_out.append(cc)
                cc = 0
        cnts = torch.stack(cnts_out)

    return RLE(h, w, cnts)
