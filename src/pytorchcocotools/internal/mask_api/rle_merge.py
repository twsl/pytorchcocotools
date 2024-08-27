import torch
from torch import Tensor

from pytorchcocotools.internal.entities import BB, RLE, Mask, RLEs


def rleMerge(Rs: RLEs, n: int, intersect: bool) -> RLE:  # noqa: N802, N803
    """Compute union or intersection of encoded masks.

    Args:
        Rs: _description_
        n: _description_
        intersect: _description_

    Returns:
        _description_
    """
    if not Rs or len({r.w for r in Rs}) != 1 or len({r.w for r in Rs}) != 1:
        return RLE()  # Return an empty RLE if empty or dimensions don't match
    n = len(Rs)
    if n == 1:
        return Rs[0]

    # Rs[0].cnts.device
    h, w = Rs[0].h, Rs[0].w

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

    h, w, m = Rs[0].h, Rs[0].w, Rs[0].m
    cnts = Rs[0].cnts.clone()
    for i in range(1, n):
        B = Rs[i]  # noqa: N806
        if B.h != h or B.w != w:
            return RLE()  # Return an empty RLE if dimensions don't match

        A = RLE(h, w, m, cnts)  # noqa: N806
        ca = A.cnts[0].clone()
        cb = B.cnts[0].clone()
        v = False
        va = False
        vb = False
        a = 1
        b = 1
        cc = 0
        ct = 1
        cnts = []
        while ct > 0:
            c = torch.min(ca, cb).clone()
            cc += c
            ct = 0
            ca -= c
            if not ca and a < A.m:
                ca = A.cnts[a].clone()
                a += 1
                va = not va
            ct += ca
            cb -= c
            if not cb and b < B.m:
                cb = B.cnts[b].clone()
                b += 1
                vb = not vb
            ct += cb
            vp = v
            v = va and vb if intersect else va or vb
            if v != vp or ct == 0:
                cnts.append(cc)
                cc = 0

    return RLE(h, w, len(cnts), torch.stack(cnts))
