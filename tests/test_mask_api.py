from pytorchcocotools._maskApi import (
    rleFrString,
    rleToString,
)
import torch


def test_rleToString():  # noqa: N802
    # create a tensor
    tensor1 = torch.Tensor([0, 20, 40, 10, 72, 6])
    # convert the tensor to a string
    string1 = rleToString(tensor1)
    tensor2 = rleFrString(string1)
    # compare the results
    assert torch.equal(tensor1, tensor2)
    assert string1 == b"022O31MO31MO00010O00010O00010O00010O00010O00010O31MO3"
