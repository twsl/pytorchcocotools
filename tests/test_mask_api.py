import pycocotools._mask as _mask
from pytorchcocotools._maskApi import rle_from_string, rle_to_string, rleFrString, rleToString
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


def test_chatgpt():
    # Example usage:
    RLE = {"h": 0, "w": 0, "m": 0, "cnts": []}

    rle_from_string(RLE, b"123456789", 3, 3)
    print(RLE)

    s = rle_to_string(RLE)
    print(s)
