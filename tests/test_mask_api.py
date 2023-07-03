from pytorchcocotools._maskApi import b, rleFrString, rleToString, stringToRLE, stringToRLE_
import torch


# write a test for rleToString
def test_rleToString():  # noqa: N802
    # create a tensor
    tensor = torch.Tensor([0, 100, 5, 10, 3, 1])
    # convert the tensor to a string
    string1 = rleToString(tensor)
    tensor2 = b(string1)
    tensor2 = rleFrString(string1)
    tensor2 = stringToRLE(string1)
    # compare the results
    assert tensor == tensor2
    assert string1 == b"\x06\x0c\x06\x06\x06\x0c"
