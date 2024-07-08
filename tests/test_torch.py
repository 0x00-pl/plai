import torch


def test_add():
    a = torch.tensor(1)
    b = torch.tensor(1)
    c = a + b
    assert c == 2
