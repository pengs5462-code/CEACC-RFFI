import torch

from model.MRAN import NetTime


def test_nettime_shapes():
    model = NetTime(c_in=2, c_out=9, nf=16).eval()
    with torch.no_grad():
        embedding, logits = model(torch.randn(2, 2, 4800))
    assert embedding.shape == (2, 64)
    assert logits.shape == (2, 9)
