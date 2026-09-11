import torch

from neurofault.models.cnn import SimpleCNN


def test_forward_pass_shape():
    model = SimpleCNN()
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    out = model(x)

    assert out.shape == (2, 10)
