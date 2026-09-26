import pytest, torch
from model import SmallCNN
from api import PredictIn

def test_output_shape():
    assert SmallCNN()(torch.zeros(2, 1, 28, 28)).shape == (2, 10)

def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        PredictIn(image=[[0.0] * 28] * 27)

def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        PredictIn(image=[[2.0] * 28] * 28)