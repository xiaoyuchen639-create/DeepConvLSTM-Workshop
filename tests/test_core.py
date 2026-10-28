import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from data import make_windows
from preprocess import fill_missing,sensor_columns,split_recordings
from models import DeepConvLSTM,FocalLoss

def test_windows_use_endpoint_and_do_not_pad():
    x=np.arange(20).reshape(10,2); y=np.arange(10)
    wx,wy=make_windows(x,y,4,3)
    assert wy.tolist()==[3,6,9]
    np.testing.assert_array_equal(wx[1],x[3:7])

def test_disjoint_recordings_and_channels():
    s=split_recordings(); assert len(set(sum(s.values(),[])))==18
    assert len(sensor_columns())==113
    assert 249 not in sensor_columns()

def test_missing_values():
    x=np.array([[np.nan,np.nan],[2,np.nan],[np.nan,np.nan],[4,np.nan]])
    np.testing.assert_allclose(fill_missing(x)[:,0],[0,2,3,4])
    assert np.isfinite(fill_missing(x)).all()

@pytest.mark.parametrize('options',[{}, {'model':'cnn'},{'rnn':'gru'},{'n_conv':0},{'n_conv':1},{'window':20},{'window':60}])
def test_network_shapes_and_gradients(options):
    m=DeepConvLSTM(**options); x=torch.rand(2,options.get('window',24),113)
    logits=m(x); assert logits.shape==(2,18)
    FocalLoss()(logits,torch.tensor([0,17])).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_focal_gamma_zero_equals_cross_entropy():
    x=torch.tensor([[2.,-1.],[0.,3.]],requires_grad=True); y=torch.tensor([0,1])
    torch.testing.assert_close(FocalLoss(0)(x,y),torch.nn.functional.cross_entropy(x,y))
