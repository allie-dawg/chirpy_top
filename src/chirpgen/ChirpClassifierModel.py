import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class ChirpClassifierModel(nn.Module):
  # A sample is a pair of I and Q data points. They have been converted from complex data type to real for
  # for simplicity of trianing. Normally I would consider a pair of I and Q samples as one sample.
  # But this neural network is simplified when converting the I and Q to real data types. 
  def __init__(self, num_in_samples):
    super().__init__()
    self.num_in_feats = num_in_samples * 2
    # self.num_hidden_feats = self.num_in_feat + int(self.num_in_feat * 0.5)
    self.num_hidden_feats = 1024
    self.layer1 = nn.Linear(in_features=self.num_in_feats, out_features=self.num_hidden_feats, dtype=torch.float32)
    self.layer2 = nn.Linear(in_features=self.num_hidden_feats, out_features=1, dtype=torch.float32)

  def forward(self, batch_iq_tensors):
    batch_sz = len(batch_iq_tensors)
    batch_iq_tensors = torch.reshape(batch_iq_tensors, shape=(batch_sz, batch_iq_tensors.shape[1] * batch_iq_tensors.shape[2]) )
    return self.layer2(self.layer1(batch_iq_tensors))

