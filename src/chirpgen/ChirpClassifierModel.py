import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class ChirpClassifierModel(nn.Module):
  def __init__(self, num_in_feat):
    super().__init__()
    self.num_in_feat = num_in_feat
    print(f"num_in_feat: {self.num_in_feat}")
    self.num_hidden_feat = self.num_in_feat + int(self.num_in_feat * 0.5)
    print(f"num_hidden_feat: {self.num_hidden_feat}")
    self.layer1 = nn.Linear(in_features=self.num_in_feat, out_features=self.num_hidden_feat)
    self.layer2 = nn.Linear(in_features=self.num_hidden_feat, out_features=1)

  def forward(self, batch_iq_tensors):
    return self.layer2(self.layer1(batch_iq_tensors))

