from datagen import datagen
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence

plot = False

def unify_vectors_to_same_size(input_list):
  print(f"Length of input list: {len(input_list)}")
  padded_tensor = pad_sequence(input_list, batch_first=True, padding_value=0)
  print(f"Shape of padded tensor {padded_tensor.shape}")
  print(f"Length of padded tensor (dim 0): {len(padded_tensor)}")
  return padded_tensor

data_set_sz = 10
filename = datagen(data_set_sz, low_snr=60, high_snr=100)

file = torch.load(f"./{filename}", weights_only=False)
iq_tensors_list = file["iq_tensors"]
labels_list = file["labels"]
has_signal_labels = torch.tensor([item["no_signal"] for item in labels_list], dtype=torch.float32)
print(f"Type of has_signal_labels: {type(has_signal_labels)}")
batch_of_iq_vectors = unify_vectors_to_same_size(iq_tensors_list)
print(has_signal_labels.shape)
fs, snr, duration, f0, f1, no_signal = labels_list[0].values()
print(f"SNR: {snr}")
print(f"No signal?: {no_signal}")

if (plot):
  t = np.linspace(0, duration, int(duration * fs), endpoint=False)
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(t, iq_tensors_list[0].real, label="I set", color="tab:blue")
  ax.plot(t, iq_tensors_list[0].imag, label="Q set", color="tab:orange")
  
  ax.set_title("I and Q", fontsize=4)
  ax.set_xlabel("time")
  ax.set_ylabel("IQ tensor data")
  ax.grid(True, linestyle="--", alpha=0.6)
  ax.set_xlim(0, 50e-9)
  
  plt.show()
