from datagen import datagen
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def unify_vectors_to_same_size(input_list):
  padded_tensor = pad_sequence(input_list, batch_first=True, padding_value=0)
  return padded_tensor

def get_and_preprocess_dataset(batch_sz, train_size, plot_first_vec=False):
  filename = datagen(batch_sz, low_snr=5, high_snr=100)
  file = torch.load(f"./{filename}", weights_only=False)
  iq_tensors_list = file["iq_tensors"]
  labels_list = file["labels"]
  batch_labels = torch.tensor([item["no_signal"] for item in labels_list], dtype=torch.float32)
  batch_iq_vectors = unify_vectors_to_same_size(iq_tensors_list)
  iq_vec_train, iq_vec_test, label_train, label_test = train_test_split(batch_iq_vectors, 
                                                                      batch_labels, 
                                                                      train_size=train_size,
                                                                      random_state=42)

  # Making sure linter understands that I am returning tensors from this function and not lists
  iq_vec_test = torch.tensor(iq_vec_test) if isinstance(iq_vec_test, list) else iq_vec_test
  iq_vec_train = torch.tensor(iq_vec_train) if isinstance(iq_vec_train, list) else iq_vec_train
  label_train = torch.tensor(label_train) if isinstance(label_train, list) else label_train
  label_test = torch.tensor(label_test) if isinstance(label_test, list) else label_test 

  if (plot_first_vec):
    fs, snr, duration, f0, f1, no_signal = labels_list[0].values()
    print(f"SNR: {snr}")
    print(f"No signal?: {no_signal}")
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    print(f"{iq_vec_test}")
    plt.style.use("dark_background")
    ax.plot(t, iq_tensors_list[0].real, label="I set", color="tab:blue")
    ax.plot(t, iq_tensors_list[0].imag, label="Q set", color="tab:orange")
    ax.set_title("I and Q", fontsize=4)
    ax.set_xlabel("time")
    ax.set_ylabel("IQ tensor data")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 50e-9)
    plt.show()

  return iq_vec_test, iq_vec_train, label_train, label_test

if __name__ == "__main__":
  plot_first_vec = True
  get_and_preprocess_dataset(1000, 0.2)
