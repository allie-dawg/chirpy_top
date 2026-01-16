import torch
from ChirpClassifierModel import ChirpClassifierModel
import gentestdata

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  iq_vec_train, iq_vec_test, label_train, label_test = gentestdata.get_and_preprocess_dataset(data_set_sz=1000, train_size=0.2)
  print(type(iq_vec_test))
  model0 = ChirpClassifierModel(1600).to(device)
