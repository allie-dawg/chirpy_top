import torch
from ChirpClassifierModel import ChirpClassifierModel
import gentestdata

if __name__ == "__main__":
  num_samples = 1600
  device = "cuda" if torch.cuda.is_available() else "cpu"
  iq_vec_test, iq_vec_train, label_test, label_train = gentestdata.get_and_preprocess_dataset(batch_sz=1000, train_size=0.2)
  print(f"shape of iq_vec_train: {iq_vec_train.shape}")
  print(f"shape of iq_vec_test: {iq_vec_test.shape}")
  print(f"shape of label_train: {label_train.shape}")
  print(f"shape of label_test: {label_test.shape}")
  model0 = ChirpClassifierModel(1600).to(device)
  untrained_preds = model0(iq_vec_test.to(device))
  print(f"Length of untrained_preds: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
  print(f"Length of test lables: {len(label_test)}, Shape: {label_test.shape}")
  print(f"First 10 predictions: {untrained_preds[:10]}")
  print(f"First 10 labels: {label_test[:10]}") 

