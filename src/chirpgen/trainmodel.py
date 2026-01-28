import torch
from torch.nn import BCEWithLogitsLoss
from ChirpClassifierModel import ChirpClassifierModel
#from chirpgen import preprocessdataset
import preprocessdataset

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc =  (correct / len(y_pred)) * 100
  return acc

if __name__ == "__main__":
  batch_sz = 1000
  num_samples_in_single_vec = 16384
  device = "cuda" if torch.cuda.is_available() else "cpu"
  iq_vec_test, iq_vec_train, label_test, label_train = preprocessdataset.get_and_preprocess_dataset(batch_sz=batch_sz, low_snr=1, high_snr=1, train_size=0.2)
  model0 = ChirpClassifierModel(num_samples_in_single_vec).to(device)
  untrained_preds = model0(iq_vec_test.to(device))

  loss_fn = BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.1)
  torch.manual_seed(42) 
  epochs = 100

  iq_vec_train, label_train = iq_vec_train.to(device), label_train.to(device)
  iq_vec_test, label_test = iq_vec_test.to(device), label_test.to(device)

  for epoch in range(epochs):

    model0.train()

    # 1. Forward pass model outputs/returns raw logits
    label_logits = model0(iq_vec_train).squeeze()
    label_pred = torch.round(torch.sigmoid(label_logits)) # Turn raw outputs -> prediction probs -> Booleans

    # 2. Calculate loss/accuracy 
    loss = loss_fn(label_logits, label_train)
    acc = accuracy_fn(y_true=label_train, y_pred=label_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer Step
    optimizer.step()

    ### Testing
    model0.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model0(iq_vec_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits))
      # 2. Calculate loss/accuracy
      test_loss = loss_fn(test_logits, label_test)
      test_acc = accuracy_fn(y_true=label_test, y_pred=test_pred)

    # Print out what is happening every 10 epocs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f} | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")

