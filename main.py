# boston.py
# Boston Area House Price regression
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11

import numpy as np
import torch as T

device = T.device('cpu')

# -----------------------------------------------------------

class BostonDataset(T.utils.data.Dataset):
  # features are in cols [0,12], median price in [13]

  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,14), dtype=np.float32)

    self.x_data = T.tensor(all_xy[:,0:12]).to(device) 
    self.y_data = \
      T.tensor(all_xy[:, [12,13]]).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx]
    price = self.y_data[idx] 
    return (preds, price)  # as a tuple

# -----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(12, 10)  # 12-(10-10)-2
    self.hid2 = T.nn.Linear(10, 10)
    self.hid3 = T.nn.Linear(10, 10)
    self.hid4 = T.nn.Linear(10, 10)
    self.oupt = T.nn.Linear(10, 2)
    self.bn = T.nn.BatchNorm1d(num_features=12, eps=0, affine=False, track_running_stats=False)

    T.nn.init.xavier_uniform_(self.hid1.weight) 
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.hid3.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.hid4.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    x = self.bn(x)
    z = T.tanh(self.hid1(x)) 
    z = T.tanh(self.hid2(z))
    z = T.tanh(self.hid3(z))
    z = T.tanh(self.hid4(z))
    z = self.oupt(z)  # no activation, aka Identity()
    return z

# -----------------------------------------------------------

def train(model, ds, bs, lr, me, le):
  # dataset, bat_size, lrn_rate, max_epochs, log interval
  train_ldr = T.utils.data.DataLoader(ds,
    batch_size=bs, shuffle=True)
  loss_func = T.nn.L1Loss()  # mean avg error
  optimizer = T.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(0, me):
    epoch_loss = 0.0  # for one full epoch
    for (b_idx, batch) in enumerate(train_ldr):
      X = batch[0]
      y = batch[1]
      optimizer.zero_grad()
      oupt = model(X)
      loss_val = loss_func(oupt, y)  # a tensor
      epoch_loss += loss_val.item()  # accumulate
      loss_val.backward()  # compute gradients
      optimizer.step()     # update weights

    if epoch % le == 0:
      print("epoch = %4d  |  loss = %0.4f" % \
        (epoch, epoch_loss)) 

# -----------------------------------------------------------

def accuracy(model, ds, pct_close):
  # one-by-one (good for analysis)
  model.eval()
  n_correct = 0; n_wrong = 0
  data_ldr =  T.utils.data.DataLoader(ds, batch_size=1,
    shuffle=False)
#   for (b_ix, batch) in enumerate(ds):
  for (b_ix, batch) in enumerate(data_ldr):
    X = batch[0]
    Y = batch[1]  # target poverty and price
    with T.no_grad():
      oupt = model(X)  # predicted price
    if T.abs(oupt[0] - Y[0]) < T.abs(pct_close * Y[0]) and \
      T.abs(oupt[1] - Y[1]) < T.abs(pct_close * Y[1]):
      n_correct += 1
    else:
      n_wrong += 1
  return (n_correct * 1.0) / (n_correct + n_wrong)

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBoston dual output regression using PyTorch ")
  np.random.seed(0) 
  T.manual_seed(0) 

  # 1. create Dataset DataLoader objects
  print("\nLoading Boston train and test Datasets ")
  train_file = "/home/dchangyu/multiple_regression/boston_train.txt"
  train_ds = BostonDataset(train_file)
  test_file = "/home/dchangyu/multiple_regression/boston_test.txt"
  test_ds = BostonDataset(test_file)

  # 2. create model
  print("\nCreating 12-(10-10)-2 regression network ")
  net = Net().to(device)
  net.train()

  # 3. train model
  print("\nbatch size = 10 ")
  print("loss = L1Loss() ")
  print("optimizer = Adam ")
  print("learn rate = 0.005 ")
  print("max epochs = 5000 ")

  print("\nStarting training ")
  train(net, train_ds, bs=10, lr=0.005, me=5000, le=1000)
  print("Done ")

  # 4. model accuracy
  net.eval()
  acc_train = accuracy(net, train_ds, 0.20)
  print("\nAccuracy on train (within 0.20) = %0.4f " % acc_train)
  acc_test = accuracy(net, test_ds, 0.20)
  print("Accuracy on test (within 0.20) = %0.4f " % acc_test)

  # 5. TODO: save model

  # 6. use model
  print("\nPredicting normalized (poverty, price) first train")
  print("Actual (poverty, price) = (8.77, 21.0) ")

#   x = np.array([0.000273, 0.000, 0.0707, -1, 0.469,
#     0.6421, 0.789, 0.049671, 0.02, 0.242, 0.178,
#     0.39690], dtype=np.float32)
  x = np.array([0.08014, 0.0, 5.96, 0.0, 0.499, 5.85, 41.5, 3.9342, 5.0, 279.0, 19.2, 396.9], dtype=np.float32)

  x = T.tensor(x, dtype=T.float32)

  with T.no_grad():
    oupt = net(x)
  print("Predicted poverty price = %0.4f %0.4f " % \
    (oupt[0], oupt[1]))

  print("\nEnd demo ")

if __name__=="__main__":
  main()