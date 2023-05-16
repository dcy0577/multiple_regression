# boston.py
# Boston Area House Price regression
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6

import numpy as np
import torch as T
from sklearn import preprocessing
import torch.multiprocessing as mp


device = T.device('cuda')

# -----------------------------------------------------------

class BostonDataset(T.utils.data.Dataset):
  # features are in cols [0,11], median price in [12,13]

  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,14), dtype=np.float32)

    self.x_data = T.tensor(all_xy[:,0:12]).to(device) 
    self.y_data = T.tensor(all_xy[:, [12,13]]).to(device)

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
    self.hid1 = T.nn.Linear(12, 512)  # 12-(10-10)-2
    self.hid2 = T.nn.Linear(512, 512)
    self.hid3 = T.nn.Linear(512, 512)
    self.hid4 = T.nn.Linear(512, 512)
    self.hid5 = T.nn.Linear(512, 512)
    self.oupt = T.nn.Linear(512, 2)
    self.drop_out = T.nn.Dropout(p=0.2)


    T.nn.init.kaiming_normal_(self.hid1.weight) 
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.kaiming_normal_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.kaiming_normal_(self.hid3.weight)
    T.nn.init.zeros_(self.hid3.bias)
    T.nn.init.kaiming_normal_(self.hid4.weight)
    T.nn.init.zeros_(self.hid4.bias)
    T.nn.init.kaiming_normal_(self.hid5.weight)
    T.nn.init.zeros_(self.hid5.bias)
    T.nn.init.kaiming_normal_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = self.drop_out(z) 
    z = T.relu(self.hid2(z))
    z = self.drop_out(z) 
    z = T.relu(self.hid3(z))
    z = self.drop_out(z) 
    z = T.relu(self.hid4(z))
    z = self.drop_out(z) 
    z = T.relu(self.hid5(z))
    z = self.drop_out(z) 
    z = self.oupt(z)  # no activation, aka Identity()
    return z

# -----------------------------------------------------------

def validate(val_loss, model, val_ldr, loss_func):
  model.eval()
  with T.no_grad():
    for (b_ix, batch) in enumerate(val_ldr):
      X = batch[0]
      Y = batch[1]  # target poverty and price
      with T.no_grad():
        oupt = model(X)  # predicted price
        loss = loss_func(oupt, Y)
        val_loss += loss.item()
    return val_loss


def train(model, train_ds, val_ds, bs, lr, me, le):
  # dataset, bat_size, lrn_rate, max_epochs, log interval
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bs, shuffle=True)
  val_ldr = T.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=True)

  # loss_func = T.nn.L1Loss()  # mean avg error
  loss_func = T.nn.MSELoss()  # mse
  # loss_func = T.nn.SmoothL1Loss()  # mae mse
  optimizer = T.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

  for epoch in range(0, me):
    epoch_train_loss = 0.0  # for one full epoch
    model.train()
    for (b_idx, batch) in enumerate(train_ldr):
      X = batch[0]
      y = batch[1]
      optimizer.zero_grad()
      oupt = model(X)
      loss_val = loss_func(oupt, y)  # a tensor
      epoch_train_loss += loss_val.item()  # accumulate
      loss_val.backward()  # compute gradients
      optimizer.step()     # update weights

    if epoch % le == 0:
      # validation
      epoch_valid_loss = 0.0
      epoch_valid_loss = validate(epoch_valid_loss, model, val_ldr, loss_func)
      print("epoch = %4d  |  train_loss = %0.4f  |  val_loss = %0.4f" % \
        (epoch, epoch_train_loss, epoch_valid_loss)) 

# -----------------------------------------------------------

def accuracy(model, ds, pct_close):
  # one-by-one (good for analysis)
  model.eval()
  n_correct = 0; n_wrong = 0
  data_ldr =  T.utils.data.DataLoader(ds, batch_size=1,
    shuffle=False)
#   for (b_ix, batch) in enumerate(ds):
  for (b_ix, batch) in enumerate(ds):
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
  print("\nCreating 12-(100-100-100-100)-2 regression network ")
  net = Net().to(device)
  net.train()

  # 3. train model
  print("\nbatch size = 20 ")
  print("loss = L1Loss() ")
  print("optimizer = Adam ")
  print("learn rate = 0.005 ")
  print("max epochs = 5000 ")

  print("\nStarting training ")
  train(net, train_ds, test_ds, bs=30, lr=0.005, me=3000, le=300)
  print("Done ")

  # 4. model accuracy
  net.eval()
  acc_train = accuracy(net, train_ds, 0.05)
  print("\nAccuracy on train (within 0.20) = %0.4f " % acc_train)
  acc_test = accuracy(net, test_ds, 0.05)
  print("Accuracy on test (within 0.20) = %0.4f " % acc_test)

  # 5. save model
  T.save(net.state_dict(), "trained_weights.pth")

  print("\nEnd demo ")

if __name__=="__main__":
  # mp.set_start_method(method='forkserver', force=True)
  main()