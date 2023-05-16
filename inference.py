from main import Net
import torch as T
import numpy as np
import joblib

def inference():

  print("\nPredicting normalized (poverty, price) first train")
  print("Actual (poverty, price) = ( 5.64,  23.90) ")
  net = Net().to("cuda")
  net.load_state_dict(T.load("trained_weights.pth"))
  net.eval()
  x = np.array([0.06076,   0.00,  11.930,  0,  0.5730,  6.9760,  91.00,  2.1675,   1,  273.0,  21.00, 396.90  ], dtype=np.float32)
  # new data normalization use presaved scaler
  scaler = joblib.load("scaler.pkl")
  x = scaler.transform(x.reshape(1,-1))

  x = T.tensor(np.squeeze(x), dtype=T.float32).to("cuda")

  with T.no_grad():
    oupt = net(x)
  print("Predicted poverty price = %0.4f %0.4f " % \
    (oupt[0], oupt[1]))

if __name__=="__main__":
  inference()