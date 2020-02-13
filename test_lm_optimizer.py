import torch
import lm_optimizer

def F(X, theta):
  return X @ (theta**2)

def residual(theta, Y, X):
  return Y - F(X, theta) # X.shape = [3, 1], Y.shape = [3, 1]

X = torch.rand(1, 10).normal_()
theta = torch.rand(10, 5).normal_().requires_grad_(True)
Y = torch.rand(1, 5).normal_()

o = residual(theta, Y, X)
opt = lm_optimizer.LMOptimizer(o.shape[0], theta, damping=0.13)

while True:
  o = residual(theta, Y, X)
  if o.mean().data < 0.0001:
    break
  print(o.mean().data.numpy())
  opt.zero_grad()
  opt.step(o)
