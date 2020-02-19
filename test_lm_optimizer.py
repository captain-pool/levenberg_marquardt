import torch
import lm_optimizer


def F(X, theta):
    return torch.stack([X @ (theta_i ** 2) for theta_i in theta]).T

def residual(theta, Y, X):
    result = F(X, theta)
    return (Y - result)


X = torch.rand(2, 3).normal_()
theta = [torch.rand(3).normal_().requires_grad_(True) for _ in range(2)]
Y = torch.rand(2, 2).normal_()
o = residual(theta, Y, X)
opt = lm_optimizer.LMOptimizer(o.shape[0], theta, damping=0.13)
x = 40
while x:
    o = residual(theta, Y, X)
    # while o.mean().numpy() < 0.0001:
    #  break
    print(o.mean().data.numpy())
    opt.zero_grad()
    opt.step(o)
    x -= 1
