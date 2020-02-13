import torch
import sparse_tensor as st



class LMOptimizer(object):
  def __init__(self, batch_size, parameters, damping=1):
    self._batch_size = batch_size
    self._parameters = parameters
    self._flattened_parameters = self._flatten(self._parameters)
    self._damping = damping
  
  def _numel(self, tensor):
    return abs(tensor.numel() // self._batch_size)

  def _flatten(self, tensor):
    numel = self._numel(tensor)
    return tensor.reshape([self._batch_size, numel])

  def zero_grad(self):
    for param in self._flattened_parameters:
      if param.grad:
        param.grad.detach_()
        param.grad.zero_()
    for param in self._parameters:
      if param.grad:
        param.grad.detach_()
        param.grad.zero_()

  def _get_jacobian(self, y, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    x = self._parameters
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
      grad_y[i] = 1.
      grad_x,  = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
      jac.append(grad_x.reshape(x.shape))
      grad_y[i] = 0.
    return torch.stack(jac).reshape([-1, self._numel(y), self._numel(x)])

  def _pseudo_inverse(self, jacobian):
    jacobian = st.SparseJacobian.to_sparse_jacobian(jacobian)
    jacobian_abs = jacobian.T.matmul(jacobian)

    marquardt = torch.eye(*jacobian_abs.shape).repeat(jacobian_abs.batch_size, 1)
    if len(marquardt.shape) != 3:
      marquardt = marquardt.unsqueeze(0)
    marquardt = st.SparseJacobian.to_sparse_jacobian(marquardt)
    return (jacobian_abs + marquardt * self._damping).inverse().matmul(jacobian.T)

  def step(self, output_tensor):
    jacobian = self._get_jacobian(output_tensor)
    ps_inv = self._pseudo_inverse(jacobian)
    self._flattened_parameters.data -= torch.squeeze(ps_inv.matmul(
                                                                  torch.unsqueeze(
                                                                    output_tensor, -1), keep_dense=True), -1)
