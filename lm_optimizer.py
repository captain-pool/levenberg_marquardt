import torch
import sparse_tensor as st


class LMOptimizer(object):
    def __init__(self, batch_size, parameters, damping=1):
        self._batch_size = batch_size
        self._parameters = parameters
        stacked_params = torch.stack(self._parameters)
        self._param_shape = stacked_params.shape[::-1]
        self._flattened_parameters = torch.flatten(stacked_params)
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
        batched_jac = []
        flat_y = y.reshape(-1)
        x = self._parameters
        grad_y = torch.zeros_like(flat_y)
        index0 = torch.tensor(list(range(self._param_shape[0])))
        for i in range(len(flat_y)):
            grad_y[i] = 1.0
            (grad_x,) = torch.autograd.grad(
                flat_y, x[i % self._param_shape[1]], grad_y, retain_graph=True, create_graph=create_graph
            )
            index1 = torch.tensor([i % self._param_shape[1]] * self._param_shape[0])
            indices = torch.stack([index0, index1])
            grad_jacobian_entry = torch.sparse.FloatTensor(indices, grad_x, self._param_shape)
            jac.append(torch.cat(list(grad_jacobian_entry)))
            if not (i + 1) % y.shape[1]:
              batched_jac.append(torch.stack(jac))
              jac = []
            grad_y[i] = 0.0
        return batched_jac

    def _pseudo_inverse(self, jacobians):
        jacobian = st.SparseJacobian(len(jacobians), jacobians[0].shape, sparse_jacobians=jacobians)
        jacobian_abs = jacobian.T.matmul(jacobian)
        marquardt = torch.eye(*jacobian_abs.shape)
        return (jacobian_abs + marquardt * self._damping).inverse().matmul(jacobian.T)

    def step(self, output_tensor):
        jacobians = self._get_jacobian(output_tensor)
        ps_inv = self._pseudo_inverse(jacobians)
        batched_psj_vp = torch.squeeze(ps_inv.matmul(
            torch.unsqueeze(output_tensor, -1), keep_dense=True))
        self._flattened_parameters.data -= batched_psj_vp.mean(0)
