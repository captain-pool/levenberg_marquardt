import sparse_tensor
from absl.testing import absltest


class TestLevenbergMarquardt(absltest.TestCase):
    def setUp(self):
        self.jac1 = sparse_tensor.torch.rand(3, 4, 4)
        self.jac2 = sparse_tensor.torch.rand(3, 4, 4)

    def test_to_sparse_jacobian(self):
        sparse_jacobian = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        self.assertIsNotNone(sparse_jacobian)

    def test_add_sparse(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        sparse_jacobian2 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac2)
        add_check = self.jac1 + self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                add_check == (sparse_jacobian1 + sparse_jacobian2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_add_tensor(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        add_check = self.jac1 + self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                add_check == (sparse_jacobian1 + self.jac2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_sub_sparse(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        sparse_jacobian2 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac2)
        sub_check = self.jac1 - self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                sub_check == (sparse_jacobian1 - sparse_jacobian2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_sub_tensor(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        sub_check = self.jac1 - self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                sub_check == (sparse_jacobian1 - self.jac2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_mul_sparse(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        sparse_jacobian2 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac2)
        mul_check = self.jac1 * self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                mul_check == (sparse_jacobian1 * sparse_jacobian2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_mul_tensor(self):
        sparse_jacobian1 = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        mul_check = self.jac1 * self.jac2
        self.assertTrue(
            sparse_tensor.torch.all(
                mul_check == (sparse_jacobian1 * self.jac2).to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_transpose(self):
        sparse_jacobian = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        transpose_check = self.jac1.transpose(1, 2)
        self.assertTrue(
            sparse_tensor.torch.all(
                transpose_check == sparse_jacobian.transpose().to_dense()
            )
            .numpy()
            .tolist()
        )

    def test_T(self):
        sparse_jacobian = sparse_tensor.SparseJacobian.to_sparse_jacobian(self.jac1)
        transpose_check = self.jac1.transpose(1, 2)
        self.assertTrue(
            sparse_tensor.torch.all(transpose_check == sparse_jacobian.T.to_dense())
            .numpy()
            .tolist()
        )


if __name__ == "__main__":
    absltest.main()
