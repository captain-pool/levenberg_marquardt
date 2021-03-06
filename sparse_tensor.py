import torch_sparse
import torch


class SparseJacobian:
    def __init__(
        self, batch_size, shape, indices=None, values=None, sparse_jacobians=None
    ):
        self.batch_size = batch_size
        self.indices = indices or SparseJacobian.map_on_batch(
            lambda x: x.coalesce().indices(), sparse_jacobians, self.batch_size
        )
        self.values = values or SparseJacobian.map_on_batch(
            lambda x: x.coalesce().values(), sparse_jacobians, self.batch_size
        )
        self.shape = shape
        self.sparse_jacobians = sparse_jacobians or SparseJacobian.map_on_batch(
            lambda arg: SparseJacobian._get_sparse_typeclass(arg[1])(
                arg[0], arg[1], self.shape
            ),
            list(zip(self.indices, self.values)),
            self.batch_size,
        )
        self._transpose = None

    @classmethod
    def to_sparse_jacobian(cls, batched_jacobian):
        batch_size = len(batched_jacobian)
        sparse_jacobians = SparseJacobian.map_on_batch(
            SparseJacobian._to_sparse, batched_jacobian
        )
        indices = SparseJacobian.map_on_batch(
            lambda x: x.coalesce().indices(), sparse_jacobians
        )
        values = SparseJacobian.map_on_batch(
            lambda x: x.coalesce().values(), sparse_jacobians
        )
        shape = sparse_jacobians[0].shape
        return cls(batch_size, shape, indices, values, sparse_jacobians)

    @property
    def T(self):
        if not self._transpose:
            self._transpose = self.transpose()
        return self._transpose

    @staticmethod
    def map_on_batch(function, batched_items, batch_size=None):
        # Find a parallelized alternative
        batch_size = batch_size or len(batched_items)
        return [function(batched_items[index]) for index in range(batch_size)]

    @staticmethod
    def _get_sparse_typeclass(x):
        x_typename = torch.typename(x).split(".")[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)
        return sparse_tensortype

    @staticmethod
    def _to_sparse(x):
        """ converts dense tensor x to sparse format """
        sparse_tensortype = SparseJacobian._get_sparse_typeclass(x)
        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())

    def matmul(self, other, keep_dense=False):
        def sparse_sparse_matmul_fn(args):
            return torch_sparse.spspmm(
                *args, m=self.shape[0], k=self.shape[1], n=other.shape[1]
            )

        def sparse_dense_matmul_fn(args):
            return torch.sparse.mm(*args)

        if isinstance(other, SparseJacobian):
            interim_items = SparseJacobian.map_on_batch(
                sparse_sparse_matmul_fn,
                list(zip(self.indices, self.values, other.indices, other.values)),
                self.batch_size,
            )
            interim_indices, interim_values = list(zip(*interim_items))
            return SparseJacobian(
                self.batch_size,
                torch.Size([self.shape[0], other.shape[1]]),
                interim_indices,
                interim_values,
            )
        dense_products = SparseJacobian.map_on_batch(
            sparse_dense_matmul_fn,
            list(zip(self.sparse_jacobians, other)),
            self.batch_size,
        )
        if keep_dense:
            return torch.stack(dense_products)
        return SparseJacobian.to_sparse_jacobian(dense_products)

    def transpose(self):
        def transpose_fn(sparse_jacobian):
            return sparse_jacobian.transpose(0, 1)

        sparse_jacobians = SparseJacobian.map_on_batch(
            transpose_fn, self.sparse_jacobians, self.batch_size
        )
        return SparseJacobian(
            self.batch_size,
            torch.Size([self.shape[1], self.shape[0]]),
            sparse_jacobians=sparse_jacobians,
        )

    def __add__(self, other):
        def add_fn(sparses):
            return sparses[0] + sparses[1]

        def add_sparse_dense(params):
            dummy_sparse = SparseJacobian._to_sparse(params[1]).coalesce()
            values = dummy_sparse.values()
            index0, index1 = dummy_sparse.indices()
            indices1 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            index0, index1 = params[0].coalesce().indices()
            indices2 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            final_indices = list(indices1 | indices2)
            final_values = []
            for index0, index1 in final_indices:
                final_values.append(
                    dummy_sparse[index0][index1] + params[0][index0][index1]
                )
            return SparseJacobian._get_sparse_typeclass(params[1][0])(
                torch.tensor(list(zip(*final_indices))),
                torch.tensor(final_values),
                params[1].shape,
            )

        if isinstance(other, SparseJacobian):
            added_jacobians = SparseJacobian.map_on_batch(
                add_fn,
                list(zip(self.sparse_jacobians, other.sparse_jacobians)),
                self.batch_size,
            )
        else:
            added_jacobians = SparseJacobian.map_on_batch(
                add_sparse_dense,
                list(zip(self.sparse_jacobians, other)),
                self.batch_size,
            )

        return SparseJacobian(
            self.batch_size, self.shape, sparse_jacobians=added_jacobians
        )

    def __sub__(self, other):
        def sub_fn(sparses):
            return sparses[0] - sparses[1]

        def sub_sparse_dense(params):
            dummy_sparse = SparseJacobian._to_sparse(params[1]).coalesce()
            values = dummy_sparse.values()
            index0, index1 = dummy_sparse.indices()
            indices1 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            index0, index1 = params[0].coalesce().indices()
            indices2 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            final_indices = list(indices1 | indices2)
            final_values = []
            for index0, index1 in final_indices:
                final_values.append(
                    params[0][index0][index1] - dummy_sparse[index0][index1]
                )
            return SparseJacobian._get_sparse_typeclass(params[1][0])(
                torch.tensor(list(zip(*final_indices))),
                torch.tensor(final_values),
                params[1].shape,
            )

        if isinstance(other, SparseJacobian):
            subbed_jacobians = SparseJacobian.map_on_batch(
                sub_fn,
                list(zip(self.sparse_jacobians, other.sparse_jacobians)),
                self.batch_size,
            )
        else:
            subbed_jacobians = SparseJacobian.map_on_batch(
                sub_sparse_dense,
                list(zip(self.sparse_jacobians, other)),
                self.batch_size,
            )
        return SparseJacobian(
            self.batch_size, self.shape, sparse_jacobians=subbed_jacobians
        )

    def __mul__(self, other):
        def mul_fn(sparses):
            print(sparses[0].is_sparse, sparses[1].is_sparse)
            return sparses[0] * sparses[1]

        def mul_sparse_dense(params):
            dummy_sparse = SparseJacobian._to_sparse(params[1]).coalesce()
            values = dummy_sparse.values()
            index0, index1 = dummy_sparse.indices()
            indices1 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            index0, index1 = params[0].coalesce().indices()
            indices2 = set(list(zip(index0.numpy().tolist(), index1.numpy().tolist())))
            final_indices = list(indices1 | indices2)
            final_values = []
            for index0, index1 in final_indices:
                final_values.append(
                    params[0][index0][index1] * dummy_sparse[index0][index1]
                )
            return SparseJacobian._get_sparse_typeclass(params[1][0])(
                torch.tensor(list(zip(*final_indices))),
                torch.tensor(final_values),
                params[1].shape,
            )

        if isinstance(other, SparseJacobian):
            multiplied_jacobians = SparseJacobian.map_on_batch(
                mul_fn,
                list(zip(self.sparse_jacobians, other.sparse_jacobians)),
                self.batch_size,
            )
        else:
            if len(getattr(other, "shape", [])) == 3:
                multiplied_jacobians = SparseJacobian.map_on_batch(
                    mul_sparse_dense,
                    list(zip(self.sparse_jacobians, other)),
                    self.batch_size,
                )
            else:
                multiplied_jacobians = SparseJacobian.map_on_batch(
                    lambda x: x * other, self.sparse_jacobians, self.batch_size
                )
        return SparseJacobian(
            self.batch_size, self.shape, sparse_jacobians=multiplied_jacobians
        )

    def inverse(self):
        def inverse_fn(sparse):
            # This is a really naive approach. Look for a better alternative.
            # This is the only solution as of now. Plus there's another reason
            # Inverse of a sparse matrix might not be a sparse matrix
            return torch.inverse(sparse.to_dense())

        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"Cannot invert matrix of shape: [{self.shape[0]}, {self.shape[1]}]. Need a square matrix"
            )
        denses = SparseJacobian.map_on_batch(
            inverse_fn, self.sparse_jacobians, self.batch_size
        )
        return SparseJacobian.to_sparse_jacobian(denses)

    def to_dense(self):
        def to_dense_fn(sparse):
            return sparse.to_dense()

        denses = SparseJacobian.map_on_batch(
            to_dense_fn, self.sparse_jacobians, self.batch_size
        )
        return torch.stack(denses)
