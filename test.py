# coding: utf-8
import numpy
import torch
import torch_cusolver

A = torch.rand(2, 3, 3).cuda()
A = A.transpose(1, 2).matmul(A)
w, V = torch_cusolver.cusolver_batch_eigh(A,
                                          False,
                                          True,
                                          1e-7,
                                          100,
                                          False)

for i in range(A.shape[0]):
    a = A[i]
    e = V[i].t().matmul(w[i].diag()).matmul(V[i])
    torch.testing.assert_allclose(a, e)


A = torch.rand(3, 3).cuda()
A = A.transpose(0, 1).matmul(A)
B = torch.rand(3, 3).cuda()
B = B.transpose(0, 1).matmul(B)
A = torch.cuda.FloatTensor(
    [[3.5, 0.5, 0.0],
     [0.5, 3.5, 0.0],
     [0.0, 0.0, 2.0]])
B = torch.cuda.FloatTensor(
    [[10, 2, 3],
     [2, 10, 5],
     [3, 5, 10]])

# A = torch.eye(2).cuda()
# B = torch.eye(2).cuda()

w, V, L = torch_cusolver.cusolver_generalized_eigh(A, True, B, True, False, 1e-7, 100)
# print(w)
# print(V)
# type 1 only
# print(A.mm(V))
# print(B.mm(V).mm(w.diag()))

# type 1 or 2
print(V.t().mm(B).mm(V))


# for i in range(A.shape[0]):
#     a = A[i]
#     e = V[i].t().matmul(w[i].diag()).matmul(V[i])
#     torch.testing.assert_allclose(a, e)
