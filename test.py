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
