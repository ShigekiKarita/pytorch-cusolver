# coding: utf-8
import torch
import torch_cusolver

A = torch.rand(2, 3, 3).cuda()
A = A.transpose(1, 2).matmul(A)
print(A)
w, V = torch_cusolver.cusolver_batch_eigh(A,
                                          False,
                                          True,
                                          1e-7,
                                          100,
                                          False)
print(w, V)
for i in range(A.shape[0]):
    print(i)
    print(A[i])
    print(V[i].t().matmul(w[i].diag()).matmul(V[i]))

