# coding: utf-8
import numpy
import torch
import torch_cusolver

A = torch.rand(2, 5, 5).cuda()
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


# A = torch.rand(3, 3).cuda()
# A = A.transpose(0, 1).matmul(A)
# B = torch.rand(3, 3).cuda()
# B = B.transpose(0, 1).matmul(B)

# example from https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
A = torch.cuda.FloatTensor(
    [[3.5, 0.5, 0.0],
     [0.5, 3.5, 0.0],
     [0.0, 0.0, 2.0]])
B = torch.cuda.FloatTensor(
    [[10, 2, 3],
     [2, 10, 5],
     [3, 5, 10]])
w_expect = torch.cuda.FloatTensor([0.158660256604, 0.370751508101882, 0.6])

for upper in [True, False]:
    for jacob in [True, False]:
        w, V, L = torch_cusolver.cusolver_generalized_eigh(A, False, B, False, upper, jacob, 1e-7, 100)
        torch.testing.assert_allclose(w, w_expect)
        torch.testing.assert_allclose(V.mm(B).mm(V.t()), torch.eye(A.shape[0], device=A.device))
        for i in range(3):
            torch.testing.assert_allclose(A.matmul(V[i]), B.matmul(V[i]) * w[i])


## example from https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
A = torch.cuda.FloatTensor(
    [[[ 1, -1],
      [-1,  2],
      [ 0,  0]],
     [[3, 4],
      [4, 7],
      [0, 0]]]) # .transpose(1, 2).contiguous()
s_expect = torch.cuda.FloatTensor(
    [[2.6180, 0.382],
     [9.4721, 0.5279]])
U, s, V = torch_cusolver.cusolver_batch_svd(A, False, 0.0, 100)

print(s_expect)
print(s)

# # s (2, 2) -> (2, 3)
for i in range(A.shape[0]):
    spad = torch.zeros(3, 2, device=A.device)
    spad.diagonal()[:2] = s[i]
    print(i)
    print(A[i])
    print(U[i].mm(spad).mm(V[i].t()))
