import torch
import torch.nn as nn
import numpy as np

from orthogonal_layer import Orthogonal, Orthogonal2

d = 3
q1 = Orthogonal(d, d)
q2 = Orthogonal2(d, d)

q2.log_orthogonal_kernel.data = q1.log_orthogonal_kernel.data
q2.orthogonal_kernel.data = q1.orthogonal_kernel.data

lr = 1e-3

o1 = torch.optim.Adam(q1.parameters(), lr=lr)
o2 = torch.optim.Adam(q2.parameters(), lr=lr)

x = torch.Tensor([[0., 0.5, 0.5]])
y = torch.Tensor([[0.5, 0.5, 0.]])

print("START")

for i in range(100):
    print(f"STEP {i}")

    print("BEFORE forward()")
    print()
    print('q1.log_orthogonal_kernel.data')
    print(q1.log_orthogonal_kernel.data)
    print('q2.log_orthogonal_kernel.data')
    print(q2.log_orthogonal_kernel.data)
    print()
    print('q1.orthogonal_kernel.data')
    print(q1.orthogonal_kernel.data)
    print('q2.orthogonal_kernel.data')
    print(q2.orthogonal_kernel.data)
    print()
    print('q1._B')
    print(q1._B)
    print('q2._B')
    print(q2._B)

    # q1.orthogonal_kernel.data = q1._B

    l1 = (y - q1(x)).norm()
    l2 = (y - q2(x)).norm()
    # assert (q1.orthogonal_kernel.data.eq(q2.orthogonal_kernel.data).all())

    # print(q1.orthogonal_kernel)
    # print(q2.orthogonal_kernel)

    # print(q1.log_orthogonal_kernel.data)
    # print(q2.log_orthogonal_kernel.data)

    # print()

    print()
    print("AFTER forward()")
    print()
    print('q1.log_orthogonal_kernel.data')
    print(q1.log_orthogonal_kernel.data)
    print('q2.log_orthogonal_kernel.data')
    print(q2.log_orthogonal_kernel.data)
    print()
    print('q1.orthogonal_kernel.data')
    print(q1.orthogonal_kernel.data)
    print('q2.orthogonal_kernel.data')
    print(q2.orthogonal_kernel.data)
    print()
    print('q1._B')
    print(q1._B)
    print('q2._B')
    print(q2._B)

    assert((q2._B == q2.orthogonal_kernel.data).all())
    assert((q1._B == q1.orthogonal_kernel.data).all())

    assert ((q1.orthogonal_kernel.data == q2.orthogonal_kernel.data).all())

    o1.zero_grad()
    l1.backward()
    q1.orthogonal_step(o1)
    print(l1)

    o2.zero_grad()
    l2.backward()
    o2.step()
    print(l2)

    assert(q1.log_orthogonal_kernel.data.eq(q2.log_orthogonal_kernel.data).all())

    # assert(q1.orthogonal_kernel.data.eq(q2.orthogonal_kernel.data).all())

print("DONE")

print(q1._B)
print(q2._B)