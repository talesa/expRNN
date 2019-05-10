import torch
import torch.nn as nn

from exp_numpy import expm, expm_frechet
from initialization import henaff_init


class Orthogonal(nn.Module):
    """
    Implements a non-square linear with orthogonal colums
    """
    def __init__(self, input_size, output_size):
        super(Orthogonal, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.orthogonal_kernel = torch.empty(self.max_size, self.max_size, requires_grad=True)
        self.skew_initializer = henaff_init

        self.log_orthogonal_kernel.data = \
            torch.as_tensor(self.skew_initializer(self.max_size),
                            dtype=self.log_orthogonal_kernel.dtype,
                            device=self.log_orthogonal_kernel.device)
        self.orthogonal_kernel.data = self._B

    @property
    def _A(self):
        A = self.log_orthogonal_kernel.data
        # print(f"A q1: {A}")
        A = A.triu(diagonal=1)
        return A - A.t()

    @property
    def _B(self):
        return expm(self._A)

    def orthogonal_step(self, optim):
        A = self._A
        B = self.orthogonal_kernel.data
        G = self.orthogonal_kernel.grad.data
        BtG = B.t().mm(G)
        grad = 0.5*(BtG - BtG.t())
        frechet_deriv = B.mm(expm_frechet(-A, grad))

        self.log_orthogonal_kernel.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)

        print(f'orthogonal_step() self.log_orthogonal_kernel.data before optim.step()\n{self.log_orthogonal_kernel.data}')

        optim.step()

        print(f'orthogonal_step() self.log_orthogonal_kernel.data after optim.step()\n{self.log_orthogonal_kernel.data}')

        # B = self._B
        # print(f'B {B}')
        self.orthogonal_kernel.data = self._B
        self.orthogonal_kernel.grad.data.zero_()
        # print(f'orthogonal_step() orthogonal_kernel.data\n{self.orthogonal_kernel.data}')
        # print(f'orthogonal_step() _B\n{self._B}')

    def forward(self, input):
        # print(f"self._B q1 {self.orthogonal_kernel.data}")
        # assert ((self.orthogonal_kernel.data == self._B).all())
        return input.matmul(self.orthogonal_kernel[:self.input_size, :self.output_size])


class Orthogonal2(nn.Module):
    """
    Implements a non-square linear with orthogonal colums
    """
    def __init__(self, input_size, output_size, lr_factor=0.1):
        super(Orthogonal2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.log_orthogonal_kernel.register_hook(lambda: print("This should not be executed"))
        self.register_buffer('orthogonal_kernel', torch.empty(self.max_size, self.max_size, requires_grad=True))
        # self.orthogonal_kernel = torch.empty(self.max_size, self.max_size, requires_grad=True)
        self.orthogonal_kernel.register_hook(self.orthogonal_kernel_grad_hook)
        self.skew_initializer = henaff_init

        self.log_orthogonal_kernel.data = \
            torch.as_tensor(self.skew_initializer(self.max_size),
                            dtype=self.log_orthogonal_kernel.dtype,
                            device=self.log_orthogonal_kernel.device)
        self.orthogonal_kernel.data = self._B

        self.lr_factor = lr_factor

    @property
    def _A(self):
        A = self.log_orthogonal_kernel.data
        # print(f"A q2: {A}")
        A = A.triu(diagonal=1)
        return A - A.t()

    @property
    def _B(self):
        return expm(self._A)

    def orthogonal_kernel_grad_hook(self, orthogonal_kernel_grad):
        A = self._A
        B = self.orthogonal_kernel.data
        G = orthogonal_kernel_grad # self.orthogonal_kernel.grad.data
        BtG = B.t().mm(G)
        grad = 0.5*(BtG - BtG.t())
        frechet_deriv = B.mm(expm_frechet(-A, grad))

        self.log_orthogonal_kernel.grad = self.lr_factor * (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)

        return None

    def forward(self, input):
        self.orthogonal_kernel.data = self._B
        if self.orthogonal_kernel.grad is not None:
            self.orthogonal_kernel.grad.data.zero_()
        # print(f"self._B q2 {self.orthogonal_kernel.data}")
        # assert ((self.orthogonal_kernel.data == self._B).all())
        return input.matmul(self.orthogonal_kernel[:self.input_size, :self.output_size])