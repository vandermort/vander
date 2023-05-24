import torch
import torch.nn.functional as F
import numpy as np

from vndr.modules import torch_interleave_columns, gale
from scipy.special import logit


class SigmoidBottleneckLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, feature_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature_dim = feature_dim

        self.mlp = torch.nn.Sequential(
            # Project from input dim to label dim
            torch.nn.Linear(self.in_dim, self.feature_dim, bias=True),
            torch.nn.Linear(self.feature_dim, self.out_dim, bias=False)
        )

    def forward(self, src):
        logits = self.mlp(src)
        return logits

    def encode(self, labels):
        return labels

    def decode(self, labels):
        return labels

    def decode_logits(self, logits):
        return logits


class KSparseClassifier(torch.nn.Module):
    """
    A Linear layer that guarantees that k-sparse labels are argmaxable.
    For multi-label classification this means that outputs with k labels "on" are argmaxable.
    """
    def __init__(self, in_dim, out_dim, k, slack_dims=0, param='gale', freeze=False, use_init=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.slack_dims = slack_dims
        assert param in ['vander', 'gale']
        self.param = param
        self.freeze = freeze
        self.use_init = use_init

        self.D = 2 * self.k + 1
        self.N = out_dim

        self.total_D = self.D + self.slack_dims

        assert self.N <= 10000
        # Keep 5% of the interval for "padding" to enforce
        # distinct values
        self.EPSILON = (1. / self.N) * 5e-2
        # Need to project from in_dim to cardinality dim
        self.proj = torch.nn.Linear(self.in_dim, self.total_D, bias=True)

        if self.use_init:
            # The first column of W is the all 1/np.sqrt(N) vector
            # let's initialise the bias of the linear projection
            # such that the output is the MLE.
            # This makes sense since we know that our labels are sparse
            mle = self.k / self.N
            self.proj.bias.data[0] = logit(mle) * np.sqrt(self.N)

        wt = torch.empty(self.N, dtype=torch.float32)
        self.wt = torch.nn.parameter.Parameter(wt, requires_grad=not self.freeze)
        torch.nn.init.uniform_(self.wt, -1, 1)

        # Slack dimensions
        if self.slack_dims:
            self.Ws = torch.nn.Linear(self.slack_dims, self.N, bias=False)

    def compute_t(self):
        # ENFORCES:
        # a) t is increasing
        # b) t is distinct
        wt = torch.softmax(self.wt, dim=0)
        # We keep N * EPSILON units of length so that we can ensure that
        # the entries of wt are always offset by at least EPSILON from each other
        # this should ensure that $t_i$ are always distinct.
        # Rescales nums to sum to 1 - N * EPSILON
        wt = wt * (1 - self.N * self.EPSILON)
        t = torch.cumsum(wt + self.EPSILON, dim=0)
        return t

    def compute_W(self):
        t = self.compute_t()

        if self.param == 'vander':
            # Vander parametrisation
            # MAX_OUT = 1e1
            # # What number when raised to the power D gives us MAX_OUT?
            # t_max = np.power(MAX_OUT, 1./self.D)

            # Stretch out t to range we are comfortable with
            # t = (2 * t - 1.) * t_max
            t = (2 * t - 1.)
            Wc = torch.linalg.vander(t, N=self.D)
            Wc = Wc / torch.linalg.norm(Wc, dim=0, keepdim=True)

        elif self.param == 'gale':
            # Gale parametrisation - only guarantees correct results
            # when self.D is odd
            t = t * 2 * torch.pi
            d = torch.arange(1, (self.D//2) + 1, dtype=torch.float32, device=self.wt.device)
            d = torch.repeat_interleave(d, 2)

            # Prepare to outer product
            t = t.view(-1, 1)
            d = d.view(1, -1)

            # Outer product
            W = t @ d
            # W = torch.hstack([torch.cos(W[:, ::2]), torch.sin(W[:, 1::2])])
            # Construct Gale Parametrisation
            W = torch_interleave_columns(torch.cos(W[:, ::2]), torch.sin(W[:, 1::2]))

            # Affinisation (add column of ones to first position)
            Wc = F.pad(W, (1, 0), 'constant', 1.)

            # Normalize
            Wc[:, 0] = Wc[:, 0] * np.sqrt(1/self.N)
            Wc[:, 1:] = Wc[:, 1:] * np.sqrt(2/self.N)
            # print(Wc.T @ Wc)
            # _, s, _ = np.linalg.svd(Wc.detach().cpu().numpy())
            # print(s)
            # print(np.linalg.cond(Wc.detach().cpu().numpy()))

        if self.slack_dims:
            Ws = torch.hstack([Wc, self.Ws.weight])
        else:
            Ws = Wc
        return Ws

    def forward(self, xx):

        xx = self.proj(xx)

        W = self.compute_W()
        yy = F.linear(xx, W)

        return yy

    def encode(self, labels):
        return labels

    def decode(self, labels):
        return labels

    def decode_logits(self, logits):
        return logits


class KSparseFFTClassifier(torch.nn.Module):
    """
    A Linear layer that guarantees that k-sparse labels are argmaxable.
    For multi-label classification this means that outputs with k labels "on" are argmaxable.
    """
    def __init__(self, in_dim, out_dim, k, slack_dims=0, use_init=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.slack_dims = slack_dims
        self.use_init = use_init

        self.D = 2 * self.k + 1
        self.N = out_dim

        self.total_D = self.D + self.slack_dims

        self.proj = torch.nn.Linear(self.in_dim, self.total_D, bias=True)

        if self.use_init:
            # The first column of W is the all 1/np.sqrt(N) vector
            # let's initialise the bias of the linear projection
            # such that the output is the MLE.
            # This makes sense since we know that our labels are sparse
            mle = self.k / self.N
            self.proj.bias.data[0] = logit(mle) * np.sqrt(self.N)

        # Slack dimensions
        if self.slack_dims:
            self.Ws = torch.nn.Linear(self.slack_dims, self.N, bias=False)

    def compute_W(self):
        Wc = torch.tensor(gale(self.N, self.D),
                          dtype=torch.float32,
                          device=self.proj.weight.device)
        # Wc[:, 1:] = Wc[:, 1:] / np.sqrt(2) 
        # print(Wc.T @ Wc)
        # _, s, _ = np.linalg.svd(Wc.detach().cpu().numpy())
        # print(s)
        # print(np.linalg.cond(Wc.detach().cpu().numpy()))

        if self.slack_dims:
            Ws = torch.hstack([Wc, self.Ws.weight])
        else:
            Ws = Wc
        return Ws

    def forward(self, xx):

        bs, dim = xx.shape

        xx = self.proj(xx)
        # FFT does not make cols orthonormal - multiply here to make
        # forward and mat mul with computed W equivalent
        xx[:, 1:self.D] = xx[:, 1:self.D] * np.sqrt(2)

        # Interpret activation as complex number for dft
        dc = torch.complex(xx[:, 0], torch.zeros(bs, device=xx.device))
        # Use conjugate as cyclic polytope is framed with + sign for sin
        cx = torch.view_as_complex(xx[:, 1:self.D].view(bs, -1, 2).contiguous()).conj()

        # Concatenate dc term and cos, sin terms
        cx = torch.cat([dc.view(-1, 1), cx], axis=1)

        yy_dft = torch.fft.ifft(cx, n=self.N, norm='ortho', dim=-1).real

        if self.slack_dims > 0:
            yy_slack = self.Ws(xx[:, self.D:])

            yy = yy_dft + yy_slack
        else:
            yy = yy_dft
        return yy

    def encode(self, labels):
        return labels

    def decode(self, labels):
        return labels

    def decode_logits(self, logits):
        return logits
