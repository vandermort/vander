import numpy as np
import torch

from vndr import MLBLPath
from vndr.components import KSparseClassifier
from vndr.modules import gale
from vndr.verifier import ArgmaxableSubsetVerifier


def test_init_vander():
    D = 10
    N = 1000 
    K = 10

    ks = KSparseClassifier(D, N, K, slack_dims=0, param='vander')
    W = ks.compute_W().detach().cpu().numpy()

    x = np.zeros(2*K + 1)
    x[0] = -1
    assert np.array_equal(W.dot(x) > 0, np.zeros(N))


def test_init_gale():
    D = 10
    N = 10
    K = 2

    ks = KSparseClassifier(D, N, K, slack_dims=0, param='gale')
    W = ks.compute_W().detach().cpu().numpy()

    x = np.zeros(2*K + 1)
    x[0] = -1
    assert np.array_equal(W.dot(x) > 0, np.zeros(N))



if __name__ == "__main__":
    N, D = 10, 1000

    torch.manual_seed(13)

    test_init_vander()
    test_init_gale()
