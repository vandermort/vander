import numpy as np
from itertools import combinations

from vndr.verifier import ArgmaxableSubsetVerifier
from vndr.modules import vander


def k_hot_to_alternating(sv, N):
    dense = [0,] * N
    for s in sv:
        dense[s] = 1
    
    alt = [1]
    for s in dense:
        if s:
            alt.append(-alt[-1])
        else:
            alt.append(alt[-1])
    alt = [i for i, v in enumerate(alt) if v > 0]
    return alt


def test_argmaxable():
    N = 100
    D = 4

    W = vander(N, D)

    ver = ArgmaxableSubsetVerifier(W)

    samples = [[5], [1,2,3], [10, 50, 90]]

    for res in ver(samples):
        print(res['pos_idxs'], res['radius'])

    samples = [k_hot_to_alternating(s, N-1) for s in samples]
    print(samples)

    for res in ver(samples):
        print(res['pos_idxs'], res['radius'])


if __name__ == "__main__":

    test_argmaxable()
