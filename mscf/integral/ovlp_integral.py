# TODO make ovpintegral with Obara-Saika
# TODO transformation primitive to contracted
# TODO transformatino contracted to sphere

from mscf.basis.tools import get_basis
from mscf.mole.mole import Mole
import numpy as np


def S_ij(I, J, Ax, Bx, a, b):
    p = a + b
    mu = a*b/p
    Px = (a*Ax + b + Bx) / p
    Xpa = Px - Ax
    Xpb = Px - Bx
    Xab = Ax - Bx
    S = [[0 for j in range(J+1)] for i in range(I+2)]

    S[0][0] = np.sqrt(np.pi/p) * np.exp(-mu*Xab**2)
    for i in range(I+1):
        S[i+1][0]= Xpa*S[i][0] + (1/(2.0*p))*(i*S[i-1][0])*(i != 1)
    for j in range(J):
        for i in range(I+2):
            if j == J-1 and i == I+1:
                continue
            S[i][j+1] = S[i+1][j] + Xab*S[i][j]
    S = S[:I+1]
    return S


M = Mole([['Li', 0, 0, -0.7], ['H', 0, 0, 0.7]], 'sto3g')

basis = M.basis
a = basis[2]
b = basis[3]
print(a)
print(b)
s = S_ij(a[1], b[1], a[0][0], b[0][0], a[2][0], b[2][0])
print(s)