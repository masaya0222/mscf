# TODO make ovpintegral with Obara-Saika
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

def S_ab(basis_a, basis_b):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    wfact = [1, 1, 3, 15, 105] # (2*i-1)!! 0<=i<=4

    Sij = [[S_ij(I, J, Ra[0], Rb[0], ai, bi) for bi in b] for ai in a]
    Skl = [[S_ij(I, J, Ra[1], Rb[1], ai, bi) for bi in b] for ai in a]
    Smn = [[S_ij(I, J, Ra[2], Rb[2], ai, bi) for bi in b] for ai in a]

    # i+k+m = L && j+l+n = L を利用
    Sab = [[[[None for l in range(J+1)] for k in range(I+1)] for j in range(J+1)] for i in range(I+1)] #L*L*L*L
    for i in range(I+1):
        for j in range(J+1):
            for k in range(I+1-i):
                for l in range(J+1-j):
                    m = I - i - k
                    n = J - j - l
                    Sab[i][j][k][l] = 0
                    for p in range(len(a)):
                        for q in range(len(b)):
                            ans = da[p]*db[q]*Sij[p][q][i][j]*Skl[p][q][k][l]*Smn[p][q][m][n]
                            Na = (2 * a[p] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * a[p]) ** I) / (wfact[i] * wfact[k] * wfact[m]))
                            Nb = (2 * b[q] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * b[q]) ** J) / (wfact[j] * wfact[l] * wfact[n]))

                            ans *= Na * Nb
                            Sab[i][j][k][l] += ans
    return Sab


M = Mole([['H', 0, 0, -0.7], ['H', 0, 0, 0.7]], 'sto3g')

basis = M.basis
a = basis[0]
b = basis[1]
print(a)
print(b)

Saa = S_ab(a,a)
Sab = S_ab(a,b)
Sba = S_ab(b,a)
Sbb = S_ab(b,b)
print(Saa, Sab, Sba, Sbb)

