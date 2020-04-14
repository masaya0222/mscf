
from mscf.mole.mole import Mole
import numpy as np
import math
from scipy import special
from typing import List


def S_ij(I, J, Ax, Bx, ai, bi) -> List[List[int]]:
    p = ai + bi
    mu = ai * bi / p
    Xab = Ax - Bx
    S = [[0 for j in range(J + 1)] for i in range(I + J + 1)]

    S[0][0] = np.sqrt(np.pi / p) * np.exp(-mu * Xab ** 2)
    for i in range(I + J):
        S[i + 1][0] = (-bi / p) * Xab * S[i][0] + (1 / (2.0 * p)) * (i * S[i - 1][0])
    for j in range(J):
        for i in range(I + J):
            if i + j >= I + J + 1:
                continue
            S[i][j + 1] = S[i + 1][j] + Xab * S[i][j]
    S = S[:I + 1]
    return S


def cont_Sij(basis_a, basis_b) -> List[List[List[List[float]]]]:
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    w_fact = [1, 1, 3, 15, 105]  # (2*i-1)!! 0<=i<=4

    Sij = [[S_ij(I, J, Ra[0], Rb[0], ai, bi) for bi in b] for ai in a]
    Skl = [[S_ij(I, J, Ra[1], Rb[1], ai, bi) for bi in b] for ai in a]
    Smn = [[S_ij(I, J, Ra[2], Rb[2], ai, bi) for bi in b] for ai in a]

    # i+k+m = L && j+l+n = L を利用
    Sab = [[[[0 for l in range(J + 1)] for k in range(I + 1)] for j in range(J + 1)] for i in
           range(I + 1)]  # L*L*L*L Debugのときは0じゃなくてNoneでも可能
    for i in range(I + 1):
        for j in range(J + 1):
            for k in range(I + 1 - i):
                for l in range(J + 1 - j):
                    m = I - i - k
                    n = J - j - l
                    Sab[i][j][k][l] = 0
                    for p in range(len(a)):
                        for q in range(len(b)):
                            ans = da[p] * db[q] * Sij[p][q][i][j] * Skl[p][q][k][l] * Smn[p][q][m][n]
                            Na = (2 * a[p] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * a[p]) ** I) / (w_fact[I]))  # (w_fact[i] * w_fact[k] * w_fact[m]))
                            Nb = (2 * b[q] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * b[q]) ** J) / (w_fact[J]))  # (w_fact[j] * w_fact[l] * w_fact[n]))
                            ans *= Na * Nb

                            Sab[i][j][k][l] += ans
    return Sab


def S_lm(basis_a, basis_b) -> List[List[int]]:
    Sab = cont_Sij(basis_a, basis_b)
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    max_l = max(la, lb)
    fact = [math.factorial(i) for i in range(2 * max_l + 1)]
    comb = [[special.comb(i, j, exact=True) for j in range(max_l + 1)] for i in range(max_l + 1)]
    S_mamb = [[0 for mb in range(2 * lb + 1)] for ma in range(2 * la + 1)]
    # i番目はma = i-laに対応 i=0=>ma=-la, i=La=>ma=0, i=2*La=>ma=la
    C_a = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[la][t] * comb[la - t][ma_ + t] * comb[t][u] * comb[ma_][v]
              if (la >= t >= u and la - t >= ma_ + t and ma_ >= 2 * v / 2.0) else 0
              for v in range(la + 1)] for u in range(la // 2 + 1)] for t in range(la // 2 + 1)] for ma_ in
           range(la + 1)]
    C_b = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[lb][t] * comb[lb - t][mb_ + t] * comb[t][u] * comb[mb_][v]
              if (lb >= t >= u and lb - t >= mb_ + t and mb_ >= 2 * v / 2.0) else 0
              for v in range(lb + 1)] for u in range(lb // 2 + 1)] for t in range(lb // 2 + 1)] for mb_ in
           range(lb + 1)]
    # vはv=v/2.0をあらわす

    for i in range(2 * la + 1):
        for j in range(2 * lb + 1):
            ma, mb = i - la, j - lb
            ma_, mb_ = abs(ma), abs(mb)
            Nma = (1.0 / (2 ** ma_ * fact[la])) * np.sqrt(2 * fact[la + ma_] * fact[la - ma_] / (2.0 - (ma != 0)))
            Nmb = (1.0 / (2 ** mb_ * fact[lb])) * np.sqrt(2 * fact[lb + mb_] * fact[lb - mb_] / (2.0 - (mb != 0)))

            for ta in range((la - ma_) // 2 + 1):
                for tb in range((lb - mb_) // 2 + 1):
                    for ua in range(ta + 1):
                        for ub in range(tb + 1):
                            flag = False
                            for va in range(ma_ // 2 + 1):
                                for vb in range(mb_ // 2 + 1):
                                    f = -2 * ((va + vb) % 2) + 1
                                    va_ = va
                                    vb_ = vb
                                    if ma < 0:
                                        va_ += 1 / 2.0
                                        if va_ > (ma_ - 1) // 2 + 1 / 2.0:
                                            flag = True
                                            break
                                    if mb < 0:
                                        vb_ += 1 / 2.0
                                        if vb_ > (mb_ - 1) // 2 + 1 / 2.0:
                                            break
                                    pow_xa = math.floor(2 * ta + ma_ - 2 * (ua + va_))
                                    pow_xb = math.floor(2 * tb + mb_ - 2 * (ub + vb_))
                                    pow_ya = math.floor(2 * (ua + va_))
                                    pow_yb = math.floor(2 * (ub + vb_))

                                    S_mamb[i][j] += f * C_a[ma_][ta][ua][math.floor(2 * va_)] * \
                                                    C_b[mb_][tb][ub][math.floor(2 * vb_)] * \
                                                    Sab[pow_xa][pow_xb][pow_ya][pow_yb]

                                if flag:
                                    break
            S_mamb[i][j] *= Nma * Nmb
    return S_mamb


def get_ovlp(mol: Mole) -> List[List[int]]:
    basis = mol.basis
    S = np.array([[None for j in range(mol.basis_num)] for i in range(mol.basis_num)])
    ind_i = 0
    change = [[0], [1, 2, 0], [0, 1, 2, 3, 4]]  # p軌道だけ m=0,1,-1 ( x,y,z)順
    for i in range(len(basis)):
        ind_j = 0
        for j in range(len(basis)):
            la, lb = basis[i][1], basis[j][1]
            Slm = S_lm(basis[i], basis[j])
            for k in range(2 * la + 1):
                for l in range(2 * lb + 1):
                    S[ind_i + change[la][k]][ind_j + change[lb][l]] = Slm[k][l]
            ind_j += 2 * lb + 1
        ind_i += 2 * la + 1
    return S


