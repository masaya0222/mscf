# TODO make ovlp_integral with Obara-Saika
# TODO transformation contracted to sphere

from mscf.mole.mole import Mole
import numpy as np
import math
from scipy import special


def S_ij(I, J, Ax, Bx, ai, bi):
    p = ai + bi
    mu = ai * bi / p
    Px = (ai * Ax + bi * Bx) / p
    Xpa = Px - Ax
    # Xpb = Px - Bx
    Xab = Ax - Bx
    S = [[0 for j in range(J + 1)] for i in range(I + 2)]

    S[0][0] = np.sqrt(np.pi / p) * np.exp(-mu * Xab ** 2)
    for i in range(I + 1):
        S[i + 1][0] = Xpa * S[i][0] + (1 / (2.0 * p)) * (i * S[i - 1][0])
    for j in range(J):
        for i in range(I + 2):
            if j == J - 1 and i == I + 1:
                continue
            S[i][j + 1] = S[i + 1][j] + Xab * S[i][j]
    S = S[:I + 1]
    return S


def S_ab(basis_a, basis_b):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    w_fact = [1, 1, 3, 15, 105]  # (2*i-1)!! 0<=i<=4

    Sij = [[S_ij(I, J, Ra[0], Rb[0], ai, bi) for bi in b] for ai in a]
    Skl = [[S_ij(I, J, Ra[1], Rb[1], ai, bi) for bi in b] for ai in a]
    Smn = [[S_ij(I, J, Ra[2], Rb[2], ai, bi) for bi in b] for ai in a]

    # i+k+m = L && j+l+n = L を利用
    Sab = [[[[None for l in range(J + 1)] for k in range(I + 1)] for j in range(J + 1)] for i in
           range(I + 1)]  # L*L*L*L
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
                                ((4 * a[p]) ** I) / (w_fact[i] * w_fact[k] * w_fact[m]))
                            Nb = (2 * b[q] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * b[q]) ** J) / (w_fact[j] * w_fact[l] * w_fact[n]))

                            ans *= Na * Nb
                            Sab[i][j][k][l] += ans
    return Sab


def S_lm(basis_a, basis_b):
    Sab = S_ab(basis_a, basis_b)
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    L = max(la, lb)
    k = la / 2
    fact = [math.factorial(i) for i in range(2 * L + 1)]
    comb = [[special.comb(i, j, exact=True) for j in range(L + 1)] for i in range(L + 1)]
    S_mamb = [[0 for mb in range(2 * lb + 1)] for ma in
              range(2 * la + 1)]  # i番目はma = i-laに対応 i=0=>ma=-la, i=La=>ma=0, i=2*La=>ma=la
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
            vma, vmb = 0, 0
            if ma < 0:
                vma = 1 / 2.0
            if mb < 0:
                vmb = 1 / 2.0

            Nma = (1.0 / (2 ** ma_ * fact[la])) * np.sqrt(2 * fact[la + ma_] * fact[la - ma_] / (2.0 - (ma != 0)))
            Nmb = (1.0 / (2 ** mb_ * fact[lb])) * np.sqrt(2 * fact[lb + mb_] * fact[lb - mb_] / (2.0 - (mb != 0)))

            for ta in range((la - ma_) // 2 + 1):
                for tb in range((lb - mb_) // 2 + 1):
                    for ua in range(ta + 1):
                        for ub in range(tb + 1):
                            flag = False
                            for va in range(ma_ // 2 + 1):
                                for vb in range(mb_ // 2 + 1):
                                    f = -2 * (va + vb % 2) + 1
                                    if ma < 0:
                                        va += 1 / 2.0
                                        if va > (ma_ - 1) // 2 + 1 / 2.0:
                                            flag = True
                                            break
                                    if mb < 0:
                                        vb += 1 / 2.0
                                        if vb > (mb_ - 1) // 2 + 1 / 2.0:
                                            break
                                    pow_xa = math.floor(2 * ta + ma_ - 2 * (ua + va))
                                    pow_xb = math.floor(2 * tb + mb_ - 2 * (ub + vb))
                                    pow_ya = math.floor(2 * (ua + va))
                                    pow_yb = math.floor(2 * (ub + vb))
                                    S_mamb[i][j] += f * C_a[ma_][ta][ua][math.floor(2 * va)] * \
                                                    C_b[mb_][tb][ub][math.floor(2 * vb)] * \
                                                    Sab[pow_xa][pow_xb][pow_ya][pow_yb]
                                if flag:
                                    break
            S_mamb[i][j] *= Nma * Nmb
    return S_mamb


def get_ovlp(mol):
    basis = mol.basis
    S = np.array([[None for j in range(mol.basis_num)] for i in range(mol.basis_num)])
    ind_i = 0
    for i in range(len(basis)):
        ind_j = 0
        for j in range(len(basis)):
            la, lb = basis[i][1], basis[j][1]
            Slm = S_lm(basis[i], basis[j])
            for k in range(2 * la + 1):
                for l in range(2 * lb + 1):
                    S[ind_i + (k+1) % (2 * la + 1)][ind_j + (l + 1) % (2 * lb + 1)] = Slm[k][l]
            ind_j += 2 * lb + 1
        ind_i += 2 * la + 1

    return S


M = Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')

basis = M.basis
a = basis[1]
b = basis[3]

S = get_ovlp(M)
print(S)


from pyscf import gto, scf, ao2mo
import numpy
mol = gto.Mole()
xx = 0.52918

mol.build(
    atom='H 0 0 %f; Li 0 0 %f' %(-0.7 * xx, 0.7* xx),
    basis='sto3g')
S1 = mol.intor('int1e_ovlp')

mol1 = gto.Mole()
mol1.build(
    atom='K 0 0 %f; H 0 0 %f' %(0*xx, 1*xx),
    basis='sto3g',
)
print(mol1.atom)
s1 = mol1.intor('int1e_ovlp')

m = Mole([['K', 0, 0, 0], ['H', 0, 0, 1]], 'sto3g',)
for i in m.basis:
    print(i)
s = get_ovlp(m)
print(s.shape)
print(s1.shape)
ans = 0


for i in range(len(s)):
    for j in range(len(s[0])):
        ans += abs(s[i][j] - s1[i][j])
print(ans)
