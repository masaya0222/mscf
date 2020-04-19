import numpy as np
from scipy import special
from mscf.mole.mole import Mole
import math


def V_ijklmn(I, J, Ra, Rb, Rc_list, Zc_list, ai, bi):
    p = ai + bi
    Rp = [(ai * Ra[i] + bi * Rb[i]) / p for i in range(3)]
    Rtuv = np.array(
        [[[0.0 for v in range((I + J) + 1 - t - u)] for u in range((I + J) + 1 - t)] for t in range((I + J) + 1)])
    for Rc, Zc in zip(Rc_list, Zc_list):
        r = R_tuv(I + J, Rp, Rc, p)
        for t in range((I + J) + 1):
            for u in range((I + J) + 1 - t):
                for v in range((I + J) + 1 - t - u):
                    Rtuv[t][u][v] += Zc * r[t][u][v]

    Eijt = E_ij_t(I, J, Ra[0], Rb[0], ai, bi)
    Eklu = E_ij_t(I, J, Ra[1], Rb[1], ai, bi)
    Emnv = E_ij_t(I, J, Ra[2], Rb[2], ai, bi)

    Vijklmn = [[[[0 for l in range(J + 1 - j)] for k in range(I + 1 - i)] for j in range(J + 1)] for i in range(I + 1)]
    for i in range(I + 1):
        for j in range(J + 1):
            for k in range(I + 1 - i):
                for l in range(J + 1 - j):
                    m = I - i - k
                    n = J - j - l
                    for t in range(i + j + 1):
                        for u in range(k + l + 1):
                            for v in range(m + n + 1):
                                Vijklmn[i][j][k][l] += Eijt[i][j][t] * Eklu[k][l][u] * Emnv[m][n][v] * Rtuv[t][u][v]

                    Vijklmn[i][j][k][l] *= -2 * np.pi / p
    return Vijklmn


def R_tuv(IJ, Rp, Rc, p):
    Xpc = Rp[0] - Rc[0]
    Ypc = Rp[1] - Rc[1]
    Zpc = Rp[2] - Rc[2]
    Rpc_2 = Xpc ** 2 + Ypc ** 2 + Zpc ** 2
    R = [np.array([[[0.0 for v in range(IJ - N + 1 - t - u)] for u in range(IJ - N + 1 - t)] for t in
                   range(IJ - N + 1)]) for N in range(IJ + 1)]
    for n in range(IJ + 1):
        R[n][0][0][0] += (-2 * p) ** n * Fn(n, p * Rpc_2)
    for n in reversed(range(IJ)):
        for t in range(IJ - n + 1):
            for u in range(IJ - n + 1 - t):
                for v in range(IJ - n + 1 - t - u):
                    if t >= 1:
                        if t >= 2:
                            R[n][t][u][v] += (t - 1) * R[n + 1][t - 2][u][v]
                        R[n][t][u][v] += Xpc * R[n + 1][t - 1][u][v]
                    elif u >= 1:
                        if u >= 2:
                            R[n][t][u][v] += (u - 1) * R[n + 1][t][u - 2][v]
                        R[n][t][u][v] += Ypc * R[n + 1][t][u - 1][v]
                    elif v >= 1:
                        if v >= 2:
                            R[n][t][u][v] += (v - 1) * R[n + 1][t][u][v - 2]
                        R[n][t][u][v] += Zpc * R[n + 1][t][u][v - 1]
                    else:
                        pass
    return R[0]


def Fn(n, x):
    if x <= 0.10:
        result = 1/(2*n+1)
        result_k = 1
        result_x = 1
        for k in range(1, 7):
            result_x *= (-x)
            result_k *= k
            result += result_x/(result_k*(2*n+2*k+1))
        return result
    value_a = special.gamma(n + 1 / 2)
    value_b = special.gammainc(n + 1 / 2, x)
    value_c = 2 * x ** (n + 1 / 2)
    return value_a * value_b / value_c


def E_ij_t(I, J, Ax, Bx, ai, bi):
    p = ai + bi
    mu = ai * bi / p
    Xab = Ax - Bx
    E = [[[0.0 for t in range(i + j + 1)] for j in range(J + 1)] for i in range(I + 1)]
    E[0][0][0] = np.exp(-mu * Xab ** 2)
    for j in range(J + 1):
        for i in range(I + 1):
            if i == j == 0:
                continue
            if i == 0:
                for t in range(i + j + 1):
                    if 0 <= t - 1:
                        E[i][j][t] += (1 / (2 * p)) * E[i][j - 1][t - 1]
                    if t <= i + j - 1:
                        E[i][j][t] += (ai / p) * Xab * E[i][j - 1][t]
                    if t + 1 <= i + j - 1:
                        E[i][j][t] += (t + 1) * E[i][j - 1][t + 1]
            else:
                for t in range(i + j + 1):
                    if 0 <= t - 1:
                        E[i][j][t] += (1 / (2 * p)) * E[i - 1][j][t - 1]
                    if t <= i + j - 1:
                        E[i][j][t] += (-bi / p) * Xab * E[i - 1][j][t]
                    if t + 1 <= i + j - 1:
                        E[i][j][t] += (t + 1) * E[i - 1][j][t + 1]
    return E


def cont_V1e(basis_a, basis_b, Rc_list, Zc_list):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    w_fact = [1, 1, 3, 15, 105]  # (2*i-1)!! 0<=i<=4
    Vijklmn = [[V_ijklmn(I, J, Ra, Rb, Rc_list, Zc_list, ai, bi) for bi in b] for ai in a]

    contV = [[[[0.0 for l in range(J + 1 - j)] for k in range(I + 1 - i)] for j in range(J + 1)] for i in range(I + 1)]
    for i in range(I + 1):
        for j in range(J + 1):
            for k in range(I + 1 - i):
                for l in range(J + 1 - j):
                    m = I - i - k
                    n = J - j - l
                    for p in range(len(a)):
                        for q in range(len(b)):
                            ans = da[p] * db[q] * Vijklmn[p][q][i][j][k][l]
                            Na = (2 * a[p] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * a[p]) ** I) / (w_fact[I]))  # (w_fact[i] * w_fact[k] * w_fact[m]))
                            Nb = (2 * b[q] / np.pi) ** (3 / 4.0) * np.sqrt(
                                ((4 * b[q]) ** J) / (w_fact[J]))  # (w_fact[j] * w_fact[l] * w_fact[n]))
                            ans *= Na * Nb

                            contV[i][j][k][l] += ans
    return contV


def V1e_lm(basis_a, basis_b, Rc_list, Zc_list):
    V1e_ab = cont_V1e(basis_a, basis_b, Rc_list, Zc_list)
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    max_l = max(la, lb)
    fact = [math.factorial(i) for i in range(2 * max_l + 1)]
    comb = [[special.comb(i, j, exact=True) for j in range(max_l + 1)] for i in range(max_l + 1)]
    V1e_mamb = [[0 for mb in range(2 * lb + 1)] for ma in range(2 * la + 1)]
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

                                    V1e_mamb[i][j] += f * C_a[ma_][ta][ua][math.floor(2 * va_)] * \
                                                      C_b[mb_][tb][ub][math.floor(2 * vb_)] * \
                                                      V1e_ab[pow_xa][pow_xb][pow_ya][pow_yb]

                                if flag:
                                    break
            V1e_mamb[i][j] *= Nma * Nmb
    return V1e_mamb


def get_v1e(mol):
    basis = mol.basis
    Rc_list, Zc_list = [], []
    for nuc in mol.nuc:
        Zc_list.append(nuc[0])
        Rc_list.append(nuc[1:])
    V1e = np.zeros((mol.basis_num, mol.basis_num))
    basis_len = len(basis)
    check = np.zeros((basis_len, basis_len))
    ind_i = 0
    change = [[0], [1, 2, 0], [0, 1, 2, 3, 4]]  # p軌道だけ m=0,1,-1 ( x,y,z)順
    for i in range(basis_len):
        ind_j = 0
        for j in range(basis_len):
            la, lb = basis[i][1], basis[j][1]
            if not(check[i][j]):
                check[i][j] = check[j][i] = 1
                V1elm = V1e_lm(basis[i], basis[j], Rc_list, Zc_list)
                for k in range(2 * la + 1):
                    for l in range(2 * lb + 1):
                        ans = V1elm[k][l]
                        V1e[ind_i + change[la][k]][ind_j + change[lb][l]] = ans # = V1elm[k][l]
                        V1e[ind_j + change[lb][l]][ind_i + change[la][k]] = ans # V1elm[k][l]
            ind_j += 2 * lb + 1
        ind_i += 2 * la + 1
    return V1e
