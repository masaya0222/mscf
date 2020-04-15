from mscf.integral.int1e_nuc import R_tuv, E_ij_t
from mscf.integral.int1e_nuc import Fn
from mscf.mole.mole import Mole
import numpy as np
import math
from scipy import special


def g_abcd(I1, J1, I2, J2, Ra, Rb, Rc, Rd, ai, bi, ci, di):
    p = ai + bi
    q = ci + di
    alpha = p * q / (p + q)
    Rp = [(ai * Ra[i] + bi * Rb[i]) / p for i in range(3)]
    Rq = [(ci * Rc[i] + di * Rd[i]) / q for i in range(3)]

    Rtuv = R_tuv(I1 + J1 + I2 + J2, Rp, Rq, alpha)
    gabcd = [[[[[[[[0.0 for l2 in range(J2 + 1 - j2)] for k2 in range(I2 + 1 - i2)] for j2 in range(J2 + 1)] for i2 in range(I2 + 1)] for l1 in range(J1 + 1 - j1)] for k1 in range(I1 + 1 - i1)] for j1 in range(J1 + 1)] for i1 in range(I1 + 1)]

    Eijt1 = E_ij_t(I1, J1, Ra[0], Rb[0], ai, bi)
    Eklu1 = E_ij_t(I1, J1, Ra[1], Rb[1], ai, bi)
    Emnv1 = E_ij_t(I1, J1, Ra[2], Rb[2], ai, bi)

    Eijt2 = E_ij_t(I2, J2, Rc[0], Rd[0], ci, di)
    Eklu2 = E_ij_t(I2, J2, Rc[1], Rd[1], ci, di)
    Emnv2 = E_ij_t(I2, J2, Rc[2], Rd[2], ci, di)
    for i1 in range(I1 + 1):
        for j1 in range(J1 + 1):
            for k1 in range(I1 + 1 - i1):
                for l1 in range(J1 + 1 - j1):
                    m1 = I1 - i1 - k1
                    n1 = J1 - j1 - l1
                    for i2 in range(I2 + 1):
                        for j2 in range(J2 + 1):
                            for k2 in range(I2 + 1 - i2):
                                for l2 in range(J2 + 1 - j2):
                                    m2 = I2 - i2 - k2
                                    n2 = J2 - j2 - l2
                                    ans1 = 0.0
                                    for t1 in range(i1 + j1 + 1):
                                        for u1 in range(k1 + l1 + 1):
                                            for v1 in range(m1 + n1 + 1):
                                                Etuv = Eijt1[i1][j1][t1] * Eklu1[k1][l1][u1] * Emnv1[m1][n1][v1]
                                                ans2 = 0.0
                                                for t2 in range(i2 + j2 + 1):
                                                    for u2 in range(k2 + l2 + 1):
                                                        for v2 in range(m2 + n2 + 1):
                                                            f = 1 - 2 * ((t2 + u2 + v2) % 2)  # (-1)**(t2+u2+v2)
                                                            ans2 += f * Eijt2[i2][j2][t2] * Eklu2[k2][l2][u2] * Emnv2[m2][n2][v2] * Rtuv[t1 + t2][u1 + u2][v1 + v2] #
                                                ans1 += ans2 * Etuv
                                    gabcd[i1][j1][k1][l1][i2][j2][k2][l2] = ans1 * (2.0 * np.pi ** (5 / 2.0))/(p * q * np.sqrt(p + q))
    return gabcd


def cont_V2(basis_a, basis_b, basis_c, basis_d):
    Ra, I1, a, da = basis_a
    Rb, J1, b, db = basis_b
    Rc, I2, c, dc = basis_c
    Rd, J2, d, dd = basis_d
    w_fact = [1, 1, 3, 15, 105]  # (2*i-1)!! 0<=i<=4
    gabcd = [[[[g_abcd(I1, J1, I2, J2, Ra, Rb, Rc, Rd, ai, bi, ci, di) for di in d] for ci in c] for bi in b] for ai in a]
    contV = [[[[[[[[0.0 for l2 in range(J2 + 1 - j2)] for k2 in range(I2 + 1 - i2)] for j2 in range(J2 + 1)] for i2 in range(I2 + 1)] for l1 in range(J1 + 1 - j1)] for k1 in range(I1 + 1 - i1)] for j1 in range(J1 + 1)] for i1 in range(I1 + 1)]
    for i1 in range(I1 + 1):
        for j1 in range(J1 + 1):
            for k1 in range(I1 + 1 - i1):
                for l1 in range(J1 + 1 - j1):
                    m1 = I1 - i1 - k1
                    n1 = J1 - j1 - l1
                    for i2 in range(I2 + 1):
                        for j2 in range(J2 + 1):
                            for k2 in range(I2 + 1 - i2):
                                for l2 in range(J2 + 1 - j2):
                                    m2 = I2 - i2 - k2
                                    l2 = J2 - j2 - l2
                                    ans = 0.0
                                    for p1 in range(len(a)):
                                        for q1 in range(len(b)):
                                            for p2 in range(len(c)):
                                                for q2 in range(len(d)):
                                                    Na = (2 * a[p1] / np.pi) ** (3 / 4.0) * np.sqrt(
                                                        ((4 * a[p1]) ** I1) / (w_fact[I1]))
                                                    Nb = (2 * b[q1] / np.pi) ** (3 / 4.0) * np.sqrt(
                                                        ((4 * b[q1]) ** J1) / (w_fact[J1]))
                                                    Nc = (2 * c[p2] / np.pi) ** (3 / 4.0) * np.sqrt(
                                                        ((4 * c[p2]) ** I2) / (w_fact[I2]))
                                                    Nd = (2 * d[q2] / np.pi) ** (3 / 4.0) * np.sqrt(
                                                        ((4 * d[q2]) ** J2) /(w_fact[J2]))

                                                    ans += da[p1] * db[q1] * dc[p2] * dd[q2] * gabcd[p1][q1][p2][q2][i1][j1][k1][l1][i2][j2][k2][l2] * Na * Nb * Nc * Nd
                                    contV[i1][j1][k1][l1][i2][j2][k2][l2] = ans
    return contV


def V2e_lm(basis_a, basis_b, basis_c, basis_d):
    V2e_abcd = cont_V2(basis_a, basis_b, basis_c, basis_d)
    Ra, I1, a, da = basis_a
    Rb, J1, b, db = basis_b
    Rc, I2, c, dc = basis_c
    Rd, J2, d, dd = basis_d
    max_l = max(I1, J1, I2, J2)
    fact = [math.factorial(i) for i in range(2 * max_l + 1)]
    comb = [[special.comb(i, j, exact=True) for j in range(max_l + 1)] for i in range(max_l + 1)]
    V2mamb = [[[[0.0 for md in range(2 * J2 + 1)] for mc in range(2 * I2 + 1)] for mb in range(2 * J1 + 1)] for ma in range(2 * I1 + 1)]
    # i番目はma = i-laに対応 i=0=>ma=-la, i=La=>ma=0, i=2*La=>ma=la
    C_a = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[I1][t] * comb[I1 - t][ma_ + t] * comb[t][u] * comb[ma_][v]
              if (I1 >= t >= u and I1 - t >= ma_ + t and ma_ >= 2 * v / 2.0) else 0
              for v in range(I1 + 1)] for u in range(I1 // 2 + 1)] for t in range(I1 // 2 + 1)] for ma_ in
           range(I1 + 1)]
    C_b = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[J1][t] * comb[J1 - t][mb_ + t] * comb[t][u] * comb[mb_][v]
              if (J1 >= t >= u and J1 - t >= mb_ + t and mb_ >= 2 * v / 2.0) else 0
              for v in range(J1 + 1)] for u in range(J1 // 2 + 1)] for t in range(J1 // 2 + 1)] for mb_ in
           range(J1 + 1)]
    C_c = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[I2][t] * comb[I2 - t][mc_ + t] * comb[t][u] * comb[mc_][v]
              if (I2 >= t >= u and I2 - t >= mc_ + t and mc_ >= 2 * v / 2.0) else 0
              for v in range(I2 + 1)] for u in range(I2 // 2 + 1)] for t in range(I2 // 2 + 1)] for mc_ in
           range(I2 + 1)]
    C_d = [[[[(-2 * (t % 2) + 1.0) * (1.0 / 4 ** t) * comb[J2][t] * comb[J2 - t][md_ + t] * comb[t][u] * comb[md_][v]
              if (J2 >= t >= u and J2 - t >= md_ + t and md_ >= 2 * v / 2.0) else 0
              for v in range(J2 + 1)] for u in range(J2 // 2 + 1)] for t in range(J2 // 2 + 1)] for md_ in
           range(J2 + 1)]
    # vはv=v/2.0をあらわす

    for i in range(2 * I1 + 1):
        for j in range(2 * J1 + 1):
            for k in range(2 * I2 + 1):
                for l in range(2 * J2 + 1):
                    ma, mb, mc, md = i - I1, j - J1, k - I2, l - J2
                    ma_, mb_, mc_, md_ = abs(ma), abs(mb), abs(mc), abs(md)
                    Nma = (1.0 / (2 ** ma_ * fact[I1])) * np.sqrt(2 * fact[I1 + ma_] * fact[I1 - ma_] / (2.0 - (ma != 0)))
                    Nmb = (1.0 / (2 ** mb_ * fact[J1])) * np.sqrt(2 * fact[J1 + mb_] * fact[J1 - mb_] / (2.0 - (mb != 0)))
                    Nmc = (1.0 / (2 ** mc_ * fact[I2])) * np.sqrt(2 * fact[I2 + mc_] * fact[I2 - mc_] / (2.0 - (mc != 0)))
                    Nmd = (1.0 / (2 ** md_ * fact[J2])) * np.sqrt(2 * fact[J2 + md_] * fact[J2 - md_] / (2.0 - (md != 0)))
                    for ta in range((I1 - ma_) // 2 + 1):
                        for tb in range((J1 - mb_) // 2 + 1):
                            for tc in range((I2 - mc_) // 2 + 1):
                                for td in range((J2 - md_) // 2 + 1):
                                    for ua in range(ta + 1):
                                        for ub in range(tb + 1):
                                            for uc in range(tc + 1):
                                                for ud in range(td + 1):
                                                    for va in range(ma_ // 2 + 1):
                                                        for vb in range(mb_ // 2 + 1):
                                                            for vc in range(mc_ // 2 + 1):
                                                                for vd in range(md_ // 2 + 1):
                                                                    f = 1 - 2 * ((va + vb + vc + vd) % 2)
                                                                    va_, vb_, vc_, vd_ = va, vb, vc, vd
                                                                    if ma < 0:
                                                                        va_ += 1 / 2.0
                                                                        if va_ > (ma_ - 1) // 2 + 1 / 2.0:
                                                                            break
                                                                    if mb < 0:
                                                                        vb_ += 1 / 2.0
                                                                        if vb_ > (mb_ - 1) // 2 + 1 / 2.0:
                                                                            break
                                                                    if mc < 0:
                                                                        vc_ += 1 / 2.0
                                                                        if vc_ > (mc_ - 1) // 2 + 1 / 2.0:
                                                                            break
                                                                    if md < 0:
                                                                        vd_ += 1 / 2.0
                                                                        if vd_ > (md_ - 1) // 2 + 1 / 2.0:
                                                                            break
                                                                    pow_xa = math.floor(2 * ta + ma_ - 2 * (ua + va_))
                                                                    pow_xb = math.floor(2 * tb + mb_ - 2 * (ub + vb_))
                                                                    pow_xc = math.floor(2 * tc + mc_ - 2 * (uc + vc_))
                                                                    pow_xd = math.floor(2 * td + md_ - 2 * (ud + vd_))
                                                                    pow_ya = math.floor(2 * (ua + va_))
                                                                    pow_yb = math.floor(2 * (ub + vb_))
                                                                    pow_yc = math.floor(2 * (uc + vc_))
                                                                    pow_yd = math.floor(2 * (ud + vd_))

                                                                    V2mamb[i][j][k][l] += f * C_a[ma_][ta][ua][math.floor(2 * va_)] * \
                                                                                              C_b[mb_][tb][ub][math.floor(2 * vb_)] * \
                                                                                              C_c[mc_][tc][uc][math.floor(2 * vc_)] * \
                                                                                              C_d[md_][td][ud][math.floor(2 * vd_)] * \
                                                                                              V2e_abcd[pow_xa][pow_xb][pow_ya][pow_yb][pow_xc][pow_xd][pow_yc][pow_yd]
                    V2mamb[i][j][k][l] *= Nma * Nmb * Nmc * Nmd
    return V2mamb


def get_v2e(mol):
    basis = mol.basis
    V2e = np.array([[[[0.0 for l in range(mol.basis_num)] for k in range(mol.basis_num)] for j in range(mol.basis_num)] for i in range(mol.basis_num)])
    change = [[0], [1, 2, 0], [0, 1, 2, 3, 4]]  # p軌道だけ m=0,1,-1 ( x,y,z)順
    ind_i = 0
    for i in range(len(basis)):
        ind_j = 0
        for j in range(len(basis)):
            ind_k = 0
            for k in range(len(basis)):
                ind_l = 0
                for l in range(len(basis)):
                    I1, J1, I2, J2 = basis[i][1], basis[j][1], basis[k][1], basis[l][1]
                    V2elm = V2e_lm(basis[i], basis[j], basis[k], basis[l])
                    for ma in range(2 * I1 + 1):
                        for mb in range(2 * J1 + 1):
                            for mc in range(2 * I2 + 1):
                                for md in range(2 * J2 + 1):
                                    V2e[ind_i + change[I1][ma]][ind_j + change[J1][mb]][ind_k + change[I2][mc]][ind_l + change[J2][md]] += V2elm[ma][mb][mc][md]
                    ind_l += 2 * J2 + 1
                ind_k += 2 * I2 + 1
            ind_j += 2 * J1 + 1
        ind_i += 2 * I1 + 1
    return V2e

from pyscf import gto



