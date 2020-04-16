import numpy as np
from mscf.lib.tools import load_library
import ctypes
from mscf.mole.mole import Mole


def c_S_ij(I, J, Ax, Bx, ai, bi):
    S = np.zeros((I + 1, J + 1)).astype(np.double)
    S_c = S.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I+1)*(J+1))))
    lib = load_library("libcovlp")
    f = getattr(lib, "S_ij")
    I_c = ctypes.c_int(I)
    J_c = ctypes.c_int(J)
    Ax_c = ctypes.c_double(Ax)
    Bx_c = ctypes.c_double(Bx)
    ai_c = ctypes.c_double(ai)
    bi_c = ctypes.c_double(bi)
    f(S_c, I_c, J_c, Ax_c, Bx_c, ai_c, bi_c)
    print(S[1][0])
    return S


def c_cont_Sij(basis_a, basis_b):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    Sab = np.zeros((I+1, J+1, I+1, J+1))
    Sab_c = (ctypes.c_void_p*((I+1)*(J+1)*(I+1)*(J+1)))()
    Sab_c = Sab.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I+1)*(J+1)*(I+1)*(J+1))))
    lib = load_library("libcovlp")
    f = getattr(lib, "cont_Sij")
    a = np.array(a)
    b = np.array(b)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    P = ctypes.c_int(len(a))
    Q = ctypes.c_int(len(b))
    Ra = np.array(Ra)
    Rb = np.array(Rb)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    da = np.array(da)
    db = np.array(db)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)

    f(Sab_c, ctypes.c_int(I), ctypes.c_int(J), P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c)
    return Sab


def c_S_lm(basis_a, basis_b):
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    S_mamb = np.zeros((2 * la + 1, 2 * lb + 1))
    S_mamb_c = S_mamb.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * ((2 * la + 1) * (2 * lb + 1))))
    lib = load_library("libcovlp")
    f = getattr(lib, "S_lm")
    a = np.array(a)
    b = np.array(b)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    P = ctypes.c_int(len(a))
    Q = ctypes.c_int(len(b))
    Ra = np.array(Ra)
    Rb = np.array(Rb)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    da = np.array(da)
    db = np.array(db)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)
    f(S_mamb_c, ctypes.c_int(la), ctypes.c_int(lb), P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c)
    return S_mamb


def c_get_ovlp(mol: Mole):
    S = np.zeros((mol.basis_num, mol.basis_num))
    c_S = S.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (mol.basis_num * mol.basis_num)))
    basis = mol.basis
    basis_len = len(basis)
    c_R = (ctypes.c_void_p*basis_len)()
    c_l = (ctypes.c_int*basis_len)()
    c_a = (ctypes.c_void_p*basis_len)()
    c_da = (ctypes.c_void_p*basis_len)()
    c_P = (ctypes.c_int*basis_len)()
    for i in range(basis_len):
        c_l[i] = basis[i][1]
        c_P[i] = len(basis[i][2])
    R_ = np.array([basis[i][0] for i in range(basis_len)])
    a_ = np.array([basis[i][2] for i in range(basis_len)])
    da_ = np.array([basis[i][3] for i in range(basis_len)])
    for i in range(basis_len):
        c_R[i] = R_[i].ctypes.data_as(ctypes.c_void_p)
        c_a[i] = a_[i].ctypes.data_as(ctypes.c_void_p)
        c_da[i] = da_[i].ctypes.data_as(ctypes.c_void_p)
    c_basis_len = ctypes.c_int(basis_len)
    c_basis_num = ctypes.c_int(mol.basis_num)
    lib = load_library("libcovlp")
    f = getattr(lib, "get_ovlp")
    f(c_S, c_R, c_l, c_a, c_da, c_P, c_basis_len, c_basis_num)
    return S
