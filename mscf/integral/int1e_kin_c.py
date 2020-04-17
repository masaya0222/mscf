import numpy as np
from mscf.lib.tools import load_library
import ctypes
from mscf.mole.mole import Mole


def c_T_ij(I, J, Ax, Bx, ai, bi):
    Sij = np.zeros((I + 1, J + 1)).astype(np.double)
    Sij_c = Sij.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I + 1)*(J + 1))))
    Tij = np.zeros((I + 1, J + 1)).astype(np.double)
    Tij_c = Tij.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * ((I + 1) * (J + 1))))
    lib = load_library("libckin")
    f = getattr(lib, "T_ij")
    I_c = ctypes.c_int(I)
    J_c = ctypes.c_int(J)
    Ax_c = ctypes.c_double(Ax)
    Bx_c = ctypes.c_double(Bx)
    ai_c = ctypes.c_double(ai)
    bi_c = ctypes.c_double(bi)
    f(Tij_c, Sij_c, I_c, J_c, Ax_c, Bx_c, ai_c, bi_c)
    return Tij


def c_cont_Tij(basis_a, basis_b):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    Tab = np.zeros((I+1, J+1, I+1, J+1))
    Tab_c = (ctypes.c_void_p*((I+1)*(J+1)*(I+1)*(J+1)))()
    Tab_c = Tab.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I+1)*(J+1)*(I+1)*(J+1))))
    lib = load_library("libckin")
    f = getattr(lib, "cont_Tij")
    a = np.array(a).astype(np.double)
    b = np.array(b).astype(np.double)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    P = ctypes.c_int(len(a))
    Q = ctypes.c_int(len(b))
    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    da = np.array(da).astype(np.double)
    db = np.array(db).astype(np.double)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)

    f(Tab_c, ctypes.c_int(I), ctypes.c_int(J), P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c)
    return Tab


def c_T_lm(basis_a, basis_b):
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    T_mamb = np.zeros((2 * la + 1, 2 * lb + 1))
    T_mamb_c = T_mamb.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * ((2 * la + 1) * (2 * lb + 1))))
    lib = load_library("libckin")
    f = getattr(lib, "T_lm")
    a = np.array(a).astype(np.double)
    b = np.array(b).astype(np.double)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    P = ctypes.c_int(len(a))
    Q = ctypes.c_int(len(b))
    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    da = np.array(da).astype(np.double)
    db = np.array(db).astype(np.double)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)
    f(T_mamb_c, ctypes.c_int(la), ctypes.c_int(lb), P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c)
    return T_mamb


def c_get_kin(mol: Mole):
    T = np.zeros((mol.basis_num, mol.basis_num))
    c_T = T.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (mol.basis_num * mol.basis_num)))
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
    R_ = np.array([basis[i][0] for i in range(basis_len)]).astype(np.double)
    a_ = np.array([basis[i][2] for i in range(basis_len)]).astype(np.double)
    da_ = np.array([basis[i][3] for i in range(basis_len)]).astype(np.double)
    for i in range(basis_len):
        c_R[i] = R_[i].ctypes.data_as(ctypes.c_void_p)
        c_a[i] = a_[i].ctypes.data_as(ctypes.c_void_p)
        c_da[i] = da_[i].ctypes.data_as(ctypes.c_void_p)
    c_basis_len = ctypes.c_int(basis_len)
    c_basis_num = ctypes.c_int(mol.basis_num)
    lib = load_library("libckin")
    f = getattr(lib, "get_kin")
    f(c_T, c_R, c_l, c_a, c_da, c_P, c_basis_len, c_basis_num)
    return T
