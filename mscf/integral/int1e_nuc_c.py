import numpy as np
from mscf.lib.tools import load_library
import ctypes


def c_V_ijklm(I, J, Ra, Rb, Rc_list, Zc_list, ai, bi):
    V = np.zeros((I+1, J+1, I+1, J+1)).astype(np.double)
    V_c = V.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I+1)*(J+1)*(I+1)*(J+1))))
    lib = load_library("libcnuc")
    f = getattr(lib, "V_ijklmn")
    I_c = ctypes.c_int(I)
    J_c = ctypes.c_int(J)
    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    nuc_num = len(Zc_list)
    nuc_num_c = ctypes.c_int(nuc_num)
    Rc_list = np.array(Rc_list).astype(np.double)
    Rc_list_c = Rc_list.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*(3*nuc_num)))
    Zc_list = np.array(Zc_list).astype(np.double)
    Zc_list_c = Zc_list.ctypes.data_as(ctypes.c_void_p)
    ai_c = ctypes.c_double(ai)
    bi_c = ctypes.c_double(bi)
    f(V_c, I_c, J_c, Ra_c, Rb_c, Rc_list_c, Zc_list_c, ai_c, bi_c, nuc_num_c)
    return V


def c_cont_V1e(basis_a, basis_b, Rc_list, Zc_list):
    Ra, I, a, da = basis_a
    Rb, J, b, db = basis_b
    V = np.zeros((I + 1, J + 1, I + 1, J + 1)).astype(np.double)
    V_c = V.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * ((I + 1) * (J + 1) * (I + 1) * (J + 1))))
    nuc_num = len(Zc_list)
    nuc_num_c = ctypes.c_int(nuc_num)
    Rc_list = np.array(Rc_list).astype(np.double)
    Rc_list_c = Rc_list.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (3 * nuc_num)))
    Zc_list = np.array(Zc_list).astype(np.double)
    Zc_list_c = Zc_list.ctypes.data_as(ctypes.c_void_p)
    lib = load_library("libcnuc")
    f = getattr(lib, "cont_V1e")
    I_c = ctypes.c_int(I)
    J_c = ctypes.c_int(J)
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
    f(V_c, I_c, J_c, P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c, Rc_list_c, Zc_list_c, nuc_num_c)
    return V


def c_V1e_lm(basis_a, basis_b, Rc_list, Zc_list):
    Ra, la, a, da = basis_a
    Rb, lb, b, db = basis_b
    V = np.zeros((2 * la+1, 2 * lb+1)).astype(np.double)
    V_c = V.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * ((2 * la + 1) * (2 * lb + 1))))
    nuc_num = len(Zc_list)
    nuc_num_c = ctypes.c_int(nuc_num)
    Rc_list = np.array(Rc_list).astype(np.double)
    Rc_list_c = Rc_list.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (3 * nuc_num)))
    Zc_list = np.array(Zc_list).astype(np.double)
    Zc_list_c = Zc_list.ctypes.data_as(ctypes.c_void_p)
    lib = load_library("libcnuc")
    f = getattr(lib, "V1e_lm")
    I_c = ctypes.c_int(la)
    J_c = ctypes.c_int(lb)
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
    f(V_c, I_c, J_c, P, Q, Ra_c, Rb_c, a_c, b_c, da_c, db_c, Rc_list_c, Zc_list_c, nuc_num_c)
    return V


def c_get_v1e(mol):
    V = np.zeros((mol.basis_num, mol.basis_num))
    V_c = V.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (mol.basis_num * mol.basis_num)))

    Rc_list, Zc_list = [], []
    for nuc in mol.nuc:
        Zc_list.append(nuc[0])
        Rc_list.append(nuc[1:])
    nuc_num = len(Zc_list)
    nuc_num_c = ctypes.c_int(nuc_num)
    Rc_list = np.array(Rc_list).astype(np.double)
    Rc_list_c = Rc_list.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p * (3 * nuc_num)))
    Zc_list = np.array(Zc_list).astype(np.double)
    Zc_list_c = Zc_list.ctypes.data_as(ctypes.c_void_p)

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
    lib = load_library("libcnuc")
    f = getattr(lib, "get_v1e")
    f(V_c, c_R, c_l, c_a, c_da, c_P, c_basis_len, c_basis_num, Rc_list_c, Zc_list_c, nuc_num_c)
    return V
