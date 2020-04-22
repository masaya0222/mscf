import numpy as np
from mscf.lib.tools import load_library
import ctypes


def c_g_abcd(I1, J1, I2, J2, Ra, Rb, Rc, Rd, ai, bi, ci, di):
    gabcd = np.zeros((I1+1, J1+1, I1+1, J1+1, I2+1, J2+1, I2+1, J2+1))
    gabcd_c = gabcd.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p*((I1+1)*(J1+1)*(I1+1)*(J1+1)*(I2+1)*(J2+1)*(I2+1)*(J2+1))))
    lib = load_library("libcv2e")
    f = getattr(lib, "g_abcd")
    I1_c = ctypes.c_int(I1)
    J1_c = ctypes.c_int(J1)
    I2_c = ctypes.c_int(I2)
    J2_c = ctypes.c_int(J2)
    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Rc = np.array(Rc).astype(np.double)
    Rd = np.array(Rd).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    Rc_c = Rc.ctypes.data_as(ctypes.c_void_p)
    Rd_c = Rd.ctypes.data_as(ctypes.c_void_p)
    ai_c = ctypes.c_double(ai)
    bi_c = ctypes.c_double(bi)
    ci_c = ctypes.c_double(ci)
    di_c = ctypes.c_double(di)
    f(gabcd_c, I1_c, J1_c, I2_c, J2_c, Ra_c, Rb_c, Rc_c, Rd_c, ai_c, bi_c, ci_c, di_c)
    return gabcd


def c_cont_V2(basis_a, basis_b, basis_c, basis_d):
    Ra, I1, a, da = basis_a
    Rb, J1, b, db = basis_b
    Rc, I2, c, dc = basis_c
    Rd, J2, d, dd = basis_d
    V = np.zeros((I1+1, J1+1, I1+1, J1+1, I2+1, J2+1, I2+1, J2+1))
    V_c = V.ctypes.data_as(ctypes.POINTER(
        ctypes.c_void_p * ((I1 + 1) * (J1 + 1) * (I1 + 1) * (J1 + 1) * (I2 + 1) * (J2 + 1) * (I2 + 1) * (J2 + 1))))
    lib = load_library("libcv2e")
    f = getattr(lib, "cont_V2e")
    I1_c = ctypes.c_int(I1)
    J1_c = ctypes.c_int(J1)
    I2_c = ctypes.c_int(I2)
    J2_c = ctypes.c_int(J2)

    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Rc = np.array(Rc).astype(np.double)
    Rd = np.array(Rd).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    Rc_c = Rc.ctypes.data_as(ctypes.c_void_p)
    Rd_c = Rd.ctypes.data_as(ctypes.c_void_p)

    a = np.array(a).astype(np.double)
    b = np.array(b).astype(np.double)
    c = np.array(c).astype(np.double)
    d = np.array(d).astype(np.double)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    c_c = c.ctypes.data_as(ctypes.c_void_p)
    d_c = d.ctypes.data_as(ctypes.c_void_p)

    P1 = ctypes.c_int(len(a))
    Q1 = ctypes.c_int(len(b))
    P2 = ctypes.c_int(len(c))
    Q2 = ctypes.c_int(len(d))

    da = np.array(da).astype(np.double)
    db = np.array(db).astype(np.double)
    dc = np.array(dc).astype(np.double)
    dd = np.array(dd).astype(np.double)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)
    dc_c = dc.ctypes.data_as(ctypes.c_void_p)
    dd_c = dd.ctypes.data_as(ctypes.c_void_p)

    f(V_c,I1_c,J1_c,I2_c,J2_c,P1,Q1,P2,Q2,Ra_c,Rb_c,Rc_c,Rd_c,a_c,b_c,c_c,d_c,da_c,db_c,dc_c,dd_c)
    return V


def c_V2e_lm(basis_a, basis_b, basis_c, basis_d):
    Ra, I1, a, da = basis_a
    Rb, J1, b, db = basis_b
    Rc, I2, c, dc = basis_c
    Rd, J2, d, dd = basis_d
    V = np.zeros((2*I1+1, 2*J1+1, 2*I2+1, 2*J2+1))
    V_c = V.ctypes.data_as(ctypes.POINTER(
        ctypes.c_void_p * ((2*I1+1)*(2*J1+1)*(2*I2+1)*(2*J2+1))))
    lib = load_library("libcv2e")
    f = getattr(lib, "V2e_lm")
    I1_c = ctypes.c_int(I1)
    J1_c = ctypes.c_int(J1)
    I2_c = ctypes.c_int(I2)
    J2_c = ctypes.c_int(J2)

    Ra = np.array(Ra).astype(np.double)
    Rb = np.array(Rb).astype(np.double)
    Rc = np.array(Rc).astype(np.double)
    Rd = np.array(Rd).astype(np.double)
    Ra_c = Ra.ctypes.data_as(ctypes.c_void_p)
    Rb_c = Rb.ctypes.data_as(ctypes.c_void_p)
    Rc_c = Rc.ctypes.data_as(ctypes.c_void_p)
    Rd_c = Rd.ctypes.data_as(ctypes.c_void_p)

    a = np.array(a).astype(np.double)
    b = np.array(b).astype(np.double)
    c = np.array(c).astype(np.double)
    d = np.array(d).astype(np.double)
    a_c = a.ctypes.data_as(ctypes.c_void_p)
    b_c = b.ctypes.data_as(ctypes.c_void_p)
    c_c = c.ctypes.data_as(ctypes.c_void_p)
    d_c = d.ctypes.data_as(ctypes.c_void_p)

    P1 = ctypes.c_int(len(a))
    Q1 = ctypes.c_int(len(b))
    P2 = ctypes.c_int(len(c))
    Q2 = ctypes.c_int(len(d))

    da = np.array(da).astype(np.double)
    db = np.array(db).astype(np.double)
    dc = np.array(dc).astype(np.double)
    dd = np.array(dd).astype(np.double)
    da_c = da.ctypes.data_as(ctypes.c_void_p)
    db_c = db.ctypes.data_as(ctypes.c_void_p)
    dc_c = dc.ctypes.data_as(ctypes.c_void_p)
    dd_c = dd.ctypes.data_as(ctypes.c_void_p)

    f(V_c, I1_c, J1_c, I2_c, J2_c, P1, Q1, P2, Q2, Ra_c, Rb_c, Rc_c, Rd_c, a_c, b_c, c_c, d_c, da_c, db_c, dc_c, dd_c)
    return V


def c_get_v2e(mol):
    V = np.zeros((mol.basis_num, mol.basis_num, mol.basis_num, mol.basis_num))
    V_c = V.ctypes.data_as(ctypes.POINTER(
        ctypes.c_void_p * (mol.basis_num*mol.basis_num*mol.basis_num*mol.basis_num)))
    basis = mol.basis
    basis_len = len(basis)
    c_R = (ctypes.c_void_p * basis_len)()
    c_l = (ctypes.c_int * basis_len)()
    c_a = (ctypes.c_void_p * basis_len)()
    c_da = (ctypes.c_void_p * basis_len)()
    c_P = (ctypes.c_int * basis_len)()
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
    lib = load_library("libcv2e")
    f = getattr(lib, "get_v2e")
    f(V_c, c_R, c_l, c_a, c_da, c_P, c_basis_len, c_basis_num)
    return V

