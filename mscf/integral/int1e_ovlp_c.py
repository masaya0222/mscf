import numpy as np
from mscf.lib.tools import load_library
import ctypes


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


def test():
    #S = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    S = np.array([[1,2],[3,4]]).astype(np.double)
    #_D_PTR = ctypes.POINTER(ctypes.c_double)
    #S_ptr = S.ctypes.data_as(ctypes.POINTER((ctypes.c_double * 2)*2)).contents
    S_ptr = S.ctypes.data_as(ctypes.c_void_p)
    lib = load_library("libcovlp")
    f = getattr(lib, "test")
    f(S_ptr)
    print(S)

    #S_ptr = S.ctypes.data_as(ctypes.c_void_p)




import time
from mscf.integral.int1e_ovlp import S_ij, cont_Sij, S_lm
from mscf.mole.mole import Mole
mol = Mole([['I', 0, 0, -0.7], ['I', 0, 0, 0.7]], 'sto3g')
ba = mol.basis[10]
bb = mol.basis[21]
S1 = S_lm(ba,bb)
S2 = c_S_lm(ba, bb)
print(S1)
print(S2)

"""
start1 = time.time()
for i in range(int(1e4)):
    S1 = cont_Sij(ba,bb)
time1 = time.time() - start1
#print(S1)

np.set_printoptions(precision=16)
start2 = time.time()
for j in range(int(1e4)):
    S2 = c_cont_Sij(ba,bb)
time2 = time.time() - start2
print("time1: ", time1)
print("time2: ", time2)
"""
#print(S2)
