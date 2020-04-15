import numpy as np
from ctypes import *
import ctypes.util
import os
import ctypes
from pyscf import lib


def load_library(libname):
    _loaderpath = os.path.dirname(__file__)
    return np.ctypeslib.load_library(libname, _loaderpath)


lib = load_library("libctest")
f = getattr(lib, "main")
n = np.zeros((3,3,3)).astype(np.int32)
n_h, n_w , nn= n.shape
c = n.ctypes.data_as(ctypes.POINTER(((ctypes.c_int32 * n_h) * n_w) * nn)).contents
c[0][0][1] += 1
print(n)
f2 = getattr(lib, "change")
f2(c)
#f()
print(n)
print("EE")
"""
fdrv = getattr(lib, 'add_matrix')

row = 20
col = 5
n = 5
matrix = np.random.rand(row, col)

_DOUBLE_PP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

fdrv.argtypes = [_DOUBLE_PP, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

fdrv.restype = None

tp = np.uintp
mpp = (matrix.__array_interface__['data'][0] + np.arange(matrix.shape[0])*matrix.strides[0]).astype(tp)

n = ctypes.c_int(n)
row = ctypes.c_int(row)
col = ctypes.c_int(col)

print("before:", matrix)

fdrv(mpp, row, col, n)

print("after:", matrix)
"""
"""
def S_ij(I, J, Ax, Bx, ai, bi):
    S = np.zeros((3,2))
    S[0][0] = 1
    S[0][1] = 2
    S[1][0] = 3
    S[1][1] = 4
    S[2][0] = 5
    S[2][1] = 6
    S_c = (ctypes.c_void_p*3)()
    for i in range(3):
        S_c[i] = S[i].ctypes.data_as(ctypes.c_void_p)

    lib = load_library("libcovlp")
    f = getattr(lib, "S_ij")
    print(S.shape)

    f(S_c, 3, 2)
    print(S)
extern "C" void S_ij(double **S, int I, int J) {
  cout << "hello" << endl;
  cout << M_PI << endl;
  cout << *S <<" " << **S << endl;
  cout << *S+1 << ", " << *(*S+1) << endl;
  cout << *(S+1) << " " << **(S+1) << endl;
  cout << *(S+1)+1 << " " << *(*(S+1)+1) << endl;
  cout << *(S+2) << " " << **(S+2) << endl;
  cout << *(S+2)+1 << " " << *(*(S+2)+1) << endl;
  cout << endl << endl;

  cout << S[0][0] << " " << S[0][1] << endl;
  cout << S[1][0] << " " << S[1][1] << endl;
  for(int i=0; i<I; i++){
    for(int j=0; j<J; j++){
      S[i][j]+=1;
    }
  }

  cout << "finish" << endl;
}
"""