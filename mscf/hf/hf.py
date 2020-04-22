from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.stats import unitary_group
from mscf.integral.int1e_ovlp_c import c_get_ovlp
from mscf.integral.int1e_kin_c import c_get_kin
from mscf.integral.int1e_nuc_c import c_get_v1e
from mscf.integral.int2e_c import c_get_v2e


class HF:
    def __init__(self, mol):
        self.mol = mol
        self.get_ovlp_ao()
        self.ovlp_ao = self.get_ovlp_ao()
        self.kin_ao = self.get_kin_ao()
        self.v1e_ao = self.get_v1e_ao()
        self.v2e_ao = self.get_v2e_ao()
        self.hcore = self.kin_ao + self.v1e_ao
        self.mo_num = self.mol.basis_num
        self.can_rhf = True
        self.occ, self.occ_num = self.make_occ()
        self.coeff = self.init_guess()
        self.mo_ene = None
        self.dense_ao = self.get_dense_ao()
        self.fock_ao = self.get_fock_ao()
        self.max_iteration = 10**2
        self.elec_ene = None
        self.nuc_ene = self.get_nuc_ene(self.mol.nuc)
        self.total_ene = None

    @lru_cache(maxsize=None)
    def get_ovlp_ao(self):
        return c_get_ovlp(self.mol)

    @lru_cache(maxsize=None)
    def get_kin_ao(self):
        return c_get_kin(self.mol)

    @lru_cache(maxsize=None)
    def get_v1e_ao(self):
        return c_get_v1e(self.mol)

    @lru_cache(maxsize=None)
    def get_v2e_ao(self):
        return c_get_v2e(self.mol)

    def init_guess(self):  # 簡易的に
        #return unitary_group.rvs(self.mo_num)
        return np.eye(self.mo_num)

    def get_dense_ao(self):
        D = np.zeros((self.mo_num, self.mo_num))
        for p in range(self.mo_num):
            for q in range(self.mo_num):
                D[p][q] = 2 * np.dot(self.coeff[p][:self.occ_num], self.coeff[q][:self.occ_num])
        return D

    def make_occ(self):
        occ = np.zeros(self.mo_num, dtype=int)
        for i in range(self.mol.elec_num//2):
            occ[i] = 2
        occ_num = i+1
        if self.mol.elec_num % 2 == 1:
            occ[i+1] = 1
            self.can_rhf = False
            occ_num += 1
        return occ, occ_num

    def get_fock_ao(self):
        f = self.hcore.copy()
        for mu in range(self.mo_num):
            for nu in range(self.mo_num):
                for p in range(self.mo_num):
                    for q in range(self.mo_num):
                        f[mu][nu] += self.dense_ao[p][q] * (self.v2e_ao[mu][nu][p][q] - 1/2*self.v2e_ao[mu][q][p][nu])
        return f

    def get_nuc_ene(self, nuc):
        potential = 0
        for i in range(len(nuc)):
            for j in range(i+1, len(nuc)):
                distance = np.sqrt(sum([(nuc[i][x] - nuc[j][x])**2 for x in range(1, 4)]))
                if distance <= 1e-10:
                    assert "atoms is arranged in very close"
                potential += nuc[i][0] * nuc[j][0] / distance
        return potential

    def converged(self, f, threshold):
        error_vector = np.hstack([f[i][self.occ_num:] for i in range(self.occ_num)])
        error = np.linalg.norm(error_vector)
        print("error: ",error)
        return error < threshold

    def run(self):
        if not self.can_rhf:
            assert "this molecule has odd electrons {}".format(self.mol.elec_num)
            return 0
        for i in range(self.max_iteration):
            print(i, "iteration")
            self.fock_ao = self.get_fock_ao()
            self.fock_mo = self.coeff.T@self.fock_ao@self.coeff
            self.mo_ene, self.coeff = eigh(self.fock_ao, self.ovlp_ao)
            self.dense_ao = self.get_dense_ao()
            if self.converged(self.fock_mo, threshold=1e-5):
                break
            if i == self.max_iteration-1:
                print("Not converged")
        self.elec_ene = 1/2*np.trace(self.dense_ao @ (self.hcore + self.fock_ao))
        self.total_ene = self.elec_ene + self.nuc_ene
        return self.total_ene


def eig(h, s):
    e, c = eigh(h, s)
    ind = np.argmax(abs(c.real), axis=0)
    for i in range(len(ind)):
        if c[ind[i]][i].real < 0:
            c[:][i] *= -1
    return e, c



from mscf.mole.mole import Mole
from time import time
from pyscf import gto, scf

mol =  Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], "sto3g")

hf = HF(mol)
hf.run()
print(hf.total_ene)
m = gto.Mole()
X = 0.52918
m.build(atom="H 0 0 %f; Li 0 0 %f" %(-0.7*X, 0.7*X),basis="sto3g")
mf = scf.RHF(m)
print(mf.scf())



