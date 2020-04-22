import numpy as np


class DIIS:
    def __init__(self, mol):
        self.ind = 0
        self.store_max = 5
        self.store_num = 0
        self.mo_num = mol.occ
        self.occ_num = mol.occ_num
        self.store_fock = []
        self.store_error = []

    def return_error(self, fock_ao, coeff):
        fock_mo = coeff.T@fock_ao@coeff
        error = np.hstack([fock_mo[i][self.occ_num:] for i in range(self.occ_num)])
        return error

    def insert(self, fock, coeff):
        if self.store_num < self.store_max:
            self.store_fock.append(fock.copy())
            self.store_error.append(self.return_error(fock, coeff))
            self.store_num += 1
        else:
            self.store_fock[self.ind] = fock.copy()
            self.store_error[self.ind] = self.return_error(fock, coeff)
        self.ind = (self.ind+1) % self.store_max

    def return_fock(self):
        B = np.ones((self.store_num+1, self.store_num+1))*(-1)
        B[self.store_num][self.store_num] = 0
        for i in range(self.store_num):
            for j in range(self.store_num):
                B[i][j] = np.dot(self.store_error[i], self.store_error[j])
        b = np.zeros(self.store_num+1)
        b[self.store_num] = -1
        w = b@np.linalg.pinv(B)
        new_fock = sum([self.store_fock[i]*w[i] for i in range(self.store_num)])
        return new_fock





