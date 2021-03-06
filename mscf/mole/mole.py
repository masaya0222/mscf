from mscf.basis.tools import get_basis
from mscf.mole.element_data import ELEMENTS_PROTON
import numpy as np


class Mole:
    def __init__(self, atoms, basis, charge=0):
        self.atoms = atoms
        self.charge = charge
        self.basis_name = basis
        self.basis = []
        basis = get_basis(basis)
        for l in self.atoms:
            self.basis.extend(format_basis(l, basis[l[0]]))
        self.basis_num = count_basis(self.basis)
        self.nuc = [[ELEMENTS_PROTON[atom[0]]] + atom[1:] for atom in self.atoms]
        self.elec_num = sum([ELEMENTS_PROTON[atom[0]] for atom in self.atoms]) - self.charge
        self.can_rhf = True
        self.occ, self.occ_num = self.make_occ()

    def make_occ(self):
        occ = np.zeros(self.basis_num, dtype=int)
        for i in range(self.elec_num//2):
            occ[i] = 2
        occ_num = i+1
        if self.elec_num % 2 == 1:
            occ[i+1] = 1
            occ_num += 1
        return occ, occ_num


def format_basis(atoms, basis):
    l = [[]]
    for b in basis:
        if b[0] == 'S':
            l[0].append([atoms[1:]] + [0, b[1], b[2]])
        elif b[0] == 'SP':
            if len(l) <= 1:
                l.append([])
            l[0].append([atoms[1:]] + [0, b[1], b[2]])
            l[1].append([atoms[1:]] + [1, b[1], b[3]])
        elif b[0] == "D":
            if len(l) <= 2:
                l.append([])
            l[2].append([atoms[1:]] + [2, b[1], b[2]])
    l = sum(l, [])
    return l


def count_basis(basis):
    num = 0
    for b in basis:
        num += b[1] * 2 + 1
    return num




