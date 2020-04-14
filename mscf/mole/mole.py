from mscf.basis.tools import get_basis
from mscf.mole.element_data import ELEMENTS_PROTON


class Mole:
    def __init__(self, atoms, basis):
        self.atoms = atoms
        self.basis_name = basis
        self.basis = []
        basis = get_basis(basis)
        for l in self.atoms:
            self.basis.extend(format_basis(l, basis[l[0]]))
        self.basis_num = count_basis(self.basis)
        self.nuc = [[ELEMENTS_PROTON[atom[0]]] + atom[1:] for atom in self.atoms]


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




