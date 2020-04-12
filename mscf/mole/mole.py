from mscf.basis.tools import get_basis


class Mole:
    def __init__(self, atoms, basis):
        self.atoms = atoms
        self.basis_name = basis
        self.basis = []
        basis = get_basis(basis)
        for l in atoms:
            self.basis.extend(format_basis(l, basis[l[0]]))
        self.basis_num = count_basis(self.basis)


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
    l = sum(l, [])
    return l


def count_basis(basis):
    num = 0
    for b in basis:
        num += b[1] * 2 + 1
    return num



