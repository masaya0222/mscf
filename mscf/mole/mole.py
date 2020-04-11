from mscf.basis.tools import get_basis


class Mole:
    def __init__(self, atoms, basis):
        self.atoms = atoms
        self.basis_name = basis
        self.basis = []
        basis = get_basis(basis)
        for l in atoms:
            self.basis.extend(format_basis(l, basis[l[0]]))


def format_basis(atoms, basis, ):
    l = []
    for b in basis:
        if b[0] == 'S':
            l.append([atoms[1:]] + [0, b[1], b[2]])
        elif b[0] == 'SP':
            l.append([atoms[1:]] + [0, b[1], b[2]])
            l.append([atoms[1:]] + [1, b[1], b[2]])
    return l
