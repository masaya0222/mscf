from pyscf import gto, scf, ao2mo
import numpy
mol = gto.Mole()
xx = 0.52918
w1 = -0.7 * xx
w2 = 0.7 * xx
mol.build(
    atom='H 0 0 %f; H 0 0 %f' %(w1, w2),
    basis='sto3g')
"""
mol.intor('int1e')
mf = scf.hf.SCF(mol)
conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=numpy.eye(mol.nao_nr()))
print(mf.scf())
print(e)
print(scf.hf.energy_elec(mf))

"""
print(mol.atom)
print(mol.intor('int1e_kin'))
print(mol.intor('int1e_nuc'))
print(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
print(mol.intor('int1e_ovlp'))