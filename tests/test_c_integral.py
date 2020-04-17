import unittest

from mscf.integral import int1e_ovlp_c, int1e_kin_c
import numpy as np
from math import isclose
import pyscf.gto
import mscf.mole.mole


class MyTestCase(unittest.TestCase):
    def test_Sab(self):  # check H2 sto3gのoverlap
        basis_a = [[0, 0, -0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        basis_b = [[0, 0, 0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        S_aa = int1e_ovlp_c.c_cont_Sij(basis_a, basis_a)
        S_ab = int1e_ovlp_c.c_cont_Sij(basis_a, basis_b)
        S_ba = int1e_ovlp_c.c_cont_Sij(basis_b, basis_a)
        S_bb = int1e_ovlp_c.c_cont_Sij(basis_b, basis_b)
        rel_tol = 1e-6  # 厳しいかも
        self.assertTrue(isclose(S_aa[0][0][0][0], 1.0, rel_tol=rel_tol))
        self.assertTrue(isclose(S_ab[0][0][0][0], 0.6593182, rel_tol=rel_tol))
        self.assertTrue(isclose(S_ba[0][0][0][0], 0.6593182, rel_tol=rel_tol))
        self.assertTrue(isclose(S_bb[0][0][0][0], 1.0, rel_tol=rel_tol))

        S_aa = np.array(S_aa)
        self.assertEqual(S_aa.shape, (1, 1, 1, 1))

    def test_Slm1(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; H 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        S1 = mol.intor('int1e_ovlp')

        basis_a = [[0, 0, -0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        basis_b = [[0, 0, 0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        S_aa = int1e_ovlp_c.c_S_lm(basis_a, basis_a)
        S_ab = int1e_ovlp_c.c_S_lm(basis_a, basis_b)
        S_ba = int1e_ovlp_c.c_S_lm(basis_b, basis_a)
        S_bb = int1e_ovlp_c.c_S_lm(basis_b, basis_b)
        rel_tol = 1e-5
        self.assertTrue(isclose(S_aa[0][0], S1[0][0], rel_tol=rel_tol))
        self.assertTrue(isclose(S_ab[0][0], S1[0][1], rel_tol=rel_tol))
        self.assertTrue(isclose(S_ba[0][0], S1[1][0], rel_tol=rel_tol))
        self.assertTrue(isclose(S_bb[0][0], S1[1][1], rel_tol=rel_tol))

    def test_Slm2(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; Li 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        S1 = mol.intor('int1e_ovlp')

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')
        basis = M.basis
        abs_tol = 1e-5
        for i in range(len(M.basis)):
            for j in range(len(M.basis)):
                basis_a = basis[i]
                basis_b = basis[j]
                Slm = int1e_ovlp_c.c_S_lm(basis_a, basis_b)  # y,z,x
                if i < 3 and j < 3:  # 両方ともs型のとき
                    self.assertTrue(isclose(Slm[0][0], S1[i][j], abs_tol=abs_tol))
                if i == 3 and j < 3:
                    for k in range(3):
                        self.assertTrue(isclose(Slm[k][0], S1[(k+1) % 3+3][j], abs_tol=abs_tol))

                if i < 3 and j == 3:
                    for k in range(3):
                        self.assertTrue(isclose(Slm[0][k], S1[i][(k+1) % 3+3], abs_tol=abs_tol))

                if i == j == 3:
                    for k in range(3):
                        for l in range(3):
                            self.assertTrue(isclose(Slm[k][l], S1[(k+1) % 3+3][(l+1) % 3+3], abs_tol=abs_tol))

    def test_get_ovlp1(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; Li 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        S1 = mol.intor('int1e_ovlp')

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')
        S = int1e_ovlp_c.c_get_ovlp(M)
        rel_tol = 1e-4
        abs_tol = 1e-15
        for i in range(len(S)):
            for j in range(len(S[0])):
                if abs(S[i][j]) <= 1e-12 or abs(S1[i][j]) == 1e-12:
                    self.assertTrue(isclose(S[i][j], S1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(S[i][j], S1[i][j], rel_tol=rel_tol))

    def test_get_ovlp2(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(
            atom='K 0 0 %f; H 0 0 %f' % (0 * X, 1 * X),
            basis='sto3g',
        )
        S1 = mol.intor('int1e_ovlp')
        M = mscf.mole.mole.Mole([['K', 0, 0, 0], ['H', 0, 0, 1]], 'sto3g', )
        S = int1e_ovlp_c.c_get_ovlp(M)
        rel_tol = 1e-4
        abs_tol = 1e-15
        for i in range(len(S)):
            for j in range(len(S[0])):
                if abs(S[i][j]) <= 1e-10 or abs(S1[i][j]) <= 1e-10:
                    self.assertTrue(isclose(S[i][j], S1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(S[i][j], S1[i][j], rel_tol=rel_tol))

    def test_get_ovlp3(self):  # for d軌道
        X = 0.52918  # 単位変換: angstrom -> a0
        x1, y1, z1 = 0.4, -0.1, 1
        x2, y2, z2 = -0.3, 1.2, 1.1
        x3, y3, z3 = -0.5, 0.6, -0.1
        mol = pyscf.gto.Mole()
        mol.build(
            atom='Sc %f %f %f; H %f %f %f; H %f %f %f' % (
            x1 * X, y1 * X, z1 * X, x2 * X, y2 * X, z2 * X, x3 * X, y3 * X, z3 * X),
            basis='sto3g',
            charge=+1
        )
        S1 = mol.intor('int1e_ovlp')
        M = mscf.mole.mole.Mole([['Sc', x1, y1, z1], ['H', x2, y2, z2], ['H', x3, y3, z3]], 'sto3g', )
        S = int1e_ovlp_c.c_get_ovlp(M)
        self.assertTrue(S1.shape, S.shape)
        rel_tol = 1e-4
        abs_tol = 1e-15
        for i in range(len(S)):
            for j in range(len(S[0])):
                if abs(S[i][j]) <=1e-12 or abs(S1[i][j]) <= 1e-10:
                    self.assertTrue(isclose(S[i][j], S1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(S[i][j], S1[i][j], rel_tol=rel_tol))

    def test_get_ovlp4(self):  # for d軌道
        X = 0.52918  # 単位変換: angstrom -> a0
        x1, y1, z1 = 0.4, -0.1, 1
        x2, y2, z2 = -0.3, 1.2, 1.1
        mol = pyscf.gto.Mole()
        mol.build(
            atom='I %f %f %f; H %f %f %f;' % (
                x1 * X, y1 * X, z1 * X, x2 * X, y2 * X, z2 * X),
            basis='sto3g',
            charge=0
        )
        S1 = mol.intor('int1e_ovlp')
        M = mscf.mole.mole.Mole([['I', x1, y1, z1], ['H', x2, y2, z2], ], 'sto3g', )
        S = int1e_ovlp_c.c_get_ovlp(M)
        self.assertTrue(S1.shape, S.shape)
        rel_tol = 1e-5
        abs_tol = 1e-15
        for i in range(len(S)):
            for j in range(len(S[0])):
                if abs(S[i][j]) <= 1e-10 or abs(S1[i][j]) <= 1e-10:
                    self.assertTrue(isclose(S[i][j], S1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(S[i][j], S1[i][j], rel_tol=rel_tol))

    def test_get_kin1(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; Li 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        T1 = mol.intor('int1e_kin')

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')
        T = int1e_kin_c.c_get_kin(M)
        rel_tol = 1e-4
        abs_tol = 1e-15
        for i in range(len(T)):
            for j in range(len(T[0])):
                if abs(T[i][j]) <=1e-12 or abs(T1[i][j]) <= 1e-12:
                    self.assertTrue(isclose(T[i][j], T1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(T[i][j], T1[i][j], rel_tol=rel_tol))

    def test_get_kin2(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(
            atom='K 0 0 %f; H 0 0 %f' % (0 * X, 1 * X),
            basis='sto3g',
        )
        T1 = mol.intor('int1e_kin')
        M = mscf.mole.mole.Mole([['K', 0, 0, 0], ['H', 0, 0, 1]], 'sto3g', )
        T = int1e_kin_c.c_get_kin(M)
        rel_tol = 1e-4
        abs_tol = 1e-15
        for i in range(len(T)):
            for j in range(len(T[0])):
                if abs(T[i][j]) <= 1e-12 or abs(T1[i][j]) <=1e-12:
                    self.assertTrue(isclose(T[i][j], T1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(T[i][j], T1[i][j], rel_tol=rel_tol))

    def test_get_kin3(self):  # for d軌道
        X = 0.52918  # 単位変換: angstrom -> a0
        x1, y1, z1 = 0.4, -0.1, 1
        x2, y2, z2 = -0.3, 1.2, 1.1
        x3, y3, z3 = -0.5, 0.6, -0.1
        mol = pyscf.gto.Mole()
        mol.build(
            atom='Sc %f %f %f; H %f %f %f; H %f %f %f' % (
            x1 * X, y1 * X, z1 * X, x2 * X, y2 * X, z2 * X, x3 * X, y3 * X, z3 * X),
            basis='sto3g',
            charge=+1
        )
        T1 = mol.intor('int1e_kin')
        M = mscf.mole.mole.Mole([['Sc', x1, y1, z1], ['H', x2, y2, z2], ['H', x3, y3, z3]], 'sto3g', )
        T = int1e_kin_c.c_get_kin(M)
        self.assertTrue(T1.shape, T.shape)
        rel_tol = 1e-4
        abs_tol = 1e-10
        for i in range(len(T)):
            for j in range(len(T[0])):
                if abs(T[i][j]) <= 1e-12 or abs(T1[i][j]) == 1e-12:
                    self.assertTrue(isclose(T[i][j], T1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(T[i][j], T1[i][j], rel_tol=rel_tol))

    def test_get_kin4(self):  # for d軌道
        X = 0.52918  # 単位変換: angstrom -> a0
        x1, y1, z1 = 0.4, -0.1, 1
        x2, y2, z2 = -0.3, 1.2, 1.1
        mol = pyscf.gto.Mole()
        mol.build(
            atom='I %f %f %f; H %f %f %f;' % (
                x1 * X, y1 * X, z1 * X, x2 * X, y2 * X, z2 * X),
            basis='sto3g',
            charge=0
        )
        T1 = mol.intor('int1e_kin')
        M = mscf.mole.mole.Mole([['I', x1, y1, z1], ['H', x2, y2, z2], ], 'sto3g', )
        T = int1e_kin_c.c_get_kin(M)
        self.assertTrue(T1.shape, T.shape)
        rel_tol = 1e-4
        abs_tol = 1e-10
        for i in range(len(T)):
            for j in range(len(T[0])):
                if abs(T[i][j]) <= 1e-12 or abs(T1[i][j]) == 1e-12:
                    self.assertTrue(isclose(T[i][j], T1[i][j], abs_tol=abs_tol))
                else:
                    self.assertTrue(isclose(T[i][j], T1[i][j], rel_tol=rel_tol))


if __name__ == '__main__':
    unittest.main()
