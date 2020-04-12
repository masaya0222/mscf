import unittest

from mscf.integral import ovlp_integral
import numpy as np
from math import isclose
import pyscf.gto
import mscf.mole.mole


class IntegralTestCase(unittest.TestCase):

    def test_Sab(self):  # check H2 sto3gのoverlap
        basis_a = [[0, 0, -0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        basis_b = [[0, 0, 0.7], 0, [3.425250914, 0.6239137298, 0.168855404],
                   [0.1543289673, 0.5353281423, 0.4446345422]]
        S_aa = ovlp_integral.S_ab(basis_a, basis_a)
        S_ab = ovlp_integral.S_ab(basis_a, basis_b)
        S_ba = ovlp_integral.S_ab(basis_b, basis_a)
        S_bb = ovlp_integral.S_ab(basis_b, basis_b)
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
        S_aa = ovlp_integral.S_lm(basis_a, basis_a)
        S_ab = ovlp_integral.S_lm(basis_a, basis_b)
        S_ba = ovlp_integral.S_lm(basis_b, basis_a)
        S_bb = ovlp_integral.S_lm(basis_b, basis_b)
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
                Slm = ovlp_integral.S_lm(basis_a, basis_b)  # y,z,x
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

    def test_get_ovlp(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; Li 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        S1 = mol.intor('int1e_ovlp')

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')
        S = ovlp_integral.get_ovlp(M)
        abs_tol = 1e-5
        for i in range(len(S)):
            for j in range(len(S[0])):
                self.assertTrue(isclose(S[i][j], S1[i][j], abs_tol=abs_tol))


if __name__ == '__main__':
    unittest.main()
