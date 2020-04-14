import unittest
import pyscf
import mscf
from mscf.integral import int2e
from math import isclose


class int2eTestCase(unittest.TestCase):
    def test_get_v2e1(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='H 0 0 %f; H 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        V1 = mol.intor('int2e')

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['H', 0, 0, 0.7]], 'sto3g')
        V = int2e.get_v2e(M)
        self.assertEqual(V1.shape, V.shape)
        rel_tol = 1e-5
        abs_tol = 1e-15
        for i in range(len(V)):
            for j in range(len(V[0])):
                for k in range(len(V)):
                    for l in range(len(V)):
                        if V[i][j][k][l] == 0.0 or V1[i][j][k][l] == 0.0:
                            self.assertTrue(isclose(V[i][j][k][l], V[i][j][j][k], abs_tol=abs_tol))
                        else:
                            self.assertTrue(isclose(V[i][j][k][l], V1[i][j][k][l], rel_tol=rel_tol))

    def test_get_v2e2(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        mol = pyscf.gto.Mole()
        mol.build(atom='Li 0 0 %f; Li 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
        V1 = mol.intor('int2e')

        M = mscf.mole.mole.Mole([['Li', 0, 0, -0.7], ['Li', 0, 0, 0.7]], 'sto3g')
        V = int2e.get_v2e(M)
        self.assertEqual(V1.shape, V.shape)
        rel_tol = 2e-4
        abs_tol = 1e-12
        for i in range(len(V)):
            for j in range(len(V)):
                for k in range(len(V)):
                    for l in range(len(V)):
                        if abs(V[i][j][k][l]) <= 1e-12 or abs(V1[i][j][k][l]) <= 1e-12:
                            self.assertTrue(isclose(V[i][j][k][l], V1[i][j][k][l], abs_tol=abs_tol))
                        else:
                            self.assertTrue(isclose(V[i][j][k][l], V1[i][j][k][l], rel_tol=rel_tol))


if __name__ == '__main__':
    unittest.main()
