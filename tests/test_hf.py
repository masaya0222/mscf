import unittest
import pyscf
import mscf
from mscf.hf import hf
from math import isclose


class HfTestCase(unittest.TestCase):
    def test_hf1(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        m = pyscf.gto.Mole()
        m.build(atom="H 0 0 %f; H 0 0 %f" % (-0.7 * X, 0.7 * X), basis="sto3g")
        mf = pyscf.scf.RHF(m)
        E1 = mf.scf()

        M = mscf.mole.mole.Mole([['H', 0, 0, -0.7], ['H', 0, 0, 0.7]], "sto3g")
        MF = hf.HF(M)
        E2 = MF.run()
        rel_tol = 1e-6
        self.assertTrue(isclose(E1, E2, rel_tol=rel_tol))

    def test_hf2(self):
        X = 0.52918  # 単位変換: angstrom -> a0
        m = pyscf.gto.Mole()
        m.build(atom="Li 0 0 %f; Li 0 0 %f" % (-0.7 * X, 0.7 * X), basis="sto3g")
        mf = pyscf.scf.RHF(m)
        E1 = mf.scf()

        M = mscf.mole.mole.Mole([['Li', 0, 0, -0.7], ['Li', 0, 0, 0.7]], "sto3g")
        MF = hf.HF(M)
        E2 = MF.run()
        rel_tol = 1e-5
        self.assertTrue(isclose(E1, E2, rel_tol=rel_tol))


if __name__ == '__main__':
    unittest.main()
