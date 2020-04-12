import unittest

from mscf.integral import ovlp_integral
import numpy as np
from math import isclose


class IntegralTestCase(unittest.TestCase):

    def test_ovlp(self): #check H2 sto3gのoverlap
        basis_a = [[0, 0, -0.7], 0, [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]]
        basis_b = [[0, 0, 0.7], 0, [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]]
        S_aa = ovlp_integral.S_ab(basis_a, basis_a)
        S_ab = ovlp_integral.S_ab(basis_a, basis_b)
        S_ba = ovlp_integral.S_ab(basis_b, basis_a)
        S_bb = ovlp_integral.S_ab(basis_b, basis_b)
        rel_tol = 1e-6 #厳しいかもo
        self.assertTrue(isclose(S_aa[0][0][0][0], 1.0, rel_tol=rel_tol))
        self.assertTrue(isclose(S_ab[0][0][0][0], 0.6593182, rel_tol=rel_tol))
        self.assertTrue(isclose(S_ba[0][0][0][0], 0.6593182, rel_tol=rel_tol))
        self.assertTrue(isclose(S_bb[0][0][0][0], 1.0, rel_tol=rel_tol))

        S_aa = np.array(S_aa)
        self.assertEqual(S_aa.shape, (1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
