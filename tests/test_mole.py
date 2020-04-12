import unittest

from mscf.mole import mole


class MoleTestCase(unittest.TestCase):
    def test_mole1(self):
        M = mole.Mole([['H', 0, 0, -0.7], ['H', 0, 0, 0.7]], "sto3g")
        self.assertEqual(M.atoms, [['H', 0, 0, -0.7], ['H', 0, 0, 0.7]])
        self.assertEqual(M.basis_name, "sto3g")
        self.assertEqual(M.basis[0], [[0, 0, -0.7], 0, [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                                      [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])
        self.assertEqual(M.basis[1], [[0, 0, 0.7], 0, [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                                      [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

    def test_mole2(self):
        M = mole.Mole([['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]], "sto3g")
        self.assertEqual(M.atoms, [['H', 0, 0, -0.7], ['Li', 0, 0, 0.7]])
        self.assertEqual(M.basis_name, "sto3g")
        self.assertEqual(M.basis[0], [[0, 0, -0.7], 0, [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                                      [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])
        self.assertEqual(M.basis[1], [[0, 0, 0.7], 0, [0.1611957475E+02, 0.2936200663E+01, 0.7946504870E+00],
                                      [0.1543289673E+00,  0.5353281423E+00, 0.4446345422E+00]])
        self.assertEqual(M.basis[2], [[0, 0, 0.7], 0, [0.6362897469E+00, 0.1478600533E+00, 0.4808867840E-01],
                                      [-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00]])
        self.assertEqual(M.basis[3], [[0, 0, 0.7], 1, [0.6362897469E+00, 0.1478600533E+00, 0.4808867840E-01],
                                      [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]])


if __name__ == '__main__':
    unittest.main()
