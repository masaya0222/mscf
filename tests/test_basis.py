import unittest
from mscf.basis import tools


class BasisTestCase(unittest.TestCase):
    def test_get_basis(self):
        basis = tools.get_basis("sto3g")
        self.assertIsInstance(basis, type(dict()))
        self.assertEqual(basis["H"][0], ['S', [3.425250914, 0.6239137298, 0.168855404],
                                         [0.1543289673, 0.5353281423, 0.4446345422]])
        self.assertEqual(basis["Li"][0], ['S', [16.11957475, 2.936200663, 0.794650487],
                                          [0.1543289673, 0.5353281423, 0.4446345422]])
        self.assertEqual(basis["Li"][1], ['SP', [0.6362897469, 0.1478600533, 0.0480886784], [-0.09996722919, 0.3995128261, 0.7001154689],
                                          [0.155916275, 0.6076837186, 0.3919573931]])


if __name__ == '__main__':
    unittest.main()
