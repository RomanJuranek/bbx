import unittest
import bbx


bbs = [ [-5,-5,10,10], [0,0,10,10], [10,10,10,10] ]


class TestSum(unittest.TestCase):
    def test_aspec_ratio(self):
        y = bbx.set_aspect_ratio(bbs[0], 2)
        ar = float(y[0,2] / y[0,3])
        self.assertAlmostEqual(ar, 2)


if __name__ == '__main__':
    unittest.main()
