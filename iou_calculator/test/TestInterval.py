from ..bounding_boxes_utils.interval import Interval 
import unittest

class TestInterval(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(ValueError):
            i = Interval(5, 3)
        
    def test_add(self):
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        result = i1 + i2
        self.assertEqual(result.l, 4)
        self.assertEqual(result.u, 6)

    def test_sub(self):
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        result = i1 - i2
        self.assertEqual(result.l, -3)
        self.assertEqual(result.u, -1)

    def test_mul(self):
        i1 = Interval(1, 2)
        i2 = Interval(-4, 5)
        result = i1 * i2
        self.assertEqual(result.l, -8)
        self.assertEqual(result.u, 10)
        
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        result = i1 * i2
        self.assertEqual(result.l, 3)
        self.assertEqual(result.u, 8)
        

    def test_reciprocal_positive(self):
        i = Interval(2, 3)
        result = i.reciprocal_positive()
        self.assertEqual(result.l, 1/3)
        self.assertEqual(result.u, 0.5)

        # Check that negative value is not working
        with self.assertRaises(ValueError):
            i = Interval(-2, 3)
            result = i.reciprocal_positive()

        # Check that 0 value is not working also
        with self.assertRaises(ValueError):
            i = Interval(0, 3)
            result = i.reciprocal_positive()


