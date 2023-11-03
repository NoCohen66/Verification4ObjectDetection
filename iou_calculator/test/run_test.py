import unittest
from TestInterval import TestInterval


loader = unittest.TestLoader()
suite = loader.loadTestsFromModule(TestInterval())
runner = unittest.TextTestRunner()
runner.run(suite)