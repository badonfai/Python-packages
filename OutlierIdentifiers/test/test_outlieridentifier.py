"""
Test script for outlieridentifier.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from OutlierIdentifier.Outlieridentifier.outlieridentifier import *
import numpy as np
import unittest
import json

np.random.seed(2342)
randData1 = np.random.uniform(low=-10, high=100, size=200)
randData2 = np.random.uniform(low=140, high=160, size=19)
randData3 = np.random.uniform(low=-50, high=-30, size=12)
randData = np.concatenate((randData1, randData2, randData3), axis=0)
randData = np.random.choice(randData, size=len(randData))


class Tests(unittest.TestCase):
    """Equivalence Tests for outlieridentifier.py"""

    def test_OutlierIdentifier_equal(self):
        """outlierIdentifiers equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold) | (randData >= upperThreshold)]
        test2 = OutlierIdentifier(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold) | (randData >= upperThreshold)]
        test2 = OutlierIdentifier(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold) | (randData >= upperThreshold)]
        test2 = OutlierIdentifier(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

    def test_TopOutlierIdentifier_equal(self):
        """TopOutlierIdentifier equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = randData[(randData >= upperThreshold)]
        test2 = OutlierIdentifier(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = randData[(randData >= upperThreshold)]
        test2 = TopOutlierIdentifier(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = randData[(randData >= upperThreshold)]
        test2 = TopOutlierIdentifier(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

    def test_BottomOutlierIdentifier_equal(self):
        """BottomOutlierIdentifier equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold)]
        test2 = OutlierIdentifier(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold)]
        test2 = BottomOutlierIdentifier(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = randData[(randData <= lowerThreshold)]
        test2 = BottomOutlierIdentifier(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

    def test_OutlierPosition_equal(self):
        """OutlierPosition equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold) | (randData >= upperThreshold))[0]
        test2 = OutlierPosition(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold) | (randData >= upperThreshold))[0]
        test2 = OutlierPosition(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold) | (randData >= upperThreshold))[0]
        test2 = OutlierPosition(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

    def test_TopOutlierPosition_equal(self):
        """TopOutlierPosition equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = np.where((randData >= upperThreshold))[0]
        test2 = TopOutlierPosition(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = np.where((randData >= upperThreshold))[0]
        test2 = TopOutlierPosition(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = np.where((randData >= upperThreshold))[0]
        test2 = TopOutlierPosition(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

    def test_BottomOutlierPosition_equal(self):
        """BottomOutlierPosition equivalences"""

        lowerThreshold, upperThreshold = SPLUSQuartileIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold))[0]
        test2 = BottomOutlierPosition(randData, SPLUSQuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = QuartileIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold))[0]
        test2 = BottomOutlierPosition(randData, QuartileIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())

        lowerThreshold, upperThreshold = HampelIdentifierParameters(randData)
        test1 = np.where((randData <= lowerThreshold))[0]
        test2 = BottomOutlierPosition(randData, HampelIdentifierParameters)
        self.assertEqual(test1.tolist(), test2.tolist())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(*[Tests])
    testResult = unittest.TextTestRunner(verbosity=2).run(suite)
    total = testResult.testsRun
    if total == 0:
        res = {'score': 1, 'output': []}
    else:
        errors = [x[1] for x in testResult.errors]
        failures = [x[1] for x in testResult.failures]
        score = 1 - 1.0 * (len(errors) + len(failures)) / total
        res = {'score': score, 'test_output': errors + failures}
    print(json.dumps(res))