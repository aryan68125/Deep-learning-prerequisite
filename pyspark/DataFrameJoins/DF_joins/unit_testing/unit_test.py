import unittest
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # one level higher
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from pyspark.sql import SparkSession, Row
from SparkDFTransformations.transformations.dataframe_transformations import DataFrameTransformations
from datetime import date , datetime


class TestDataFrameTransformations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession
            .builder
            .appName("PySparkUnitTest")
            .master("local[2]")
            .getOrCreate()
        )
        cls.transformer = DataFrameTransformations(cls.spark)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    # unit test methods here 

if __name__ == '__main__':
    unittest.main()
