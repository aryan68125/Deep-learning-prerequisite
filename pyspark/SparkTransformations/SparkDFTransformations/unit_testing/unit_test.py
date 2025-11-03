import unittest
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # one level higher
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from pyspark.sql import SparkSession, Row
from SparkDFTransformations.transformations.dataframe_transformations import DataFrameTransformations
from datetime import date


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

    def test_convert_to_date_type(self):
        # Sample test DataFrame
        data = [
            Row(id="1", EventDate="3/11/2025"),
            Row(id="2", EventDate="4/11/2025")
        ]

        schema = "id STRING, EventDate STRING"
        df = self.spark.createDataFrame(data, schema)

        # Apply transformation
        result_df = self.transformer.convert_to_date_type(df, "d/M/yyyy", "EventDate")

        # Collect and check types
        result = result_df.collect()

        # Assert that EventDate is converted to a Python date object
        self.assertIsInstance(result[0]['EventDate'], date)
        self.assertEqual(result[0]['EventDate'], date(2025, 11, 3))

        print("✅ convert_to_date_type() test passed!")


if __name__ == '__main__':
    unittest.main()
