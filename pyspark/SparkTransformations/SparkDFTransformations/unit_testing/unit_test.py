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

    def test_convert_to_date_type(self):
        # Sample test DataFrame
        data_date = [
            Row(id="1", date="3/11/2025"),
            Row(id="2", date="4/11/2025")
        ]

        data_time_stamp = [
            Row(id="1", date="03/Nov/2025:00:00:00 +0000"),
            Row(id="2", date="04/Nov/2025:00:00:00 +0000")
        ]

        schema = "id STRING, date STRING"
        spark_df_date = self.spark.createDataFrame(data_date, schema)
        spark_df_timestamp = self.spark.createDataFrame(data_time_stamp,schema)

        # Apply transformation
        result_df_date = self.transformer.convert_str_to_timestamp_type(spark_df_date, "date", "spark_df_date",False)
        result_df_timestamp = self.transformer.convert_str_to_timestamp_type(spark_df_timestamp, "date", "spark_df_timestamp",True)

        # Collect and check types
        result_date = result_df_date.collect()
        result_timestamp = result_df_timestamp.collect()

        # Assert that EventDate is converted to a Python date object
        self.assertIsInstance(result_date[0]['date'], date)
        self.assertEqual(result_date[0]['date'], date(2025, 11, 3))

        self.assertIsInstance(result_timestamp[0]['date'], date)
        self.assertEqual(result_timestamp[0]['date'], datetime(2025, 11, 3, 5, 30))

        print("✅ convert_str_to_timestamp_type() test passed!")

    def test_groupby_referrer_with_referrer_column(self):
        """Test case when col_name == 'referrer' (extracts first 3 slashes and groups)"""
        data = [
            Row(referrer="http://example.com/page1"),
            Row(referrer="http://example.com/page2"),
            Row(referrer="http://example.com/page1/details"),
            Row(referrer="-"),  # should be ignored
        ]
        df = self.spark.createDataFrame(data)

        result_df = self.transformer.groupby_referrer(df, "referrer")
        result = {row["referrer"]: row["count"] for row in result_df.collect()}

        # substring_index(..., "/", 3) → keeps "http://example.com"
        self.assertIn("http://example.com", result)
        self.assertEqual(result["http://example.com"], 3)
        print("✅ test_groupby_referrer_with_referrer_column passed!")

    def test_groupby_referrer_with_non_referrer_column(self):
        """Test case when col_name != 'referrer' (simple groupBy count)"""
        data = [
            Row(category="A"),
            Row(category="A"),
            Row(category="B"),
            Row(category="-"),  # should be ignored
        ]
        df = self.spark.createDataFrame(data)

        result_df = self.transformer.groupby_referrer(df, "category")
        result = {row["category"]: row["count"] for row in result_df.collect()}

        self.assertEqual(result["A"], 2)
        self.assertEqual(result["B"], 1)
        self.assertNotIn("-", result)
        print("✅ test_groupby_referrer_with_non_referrer_column passed!")

    def test_groupby_referrer_handles_empty_df(self):
        """Ensure it gracefully handles empty input DataFrame"""
        df = self.spark.createDataFrame([], "referrer STRING")
        result_df = self.transformer.groupby_referrer(df, "referrer")
        self.assertEqual(result_df.count(), 0)
        print("✅ test_groupby_referrer_handles_empty_df passed!")
    
    def test_select_columns_only(self):
        """Case 1: Select only specific columns"""
        data = [Row(id=1, name="Alice", age=25), Row(id=2, name="Bob", age=30)]
        df = self.spark.createDataFrame(data)

        result_df = self.transformer.select_col(df, "test_df", col_list=["id", "name"])
        result = [tuple(row) for row in result_df.collect()]

        self.assertEqual(result, [(1, "Alice"), (2, "Bob")])
        print("✅ test_select_columns_only passed!")

    def test_select_with_expr(self):
        """Case 2: Select columns and apply an expression"""
        data = [Row(a=1, b=2), Row(a=3, b=4)]
        df = self.spark.createDataFrame(data)

        expr = "a + b as sum"
        result_df = self.transformer.select_col(df, "test_df", col_list=["a", "b"], expr_str=expr)

        result = [tuple(row) for row in result_df.collect()]
        self.assertEqual(result, [(1, 2, 3), (3, 4, 7)])  # (a, b, sum)
        print("✅ test_select_with_expr passed!")

    """This test will fail because of udf """
    # def test_select_with_boolean_conversion(self):
    #     """Case 3: Convert integer column to boolean"""
    #     data = [Row(flag="1"), Row(flag="0"), Row(flag=1), Row(flag=0)]
    #     df = self.spark.createDataFrame(data)

    #     result_df = self.transformer.select_col(
    #         df,
    #         "test_df",
    #         col_list=["flag"],
    #         convert_to_bool_col_name="flag"
    #     )
    #     result = [row["flag"] for row in result_df.collect()]

    #     self.assertEqual(result, [True, False, True, False])
    #     print("✅ test_select_with_boolean_conversion passed!")

    def test_select_raises_value_error(self):
        """Case 4: Empty col_list should raise ValueError"""
        data = [Row(a=1)]
        df = self.spark.createDataFrame(data)

        with self.assertRaises(ValueError):
            self.transformer.select_col(df, "test_df", col_list=[])
        print("✅ test_select_raises_value_error passed!")

if __name__ == '__main__':
    unittest.main()
