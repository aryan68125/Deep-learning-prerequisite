import os
import sys
from unittest import TestCase
from pyspark.sql import SparkSession

# --- FIX: add project root to sys.path manually ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# --- FIX: use absolute imports (no leading dot) ---
from lib.utils import get_spark_app_config
from lib.ingest_data import IngestData
from transformations.dataframe_transformations import DataFrameTransformations
DataFrameTransformations

class UnitTestCase(TestCase):

    # This setup class method executes before all my unit test cases
    @classmethod
    def setUpClass(cls) -> None:
        cls.conf = get_spark_app_config()
        cls.spark = (
            SparkSession
            .builder
            .config(conf=cls.conf)
            .getOrCreate()
            )

    # Test cases
    def load_data(self):
        file_name = self.conf.get("file_name")
        project_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(project_dir,"dataset")
        file_dir = os.path.join(dataset_dir,file_name)
        ingest_data = IngestData(self.spark)
        spark_df = ingest_data.import_data_csv(file_dir)
        return spark_df
    
    def test_datafile_loading(self):
        spark_df = self.load_data()
        result_count = spark_df.count()
        self.assertEqual(result_count,175296,"Record count should be 175296")

    # def test_country_count(self):
    #     spark_df = self.load_data()
    #     dataframe_transformation = DataFrameTransformations(self.spark)
    #     count_list = dataframe_transformation.count_by_country(spark_df=spark_df).collect()
    #     # convert this df into a python dict
    #     count_dict = dict()
    #     for row in count_list:
    #         count_dict[row["Country"]] = row["count"]
        

    # This class method is responsible for cleaning up stuff
    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()
    
