# --- Add project root to sys.path ---
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .logger import Log4j
from .app_monitor import GetDataFrameMemory

from spark_dataFrame_schema.spark_dataframe_schema import FlightSchemaMixin

"""
This class ingest data from csv, json and parquet file format
"""
class IngestData():
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)
        self.df_schema = FlightSchemaMixin()

    def import_data_csv(self,file_dir):
        try:
            spark_df = (
                    self.spark_object
                    .read
                    .format("csv")
                    .option("header","true")
                    # .option("inferschema","true")
                    .schema(self.df_schema.return_flight_df_schema())
                    # Set the mode for error if the schema don't match
                    .option("mode","FAILFAST")
                    # Set the date string format
                    .option("dateFormat","M/d/y")
                    .load(file_dir)
            )
            self.log_df_metrics(spark_df=spark_df,file_dir=file_dir)
            return spark_df
        except Exception as e:
            self.logger.error(str(e))

    def import_data_json(self,file_dir):
        try:
            spark_df = (
                self.spark_object
                .read
                .format("json")
                .load(file_dir)
            )
            self.log_df_metrics(spark_df=spark_df,file_dir=file_dir)
            return spark_df
        except Exception as e:
            self.logger.error(str(e))

    def import_data_parquet(self,file_dir):
        try:
            spark_df = (
                self.spark_object
                .read
                .format("parquet")
                .load(file_dir)
            )
            self.log_df_metrics(spark_df=spark_df,file_dir=file_dir)
            return spark_df
        except Exception as e:
            self.logger.error(str(e))

    # utility methods
    def log_df_metrics(self,spark_df,file_dir):
        self.logger.info(f"import_data_csv :: spark_df created successfully from {file_dir} dataset file")
        self.logger.info(f"import_data_csv :: The memory taken by the spark dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"Spark DataFrame Schema (expanded): {schema_str}")