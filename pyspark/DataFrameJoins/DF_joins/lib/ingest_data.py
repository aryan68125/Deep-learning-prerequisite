# --- Add project root to sys.path ---
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql.functions import regexp_extract
from pyspark.sql import functions as F

from .logger import Log4j
from .app_monitor import GetDataFrameMemory



"""
This class ingest data from csv, json and parquet file format
"""
class IngestData():
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)

    def import_data_json(self,file_dir,df_schema):
        try:
            self.logger.debug("\n Reading data from the json file : Ingest data into dataFrame")
            spark_df = (
                    self.spark_object
                    .read
                    .format("json")
                    .schema(df_schema)
                    .load(file_dir)
            )
 
            # Convert FL_DATE only if present
            if "FL_DATE" in spark_df.columns:
                spark_df = spark_df.withColumn(
                    "FL_DATE",
                    F.to_timestamp("FL_DATE", "M/d/yyyy")
                )

            # Convert id safely
            if "id" in spark_df.columns:
                spark_df = spark_df.withColumn(
                    "id",
                    F.col("id").cast("bigint")
                )

            self.log_df_metrics(spark_df=spark_df,file_dir=file_dir,operation_name="import_data_csv")
            return spark_df
        except Exception as e:
            self.logger.error(str(e))

    # utility methods
    def log_df_metrics(self,spark_df,file_dir,operation_name):
        self.logger.info(f"{operation_name} :: spark_df created successfully from {file_dir} dataset file")
        self.logger.info(f"{operation_name} :: The memory taken by the spark dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"Spark DataFrame Schema (expanded): {schema_str}")
        # log the dataFrame partition
        df_partition = spark_df.rdd.getNumPartitions()
        if df_partition > 1:
            self.logger.debug(f"dataFrame has {df_partition} partitons")
        elif df_partition == 1:
            self.logger.debug(f"dataFrame has {df_partition} partiton")
        else:
            self.logger.error(f"dataFrame partition not found!")