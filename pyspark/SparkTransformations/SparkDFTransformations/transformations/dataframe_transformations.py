import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"CURRENT_DIR >> {CURRENT_DIR}")
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
print(f"PROJECT_ROOT >> {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"printing sys path of python >>>")
for p in sys.path[:5]:
    print("  ", p)

# import transformation related stuff
from pyspark.sql import functions as F
# import logging related stuff
from lib.logger import Log4j
from lib.app_monitor import GetDataFrameMemory

class DataFrameTransformations:
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)


    """This methods onverts the column with dates in string datatype to date datatype"""
    def convert_str_to_timestamp_type(self, spark_df, col_name,spark_df_name):
        try:
            self.logger.debug(f"converting date columns from string datatype to timestamp datatype in dataFrame {spark_df_name}")
            # Apply Spark’s to_timestamp() with the exact format
            parsed_col = F.to_timestamp(F.col(col_name), "dd/MMM/yyyy:HH:mm:ss Z")

            result_df = spark_df.withColumn(col_name, parsed_col)
            self.log_df_metrics(result_df,operation_name="convert_str_to_timestamp_type")
            return result_df
        except Exception as e:
            self.logger(str(e))
            raise
    
    """this method groups the data based on referrer"""
    def groupby_referrer(self,spark_df, col_name):
        try:
            if col_name == "referrer":
                result_df = (
                        spark_df
                        .where(f"trim({col_name}) != '-'")
                        .withColumn(col_name,F.substring_index(F.col(col_name),"/",3))
                        .groupBy(col_name)
                        .count()
                    )
            else:
                result_df = (
                    spark_df
                    .where(f"trim({col_name}) != '-'")
                    .groupBy(col_name)
                    .count()
                )
            self.log_df_metrics(result_df,operation_name="groupby_referrer")
            return result_df
        except Exception as e:
            self.logger(str(e))
            raise
    
    # utility methods
    def log_df_metrics(self,spark_df,operation_name):
        self.logger.info(f"{operation_name} :: The memory taken by the spark dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"Spark DataFrame Schema (expanded): {schema_str}")