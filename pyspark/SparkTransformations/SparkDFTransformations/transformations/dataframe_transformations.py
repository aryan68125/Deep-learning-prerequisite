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
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
# import logging related stuff
from lib.logger import Log4j
from lib.app_monitor import GetDataFrameMemory

class DataFrameTransformations:
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)


    """This methods onverts the column with dates in string datatype to date datatype"""
    def convert_str_to_timestamp_type(self, spark_df, col_name,spark_df_name,time_stamp:bool=False):
        try:
            self.logger.debug(f"converting date columns from string datatype to timestamp datatype in dataFrame {spark_df_name}")
            # Apply Spark’s to_timestamp() with the exact format
            if time_stamp:
                parsed_col = F.to_timestamp(F.col(col_name), "dd/MMM/yyyy:HH:mm:ss Z")
            else:
                parsed_col = F.to_date(F.col(col_name), "d/M/yyyy")

            result_df = spark_df.withColumn(col_name, parsed_col)
            self.log_df_metrics(result_df,operation_name="convert_str_to_timestamp_type")
            return result_df
        except Exception as e:
            self.logger.error(str(e))
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
            self.logger.error(str(e))
            raise

    """This method will select the columns based on column_name"""
    def select_col(self,spark_df, saprk_df_name:str = "",col_list:list = [], expr_str:str="",convert_to_bool_col_name:str=""):
        try:
            if len(col_list) and not expr_str and not convert_to_bool_col_name:
                return spark_df.select(*col_list)
            elif len(col_list) and expr_str and not convert_to_bool_col_name:
                return spark_df.select(*col_list, F.expr(expr_str))
            elif len(col_list) and not expr_str and convert_to_bool_col_name:
                bool_udf = udf(self.bool_parser, BooleanType())
                result_df = spark_df.withColumn(
                    convert_to_bool_col_name,
                    bool_udf(F.col(convert_to_bool_col_name))
                )
                return result_df.select(*col_list, convert_to_bool_col_name)
            else:
                self.logger.error(f"column list cannot be empty! you need column names to be able to select columns from {saprk_df_name} dataFrame")
                raise ValueError(f"column list cannot be empty! you need column names to be able to select columns from {saprk_df_name} dataFrame")
        except Exception as e:
            self.logger.error(str(e))
            raise

    # utility methods
    @staticmethod
    def bool_parser(bool_val):
        if bool_val in (1, "1", True):
            return True
        elif bool_val in (0, "0", False):
            return False
        else:
            raise ValueError("Numbers 1 and 0 are the only one that can be converted to boolean type boolean conversion for numbers other than this is not possible!")

    def log_df_metrics(self,spark_df,operation_name):
        self.logger.info(f"{operation_name} :: The memory taken by the spark dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"Spark DataFrame Schema (expanded): {schema_str}")