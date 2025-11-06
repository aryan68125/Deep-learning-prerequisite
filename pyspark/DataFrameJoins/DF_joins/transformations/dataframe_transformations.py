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

# import window for implementing window function
from pyspark.sql.window import Window

# import typing 
from typing import List

class DataFrameTransformations:
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)
        self.app_metrics = GetDataFrameMemory(spark)

    # dataFrame transformation methods here
    def create_unique_identifier(self,spark_df):
        try:
            spark_df = spark_df.withColumn("id",F.monotonically_increasing_id())
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise
    
    def process_date_col_year(self,spark_df,col_name : str=None, combine_date : bool = False):
        try:
            if not col_name or col_name == "":
                self.logger.error("col_name cannot be empty or None!")
                raise ValueError("col_name cannot be empty or None!")
            spark_df = spark_df.withColumn(col_name,F.expr("""
            case when year < 25 then cast(year as int) + 2000 
            when year < 100 then cast(year as int) + 1900
            else year
            end
            """))

            if combine_date:
                spark_df = spark_df.withColumn("dob",F.expr("""
                    to_date(concat(day,'/',month,'/',year),'d/M/y')
                """)).drop('day','month','year')
            self.app_metrics.get_mem_usage(spark_df=spark_df)
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise

    def drop_duplicate_rows(self,spark_df,col_name_list:List[str]):
        try:
            spark_df = spark_df.dropDuplicates(col_name_list)
            self.app_metrics.get_mem_usage(spark_df=spark_df)
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise

    def simple_aggregation_operation(self,spark_df):
        try:
            spark_df = spark_df.selectExpr(
                """
                    count(*) `count`
                """,
                """
                    count(StockCode) as `count field`
                """,
                """sum(Quantity) as TotalQuantity""",
                """avg(UnitPrice) as AverageUnitPrice"""
            )
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise
    
    def complex_aggregation_operation(self,spark_df):
        try:
            spark_df = (
                spark_df
                .groupBy("Country", "InvoiceNo")
                .agg(
                    F.sum("Quantity").alias("TotalQuantity"),
                    F.round(F.sum(F.expr("Quantity * UnitPrice")),2).alias("InvoiceValue")
                )
        )
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise

    def group_by_country_agg(self,spark_df):
        try:
            NumInvoice = F.countDistinct("InvoiceNo").alias("NumInvoice")
            TotalQuantity = F.sum("Quantity").alias("TotalQuantity")
            InvoiceValue = F.round(
                F.sum(
                    F.expr("Quantity * UnitPrice")
                ),2
            ).alias("InvoiceValue")
            saprk_df = (
                spark_df
                .where("year(InvoiceDate) == 2010")
                .withColumn("WeekNumber",F.weekofyear(
                    F.col("InvoiceDate")
                ))
                .groupBy(
                    "Country","WeekNumber"
                )
                .agg(
                    NumInvoice, TotalQuantity, InvoiceValue
                )
            )
            return saprk_df
        except Exception as e:
            self.logger.error(str(e))
            raise

    def window_aggregation(self,spark_df):
        try:
            # Step 1: Add WeekNumber and per-row InvoiceValue
            spark_df = (
                spark_df
                .withColumn("WeekNumber", F.weekofyear(F.col("InvoiceDate")))
                .withColumn("InvoiceValue", F.round(F.col("Quantity") * F.col("UnitPrice"), 2))
            )
            # Step 2 : setting up the window function
            running_total_window = (
                    Window
                    .partitionBy("Country")
                    .orderBy("WeekNumber")
                    .rowsBetween(Window.unboundedPreceding,Window.currentRow)
                )
            # Step 3 : perform window function operation on the dataFrame
            spark_df = (
                spark_df
                .withColumn(
                    "RunningTotal",
                    (F.sum("InvoiceValue")
                    .over(running_total_window))
                )
            )
            return spark_df
        except Exception as e:
            self.logger.error(str(e))
            raise

    def log_df_metrics(self,spark_df,operation_name):
        self.logger.info(f"{operation_name} :: The memory taken by the spark dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"Spark DataFrame Schema (expanded): {schema_str}")