from typing import List, Tuple, Any
from lib.logger import Log4j, LogSparkDataframe
from lib.app_monitor import GetDataFrameMemory
from pyspark.sql import functions as F

class GenerateDataFrame:
    def __init__(self, spark):
        self.spark = spark
        self.logger = Log4j(spark)
        self.sp_df_logger = LogSparkDataframe(spark)
        self.app_metrics = GetDataFrameMemory(spark)

    def generate_dataframe(self, data_list : List[Tuple[Any, ...]] = None):
        self.logger.debug(f"checking for the supplied data_list: ")
        if not data_list:
            self.logger.error("DataList required to generate a dataFrame!")
            raise ValueError(f"DataList required to generate a dataFrame!")
        self.logger.debug(f"supplied data_list found : {data_list}")

        # check if the spark session master is set to local if yes then implement repartition if not then don't
        if self.spark.sparkContext.master == "local[3]":
            generated_df = (
                self.spark
                .createDataFrame(data_list)
                .toDF("name","day","month","year")
                .repartition(3)
            )
        else:
            generated_df = (
                self.spark
                .createDataFrame(data_list)
                .toDF("name","day","month","year")
            )
        self.app_metrics.get_mem_usage(generated_df)
        self.sp_df_logger.log_df_metrics(spark_df=generated_df,spark_df_name="generated_df")
        return generated_df