from .logger import Log4j
from .app_monitor import GetDataFrameMemory
class IngestData:
    def __init__(self,spark):
        self.spark_object = spark
        self.logger = Log4j(spark)
        self.metrics = GetDataFrameMemory(spark)
    def import_data_csv(self,file_dir):
        try:
            spark_df = (
                self.spark_object
                .read
                .format("csv")
                .option("header","true")
                .option("inferschema","true")
                .load(file_dir)
                        )
            # initialize the metrics class
            
            self.logger.info(f"spark_df created successfully from {file_dir} dataset file")
            self.logger.info(f"The memory taken by the dataFrame is = {self.metrics.get_mem_usage(spark_df).get("mem")} MB")
            self.logger.info(f"DataFrame sample:\n{spark_df.limit(25).toPandas().to_string(index=False)}")
            return spark_df
        except Exception as e:
            self.logger.error(e)