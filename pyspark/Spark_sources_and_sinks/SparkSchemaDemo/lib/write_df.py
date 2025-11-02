from lib.logger import Log4j
class ExportSparkDataFrame:
    def __init__(self,spark_df,spark):
        self.spark_df = spark_df
        self.logger = Log4j(spark)
    def export_df_parquet(self,save_mode: str = "overwrite",output_path : str=""):
        try:
            (
                self.spark_df.write
                .format("parquet")
                .mode(save_mode)
                .option("path",output_path)
                .save()
            )
            return True
        except Exception as e:
            self.logger.error(str(e))
            raise
    
    def export_df_avro(self,save_mode: str = "overwrite", output_path: str = ""):
        try:
            (
                self.spark_df.write
                .format("avro")
                .mode(save_mode)
                .option("path",output_path)
                .save()
            )
            return True
        except Exception as e:
            self.logger.error(str(e))
            raise

    def export_df_json(self,save_mode : str = "overwrite", output_path : str = "",column_list : list = []):
        try:
            (
                self.spark_df.write
                .format("json")
                .mode(save_mode)
                .option("path",output_path)
                .partitionBy(column_list[0],column_list[1])
                .save()
            )
        except Exception as e:
            self.logger.error(str(e))
            raise