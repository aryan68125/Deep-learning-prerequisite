from lib.logger import Log4j
class ExportSparkDataFrame:
    def __init__(self,spark):
        self.logger = Log4j(spark)
        self.spark = spark
    def export_df_parquet(self,spark_df,save_mode: str = "overwrite",output_path : str="",max_rec : int = 100):
        try:
            (
                spark_df.write
                .format("parquet")
                .mode(save_mode)
                .option("path",output_path)
                .option("maxRecorsdPerFile",max_rec)
                .save()
            )
            return True
        except Exception as e:
            self.logger.error(str(e))
            raise
    
    def export_df_avro(self,spark_df,save_mode: str = "overwrite", output_path: str = ""):
        try:
            (
                spark_df.write
                .format("avro")
                .mode(save_mode)
                .option("path",output_path)
                .save()
            )
            return True
        except Exception as e:
            self.logger.error(str(e))
            raise

    def export_df_json(self,spark_df,save_mode : str = "overwrite", output_path : str = "",column_list : list = []):
        try:
            (
                spark_df.write
                .format("json")
                .mode(save_mode)
                .option("path",output_path)
                .partitionBy(column_list[0],column_list[1])
                .save()
            )
        except Exception as e:
            self.logger.error(str(e))
            raise

    def export_df_sql_table(self,spark_df1,spark_df2):
        try:
            self.logger.debug(">>>> Exporting dataframe to the sql table")
            # Create database if not exists
            self.spark.sql("CREATE DATABASE IF NOT EXISTS RollexDB")
            self.spark.sql("USE RollexDB")
            # save the dataFrames in a table after partitioning by bucket
            spark_df1.coalesce(1).write.bucketBy(3,"id").saveAsTable("RollexDB.flight_data1")
            spark_df2.coalesce(1).write.bucketBy(3,"id").saveAsTable("RollexDB.flight_data2")
        except Exception as e:
            self.logger.error(str(e))
            raise