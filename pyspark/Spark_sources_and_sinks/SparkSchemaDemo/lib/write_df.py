class ExportSparkDataFrame:
    def __init__(self,spark_df):
        self.spark_df = spark_df
    def export_df_parquet(self,save_mode: str = "overwrite",output_path : str=""):
        (
            self.spark_df.write
            .format("parquet")
            .mode(save_mode)
            .option("path",output_path)
            .save()
        )
        return True