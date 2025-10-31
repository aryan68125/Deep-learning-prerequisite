from lib.logger import Log4j

class SqlOperations:
    def __init__(self,spark):
        self.spark_obj = spark
        self.logger = Log4j(self.spark_obj)

    # This will create a sql view from a spark_df
    def create_view(self,spark_df,view_name):
        return spark_df.createOrReplaceTempView(view_name)
    def display_view(self,view_name):
        query = f"""
        SELECT * FROM {view_name} LIMIT 25;
        """
        result = self.spark_obj.sql(query)
        result_pd_df = result.limit(25).toPandas().to_string(index=False)
        self.logger.info(f"view ==> \n {result_pd_df}")
        result.show()