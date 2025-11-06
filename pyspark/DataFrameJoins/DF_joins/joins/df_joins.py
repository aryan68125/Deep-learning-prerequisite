from lib.logger import Log4j,LogSparkDataframe
from lib.app_monitor import GetDataFrameMemory
class DataFrameJoins:
    def __init__(self,spark):
        self.spark = spark
        self.logger = Log4j(spark)
        self.mem = GetDataFrameMemory(spark)
        self.metrics = LogSparkDataframe(spark)
    def inner_join_df(self,left_df,right_df,join_expression):
        try:
            # perform join operation
            result_df = left_df.join(right_df,join_expression,"inner")
            # logging memory taken by the df
            self.mem.get_mem_usage(result_df)
            # logging spark schema 
            self.logger.debug("logging intermediate dataframe after inner join operation")
            self.metrics.log_df_metrics(spark_df=result_df,spark_df_name="result_df")
            return result_df
        except Exception as e:
            self.logger.error(str(e))
            raise
    # left join is also called left outer join
    def left_join_df(self,left_df,right_df,join_expression):
        try:
            # perform join operation
            result_df = left_df.join(right_df,join_expression,"left")
            # logging memory taken by the df
            self.mem.get_mem_usage(result_df)
            # logging spark schema 
            self.logger.debug("logging intermediate dataframe after left join operation")
            self.metrics.log_df_metrics(spark_df=result_df,spark_df_name="result_df")
            return result_df
        except Exception as e:
            self.logger.error(str(e))
            raise
    # right join is also called right outer join
    def right_join_df(self,left_df,right_df,join_expression):
        try:
            # perform join operation
            result_df = left_df.join(right_df,join_expression,"right")
            # logging memory taken by the df
            self.mem.get_mem_usage(result_df)
            # logging spark schema 
            self.logger.debug("logging intermediate dataframe after right join operation")
            self.metrics.log_df_metrics(spark_df=result_df,spark_df_name="result_df")
            return result_df
        except Exception as e:
            self.logger.error(str(e))
            raise
    # here outer join is actually full outer join there is no difference between them
    def outer_join_df(self,left_df,right_df,join_expression):
        try:
            # perform join operation
            result_df = left_df.join(right_df,join_expression,"outer")
            # logging memory taken by the df
            self.mem.get_mem_usage(result_df)
            # logging spark schema 
            self.logger.debug("logging intermediate dataframe after outer join operation")
            self.metrics.log_df_metrics(spark_df=result_df,spark_df_name="result_df")
            return result_df
        except Exception as e:
            self.logger.error(str(e))
            raise