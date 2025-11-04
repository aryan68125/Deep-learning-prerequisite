class Log4j:
    def __init__(self, spark):
        # Get a log4j instance
        log4j = spark._jvm.org.apache.log4j
        # Create a logger attribute
        # put your organization name as a root class 
        root_class = "credencys.aditya.spark"
        conf = spark.sparkContext.getConf()
        app_name = conf.get("spark.app.name")
        self.logger = log4j.LogManager.getLogger(root_class + "." + app_name)

    def warn(self,message):
        self.logger.warn(message)
    
    def info(self,message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self,message):
        self.logger.debug(message)

class LogSparkDataframe:
    def __init__(self,spark):
        self.sp = spark
        self.logger = Log4j(spark)
    # This will log the dataframe
    def log_df(self,spark_df,spark_df_name):
        self.logger.info(f">>>>> {spark_df_name} dataframe:\n{spark_df.limit(25).toPandas().to_string(index=False)}")
    # This will log the database schema
    def log_df_metrics(self,spark_df,spark_df_name):
        schema_str = spark_df._jdf.schema().treeString()
        self.logger.debug(f"{spark_df_name} :: operation - LogSparkDataframe :: Spark DataFrame Schema (expanded): {schema_str}")