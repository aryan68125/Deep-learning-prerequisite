from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j, LogSparkDataframe
# import related to custom spark configurations
from lib.utils import get_spark_app_config
# logging related imports 
import os

# Imports related to ingest data
from lib.ingest_data import IngestData
# Transform data
from transformations.dataframe_transformations import DataFrameTransformations

if __name__ == "__main__":
    # logging related logic
    # Get the current project's directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the Log4j.properties file directory
    log4j_config_path = os.path.join(project_dir, "log4j_properties", "log4j.properties")
    # Save the directory where the generated log files must reside
    log_dir = os.path.join(project_dir, "log4j_properties", "logs")
    # Create the directory where the log files must be kept if not present
    os.makedirs(log_dir, exist_ok=True)

    conf = get_spark_app_config()
    spark = (
        SparkSession
        .builder
        .config(conf=conf)
        .config("spark.driver.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .config("spark.executor.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .getOrCreate()
    )

    # initialize logger class 
    logger = Log4j(spark)

    # initialize the spark dataframe logger 
    sp_df_logger = LogSparkDataframe(spark)

    # logging some debug related stuff 
    logger.debug(f"log4j.properties file dir = {log4j_config_path}")
    logger.debug(f"log files dir = {log_dir}")
    logger.debug(f"log dir exists = {os.path.exists(log_dir)}")
    
    logger.info("Reading the data from the directory")
    dataset_dir = os.path.join(project_dir,"dataset")
    # file_name = "sf-fire-calls.csv" nor mally we provide the file name by hard coding it in the app 
    # But here the dataset file name is supplied via spark.conf file
    file_name = conf.get("file_name_csv")
    file_dir = os.path.join(dataset_dir,file_name)
    logger.debug(f"file_name_csv dir = {file_dir}")
    
    # The function must taken in file_dir csv file and then returns a spark dataFrame
    # import data from a csv file
    ingest_data = IngestData(spark)
    spark_df = ingest_data.import_data_csv(file_dir=file_dir)

    # log spark dataframe
    sp_df_logger.log_df(spark_df=spark_df,spark_df_name="spark_df")

    # import data from a json file
    file_name = conf.get("file_name_json")
    file_dir = os.path.join(dataset_dir,file_name)
    logger.debug(f"file_name_json dir = {file_dir}")
    spark_df_json = ingest_data.import_data_json(file_dir=file_dir)

    # log spark dataframe
    sp_df_logger.log_df(spark_df=spark_df,spark_df_name="spark_df_json")

    # import data from a parquet file
    file_name = conf.get("file_name_parquet")
    file_dir = os.path.join(dataset_dir,file_name)
    logger.debug(f"file_name_json dir = {file_dir}")
    spark_df_parquet = ingest_data.import_data_parquet(file_dir=file_dir)

    # log spark dataframe
    sp_df_logger.log_df(spark_df=spark_df,spark_df_name="spark_df_parquet")

    # This line is for debugging only comment after <required to see the partitions of spark dataFrame>
    # input("Please enter")
    spark.stop()