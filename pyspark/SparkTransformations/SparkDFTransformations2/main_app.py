from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j, LogSparkDataframe
# import related to custom spark configurations
from lib.utils import get_spark_app_config
# imports related to exporting dataframe
from lib.write_df import ExportSparkDataFrame
# import writing sparkdf to tables related stuff
from lib.load_df_data_into_table import LoadSparkDFIntoTable
# logging related imports 
import os

# Imports related to ingest data
from lib.ingest_data import IngestData
# Transform data
from transformations.dataframe_transformations import DataFrameTransformations

# imports related to cleanup when the main_app.py is re-run
from lib.clean_up_file_system import CleanupAppFileSystemOnReRun

# imports related to generating dataFrame
from unit_testing.generate_dataframe import GenerateDataFrame

if __name__ == "__main__":
    # logging related logic
    # Get the current project's directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # cleanup loggic on main_app.py re-run
    # initialize the cleanup class
    cleanup = CleanupAppFileSystemOnReRun(project_dir)
    cleanup.execute_cleanup(clean_logs=True)

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
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.13:4.0.1")
        .enableHiveSupport()
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

    """Create DataFrame STARTS"""
    gen_df = GenerateDataFrame(spark)
    data_list = [
        ("Rollex","28","1","2002"),
        ("Ballistic","23","5","81"),
        ("Shotgun","12","12","6"),
        ("Artillery","7","8","63"),
        ("Ballistic","23","5","81"),
    ]
    generated_df = gen_df.generate_dataframe(data_list)
    sp_df_logger.log_df(spark_df=generated_df,spark_df_name="generated_df")
    """Create DataFrame ENDS""" 

    """Transformation STARTS"""
    df_t = DataFrameTransformations(spark)

    # add a uniquely identifiable id for the rows
    generated_df = df_t.create_unique_identifier(spark_df=generated_df) 
    # log the output dataframe
    sp_df_logger.log_df(spark_df=generated_df,spark_df_name="generated_df")
    sp_df_logger.log_df_metrics(spark_df=generated_df,spark_df_name="generated_df")

    # process date and make all the inconsitent two digit and three digit year into 4 digit year
    processed_date_df = df_t.process_date_col_year(spark_df=generated_df,col_name="year")
    # log the output dataframe
    sp_df_logger.log_df(spark_df=processed_date_df,spark_df_name="processed_date_df")
    sp_df_logger.log_df_metrics(spark_df=processed_date_df,spark_df_name="processed_date_df")
    """Transformation ENDS"""
    
    # This line is for debugging only comment after <required to see the partitions of spark dataFrame>
    # input("Please enter")
    spark.stop()




    
