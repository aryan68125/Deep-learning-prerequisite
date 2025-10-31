from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j
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

    # check if the log4j.properties file exists
    logger.debug(f"path to logger4j.properties {os.path.exists(log4j_config_path)}")
    
    logger.info("Reading the data from the directory")
    dataset_dir = os.path.join(project_dir,"dataset")
    # file_name = "sf-fire-calls.csv" nor mally we provide the file name by hard coding it in the app 
    # But here the dataset file name is supplied via spark.conf file
    file_name = conf.get("file_name")
    file_dir = os.path.join(dataset_dir,file_name)
    logger.debug(f"dataset_csv_file_dir = {file_dir}")
    
    # The function must taken in file_dir csv file and then returns a spark dataFrame
    ingest_data = IngestData(spark)
    spark_df = ingest_data.import_data_csv(file_dir=file_dir)
    # Create a View using spark_df
    spark_df.createOrReplaceTempView("fire_data")

    # Spark sql operations
    query = """
    SELECT * FROM fire_data LIMIT 25;
    """
    result = spark.sql(query)
    result_pd_df = result.limit(25).toPandas().to_string(index=False)
    logger.info(f"view ==> \n {result_pd_df}")
    result.show()
    
    # This line is for debugging only comment after <required to see the partitions of spark dataFrame>
#     input("Please enter")
    spark.stop()