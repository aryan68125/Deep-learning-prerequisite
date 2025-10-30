from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j
# import related to custom spark configurations
from lib.utils import get_spark_app_config
# logging related imports 
import os
# metrics monitoring related imports
from lib.app_monitor import GetDataFrameMemory

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
    logger.info("Reading the data from the directory")
    dataset_dir = os.path.join(project_dir,"dataset")
    # file_name = "sf-fire-calls.csv" nor mally we provide the file name by hard coding it in the app 
    # But there the dataset file name is supplied via spark.conf file
    file_name = conf.get("file_name")
    file_dir = os.path.join(dataset_dir,file_name)
    logger.info(f"dataset_csv_file_dir = {file_dir}")
    try:
        spark_df = (
            spark
            .read
            .format("csv")
            .option("header","true")
            .option("inferschema","true")
            .load(file_dir)
                    )
        # initialize the metrics class
        metrics = GetDataFrameMemory(spark)
        logger.info(f"spark_df created successfully from {file_dir} dataset file")
        logger.info(f"The memory taken by the dataFrame is = {metrics.get_mem_usage(spark_df).get("mem")} MB")
        logger.info(f"DataFrame sample:\n{spark_df.limit(5).toPandas().to_string(index=False)}")

    except Exception as e:
        logger.error(e)
    spark_df.show()
    spark.stop()