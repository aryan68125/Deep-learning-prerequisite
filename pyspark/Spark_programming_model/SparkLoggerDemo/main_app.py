from pyspark.sql import SparkSession
from lib.logger import Log4j

# logging related imports 
import os
if __name__ == "__main__":
    # logging related logic
    project_dir = os.path.dirname(os.path.abspath(__file__))
    log4j_config_path = os.path.join(project_dir, "log4j_properties", "log4j.properties")
    log_dir = os.path.join(project_dir, "log4j_properties", "logs")

    os.makedirs(log_dir, exist_ok=True)

    spark = (
        SparkSession
        .builder
        .appName("SparkLoggerDemo")
        .master("local[2]")
        .config("spark.driver.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .config("spark.executor.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .getOrCreate()
    )

    logger = Log4j(spark)
    logger.warn(">>>Starting SparkLoggerDemo")
    logger.warn(">>>Ending SparkLoggerDemo")

    spark.stop()