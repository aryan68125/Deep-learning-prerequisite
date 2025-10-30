from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j
# import related to custom spark configurations
from lib.utils import get_saprk_app_config
# logging related imports 
import os

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

    conf = get_saprk_app_config()
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

    logger = Log4j(spark)
    logger.warn(">>>Starting SparkLoggerDemo")
    conf_out = spark.sparkContext.getConf()
    logger.info(">>>{conf_out}")
    logger.debug(">>Testing the debug messages")
    logger.warn(">>>Ending SparkLoggerDemo")
    logger.error(">>>System.out.println")
    spark.stop()