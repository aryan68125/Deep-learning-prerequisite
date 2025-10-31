# MEthod 1 of creating a spark context
# from pyspark import SparkConf,SparkContext
# if __name__ == "__main__":
#     conf = (
#                 SparkConf()
#                 .setMaster("local[3]")
#                 .setAppName("RddApplication")
#             )
#     spark_context_obj = SparkContext(conf=conf)

from pyspark.sql import SparkSession
from pyspark import SparkConf
from lib.logger import Log4j
import os
import sys
from collections import namedtuple
SurveyRecord = namedtuple("SurveyRecord", ["Age", "Gender", "Country", "State"])

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

    conf = (
                SparkConf()
                .setMaster("local[3]")
                .setAppName("RddApplication")
            )
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
    spark_context = spark.sparkContext
    logger = Log4j(spark)

    if len(sys.argv) != 2:
        logger.error(f"Usage : Spark file name <filename>")
        sys.exit(-1)

    # Create an RDD after reading the text file from the file system
    linesRDD = spark_context.textFile(sys.argv[1])

    # RDD processing 
    partitionedRDD = linesRDD.repartition(2)
    
    colsRDD = partitionedRDD.map(lambda line: line.replace('"', '').split(","))
    for line in linesRDD.take(5):
        print(line)

    # Attach a schema to my RDD
    selectRDD = colsRDD.map(
    lambda cols: SurveyRecord(
            int(cols[1]) if cols[1].isdigit() else 0,
            cols[2],
            cols[3],
            cols[4]
        )
    )

    # Filter the record
    filteredRDD = selectRDD.filter(lambda r: r.Age < 40)
    # Group the record by country
    kvRDD = filteredRDD.map(lambda r: (r.Country, 1))
    # Reduce by key method
    countRDD = kvRDD.reduceByKey(lambda v1, v2: v1 + v2)

    colsList = countRDD.collect()
    for x in colsList:
        logger.info(x)
