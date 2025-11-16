from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

# For keeping track of time taken to complete the spark computation task
import time
import sys

# spark Log4j logging related imports
from pyspark import SparkConf
import os

# Logging related Spark configurations setup
# log4k.properties configuration file path setup
# Determine project directory — works in both script & notebook
try:
    project_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in interactive mode (e.g., Jupyter)
    project_dir = os.getcwd()

log4j_config_path = os.path.join(project_dir, "log4j_properties", "log4j.properties")

# Create SparkConf with custom Log4j config
conf = (
    SparkConf()
    .setAppName("CPU_Stress_Test")
    .setMaster("local[*]")
     # JVM property for log4j v1
    .set("spark.driver.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_config_path}")
    .set("spark.executor.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_config_path}")
    # Ensure executors also get it via --files equivalent
    .set("spark.files", log4j_config_path)
)


start = time.time()

#Create Spark session in local mode using all cores
spark = SparkSession.builder \
    .config(conf=conf)\
    .getOrCreate()

print("Spark master:", spark.sparkContext.master)
print("Total cores Spark sees:", spark.sparkContext.defaultParallelism)

#Create a large synthetic dataset (e.g., 100 million rows)
num_rows = 9999999999
num_partitions = spark.sparkContext.defaultParallelism  # same as CPU cores

df = spark.range(0, num_rows, numPartitions=num_partitions) \
           .withColumn("random_val", rand())

# I want to know the amount of Ram taken by the dataFrame in Mbs
# converting df to rdd rows
rdd = df.rdd.map(lambda row: row.asDict())
# Estimate memory size of one partition
def estimate_partition_size(partition):
    import sys
    size = 0
    for record in partition:
        size += sys.getsizeof(record)
    yield size

partition_sizes = rdd.mapPartitions(estimate_partition_size).collect()
total_bytes = sum(partition_sizes)
total_mb = total_bytes / (1024 * 1024)

#Apply heavy transformations — wide operations
#    Force Spark to use multiple stages and shuffles
aggregated_df = (
    df.withColumn("squared", col("random_val") * col("random_val"))
      .groupBy((col("id") % 100).alias("group"))  # 100 groups
      .avg("squared")                            # aggregation
      .orderBy("group")                          # shuffle operation
)

#Trigger computation (action)
aggregated_df.show()

end = time.time()

time_taken_in_sec = end - start
time_taken_in_min = (end - start) / 60
print(f"""
        Spark task complete!
        Time taken in seconds = {time_taken_in_sec}
        Time taken in minutes = {time_taken_in_min}
        Estimated DataFrame size in memory: {total_mb:.2f} MB
""")
