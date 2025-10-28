from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

# For keeping track of time taken to complete the spark computation task
import time
import sys

start = time.time()

#Create Spark session in local mode using all cores
spark = SparkSession.builder \
    .appName("CPU_Stress_Test") \
    .master("local[*]") \
    .getOrCreate()

print("Spark master:", spark.sparkContext.master)
print("Total cores Spark sees:", spark.sparkContext.defaultParallelism)

#Create a large synthetic dataset (e.g., 100 million rows)
num_rows = 9_999_999
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
print(f"Estimated DataFrame size in memory: {total_mb:.2f} MB")

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
""")
