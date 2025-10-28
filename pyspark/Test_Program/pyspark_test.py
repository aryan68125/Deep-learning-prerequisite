from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

# 1️⃣ Create Spark session in local mode using all cores
spark = SparkSession.builder \
    .appName("CPU_Stress_Test") \
    .master("local[*]") \
    .getOrCreate()

print("Spark master:", spark.sparkContext.master)
print("Total cores Spark sees:", spark.sparkContext.defaultParallelism)

# 2️⃣ Create a large synthetic dataset (e.g., 100 million rows)
num_rows = 999_999_999
num_partitions = spark.sparkContext.defaultParallelism  # same as CPU cores

df = spark.range(0, num_rows, numPartitions=num_partitions) \
           .withColumn("random_val", rand())

# 3️⃣ Apply heavy transformations — wide operations
#    Force Spark to use multiple stages and shuffles
aggregated_df = (
    df.withColumn("squared", col("random_val") * col("random_val"))
      .groupBy((col("id") % 100).alias("group"))  # 100 groups
      .avg("squared")                            # aggregation
      .orderBy("group")                          # shuffle operation
)

# 4️⃣ Trigger computation (action)
aggregated_df.show()
