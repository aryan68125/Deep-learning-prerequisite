from pyspark import StorageLevel
from .logger import Log4j

class GetDataFrameMemory:
    """
    This class measures the approximate memory usage of a Spark DataFrame.
    It uses caching and executor memory metrics to get insights.
    """
    def __init__(self, spark):
        self.spark_session_object = spark
        self.logger = Log4j(spark)

    def get_mem_usage(self, spark_df):
        try:
            # Persist DataFrame in memory so Spark tracks it
            spark_df.persist(StorageLevel.MEMORY_ONLY)
            spark_df.count()  # Trigger cache materialization

            # Get JVM reference
            jvm = self.spark_session_object._jvm

            # Get executor memory status (Scala Map)
            scala_map = self.spark_session_object._jsc.sc().getExecutorMemoryStatus()

            # Convert Scala Map -> Java Map
            java_map = jvm.scala.collection.JavaConverters.mapAsJavaMapConverter(scala_map).asJava()

            # Convert Java Map -> Python dict
            memory_info = {
                entry.getKey(): (entry.getValue()._1(), entry.getValue()._2())
                for entry in java_map.entrySet()
            }

            if not memory_info:
                self.logger.warn("No executor memory metrics found — possibly running in local mode.")
                approx_size = spark_df.rdd.map(lambda x: len(str(x))).sum()
                self.logger.debug(f"Approximate DataFrame size (bytes) [Driver]: {self.convert_bytes_to_mb(approx_size)} MB")
                return {"mem": approx_size}

            self.logger.debug(f"Executor memory usage snapshot: {self.convert_bytes_to_mb(memory_info)} MB")

            # Compute used memory per executor
            mem_used = {
                host: total - remaining
                for host, (total, remaining) in memory_info.items()
            }

            self.logger.debug(f"Memory used by executors (MB): {self.convert_bytes_to_mb(mem_used)} MB")
            return {"mem": self.convert_bytes_to_mb(mem_used)}

        except Exception as e:
            self.logger.error(f"Error while fetching DataFrame memory usage: {e}")
            return {}
    def convert_bytes_to_mb(self, value):
        try:
            if isinstance(value, (int, float)):
                return round(value / (1024 * 1024), 2)
            elif isinstance(value, tuple):
                return tuple(round(v / (1024 * 1024), 2) if isinstance(v, (int, float)) else v for v in value)
            elif isinstance(value, dict):
                return {k: self.convert_bytes_to_mb(v) for k, v in value.items()}
            elif isinstance(value, (list, set)):
                return [self.convert_bytes_to_mb(v) for v in value]
            else:
                return value
        except Exception as e:
            self.logger.error(f"Error in convert_bytes_to_mb: {e}")
            return value