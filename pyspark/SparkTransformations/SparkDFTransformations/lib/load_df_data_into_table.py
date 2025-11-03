from .logger import Log4j
"""
This class will load the data in a spark dataFrame into a table
"""
class LoadSparkDFIntoTable: 
    def __init__(self,spark):
        self.spark = spark
        self.logger = Log4j(spark)
    
    """This method will save the spark dataFrame into a spark managed table"""
    def save_df_to_spark_managed_table(self,spark_df,partition_config:list = [],mode:str = "overwrite",db_name:str = "",table_name:str=""):
        try:
            if not db_name == "":
                # To be on the safer side create the db just in case
                query = f"""
                CREATE DATABASE IF NOT EXISTS {db_name}
                """
                self.spark.sql(query)
                # tell spark to use the created database 
                self.spark.catalog.setCurrentDatabase(db_name)
                """
                5 is the partition no
                OP_CARRIER = column name 
                ORIGIN = column name
                """
                writer = (
                    spark_df
                    .write
                    .mode(mode)
                )
                writer = self.partition_handler(partition_config,writer)
                writer.saveAsTable(table_name)
                self.logger.info(self.spark.catalog.listTables(db_name))
            else:
                writer = (
                    spark_df
                    .write
                    .mode(mode)
                )
                writer = self.partition_handler(partition_config,writer)
                writer.saveAsTable(table_name)
                self.logger.info(self.spark.catalog.listTables("default"))
        except Exception as e:
            self.logger.error(str(e))
    
    # util methods 
    def partition_handler(self,partition_config,writer):
        try:
            # Handle None or empty list
            if not partition_config:
                partition_config = []
                return writer

            num_buckets = None
            partition_cols = []

            # Detect if first element is an integer (buckets)
            if partition_config and isinstance(partition_config[0], int):
                num_buckets = partition_config[0]
                partition_cols = partition_config[1:]
            else:
                partition_cols = partition_config

            # Apply bucketing (if defined)
            if num_buckets and partition_cols:
                self.logger.debug("Applying bucketBy on the table")
                writer = writer.bucketBy(num_buckets, *partition_cols).sortBy(*partition_cols)
            elif partition_cols:
                self.logger.debug("Applying partitionBy on the table")
                writer = writer.partitionBy(*partition_cols)
            return writer
        except Exception as e:
            self.logger.error(str(e))
            raise

    def generate_logs(self, conf):
        try:
            db_name = conf.get("db_name")
            table_name = conf.get("flight_table_name")

            # Switch to the correct database
            if db_name:
                self.spark.catalog.setCurrentDatabase(db_name)

            # Get the first 24 records
            query = f"SELECT * FROM {table_name} LIMIT 24"
            result_df = self.spark.sql(query)

            # Capture the DataFrame as a formatted string (no stdout print)
            result_str = result_df._jdf.showString(24, 0, False)
            self.logger.info(f"\nFirst 24 records from table '{table_name}':\n{result_str}")

            # Get count value properly
            count_df = self.spark.sql(f"SELECT COUNT(*) as total FROM {table_name}")
            count_value = count_df.collect()[0]['total']
            self.logger.info(f"The number of records in table '{table_name}' => {count_value}")

            # Log schema
            self.logger.debug(f"Schema of table '{db_name}.{table_name}':")
            for field in result_df.schema.fields:
                self.logger.debug(f"  {field.name}: {field.dataType.simpleString()}")

        except Exception as e:
            self.logger.error(str(e))
            raise