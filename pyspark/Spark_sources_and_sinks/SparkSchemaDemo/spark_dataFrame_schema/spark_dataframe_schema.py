from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType

class FlightSchemaMixin:
    def return_flight_df_schema(self):
        flight_schema = StructType([
            StructField("FL_DATE", DateType(), True),
            StructField("OP_CARRIER", StringType(), True),
            StructField("OP_CARRIER_FL_NUM", IntegerType(), True),
            StructField("ORIGIN", StringType(), True),
            StructField("ORIGIN_CITY_NAME", StringType(), True),
            StructField("DEST", StringType(), True),
            StructField("DEST_CITY_NAME", StringType(), True),
            StructField("CRS_DEP_TIME", IntegerType(), True),
            StructField("DEP_TIME", IntegerType(), True),
            StructField("WHEELS_ON", IntegerType(), True),
            StructField("TAXI_IN", IntegerType(), True),
            StructField("CRS_ARR_TIME", IntegerType(), True),
            StructField("ARR_TIME", IntegerType(), True),
            StructField("CANCELLED", IntegerType(), True),
            StructField("DISTANCE", IntegerType(), True)
        ])
        return flight_schema
