from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType


class FlightSchemaMixin:
    """
    This method defines the spark dataFrame schema using Programmatical method
    """
    def return_left_flight_schema(self):
        flight_schema = StructType([
            StructField("id", StringType(), True),
            StructField("FL_DATE", StringType(), True),
            StructField("OP_CARRIER", StringType(), True),
            StructField("OP_CARRIER_FL_NUM", IntegerType(), True),
            StructField("ORIGIN", StringType(), True),
            StructField("ORIGIN_CITY_NAME", StringType(), True),
            StructField("DEST", StringType(), True),
            StructField("DEST_CITY_NAME", StringType(), True)
        ])
        return flight_schema
    
    def return_right_flight_schema(self):
        flight_schema = StructType([
            StructField("id",StringType(),True),
            StructField("CRS_DEP_TIME",IntegerType(),True),
            StructField("DEP_TIME",IntegerType(),True),
            StructField("WHEELS_ON",IntegerType(),True),
            StructField("TAXI_IN",IntegerType(),True),
            StructField("CRS_ARR_TIME",IntegerType(),True),
            StructField("ARR_TIME",IntegerType(),True),
            StructField("CANCELLED",IntegerType(),True),
            StructField("DISTANCE",DoubleType(),True)
        ])
        return flight_schema