from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType


class FlightSchemaMixin:
    """
    This method defines the spark dataFrame schema using Programmatical method
    """
    def return_flight_schema(self):
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