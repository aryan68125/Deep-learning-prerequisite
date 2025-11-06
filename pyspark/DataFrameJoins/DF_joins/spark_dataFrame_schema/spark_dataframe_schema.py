from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType


class FlightSchemaMixin:
    """
    This method defines the spark dataFrame schema using Programmatical method
    """
    def return_invoice_df_schema(self):
        invoice_schema = StructType([
            StructField("InvoiceNo", IntegerType(), True),
            StructField("StockCode", StringType(), True),
            StructField("Description", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("InvoiceDate", StringType(), True),
            StructField("UnitPrice", DoubleType(), True),
            StructField("CustomerID", IntegerType(), True),
            StructField("Country", StringType(), True)
        ])
        return invoice_schema

    """
    This method defines the spark dataFrame schema using DDL method
    """
    def return_invoice_schema_ddl(self):
        invoice_schema_ddl = """
            InvoiceNo INT,
            StockCode STRING,
            Description STRING,
            Quantity INT,
            InvoiceDate TIMESTAMP,
            UnitPrice DOUBLE,
            CustomerID INT,
            Country STRING
        """
        return invoice_schema_ddl