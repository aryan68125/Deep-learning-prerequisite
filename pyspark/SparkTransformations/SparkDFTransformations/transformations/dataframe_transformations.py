from pyspark.sql.functions import to_date
class DataFrameTransformations:
    def __init__(self,spark):
        self.spark_object = spark
    def count_by_country(self,spark_df):
        intermediate_result_df = (
                spark_df
                .where("CallType is not null")
                .select("CallType","Zipcode")
                .groupby("CallType","Zipcode")
            )
        row_count = intermediate_result_df.count()
        result_df = row_count.orderBy("count",ascending=False)
        return result_df
    
    """This methods onverts the column with dates in string datatype to date datatype"""
    def convert_to_date_type(self, spark_df, date_format, col_name):
        """Converts a string column to date using the given format."""
        return spark_df.withColumn(col_name, to_date(col_name, date_format))