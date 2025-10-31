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