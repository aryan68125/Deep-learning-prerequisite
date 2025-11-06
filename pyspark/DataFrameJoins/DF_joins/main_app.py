from pyspark.sql import SparkSession
# import related to logging
from lib.logger import Log4j, LogSparkDataframe
# import related to custom spark configurations
from lib.utils import get_spark_app_config
# imports related to exporting dataframe
from lib.write_df import ExportSparkDataFrame
# import writing sparkdf to tables related stuff
from lib.load_df_data_into_table import LoadSparkDFIntoTable
# logging related imports 
import os

# Imports related to ingest data
from lib.ingest_data import IngestData
# Transform data
from transformations.dataframe_transformations import DataFrameTransformations
# imports related to dataframe joins
from joins.df_joins import DataFrameJoins

# imports related to cleanup when the main_app.py is re-run
from lib.clean_up_file_system import CleanupAppFileSystemOnReRun

# imports related to generating dataFrame
from unit_testing.generate_dataframe import GenerateDataFrame

if __name__ == "__main__":
    # logging related logic
    # Get the current project's directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # cleanup loggic on main_app.py re-run
    # initialize the cleanup class
    cleanup = CleanupAppFileSystemOnReRun(project_dir)
    cleanup.execute_cleanup(clean_logs=True)

    # Get the Log4j.properties file directory
    log4j_config_path = os.path.join(project_dir, "log4j_properties", "log4j.properties")
    # Save the directory where the generated log files must reside
    log_dir = os.path.join(project_dir, "log4j_properties", "logs")
    # Create the directory where the log files must be kept if not present
    os.makedirs(log_dir, exist_ok=True)

    conf = get_spark_app_config()
    spark = (
        SparkSession
        .builder
        .config(conf=conf)
        .config("spark.driver.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .config("spark.executor.extraJavaOptions",
                f"-Dlog4j.configuration=file:{log4j_config_path} -Dcustom.log.dir={log_dir}")
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.13:4.0.1")
        .enableHiveSupport()
        .getOrCreate()
    )

    # initialize logger class 
    logger = Log4j(spark)

    # initialize the spark dataframe logger 
    sp_df_logger = LogSparkDataframe(spark)

    # logging some debug related stuff 
    logger.debug(f"log4j.properties file dir = {log4j_config_path}")
    logger.debug(f"log files dir = {log_dir}")
    logger.debug(f"log dir exists = {os.path.exists(log_dir)}")
    
    logger.info("Reading the data from the directory")
    dataset_dir = os.path.join(project_dir,"dataset")

    ######################################
    # CREATE A DATAFRAME AND PERFORM DATAFRAME JOIN ON IT STARTS
    ######################################
    """Create DataFrame STARTS"""
    gen_df = GenerateDataFrame(spark)

    # Generate orders dataFrame
    data_list = [
                ("01", "02", 350, 1),
                ("01", "04", 580, 1),
                ("01", "07", 320, 2),
                ("02", "03", 450, 1),
                ("02", "06", 220, 1),
                ("03", "01", 195, 1),
                ("04", "09", 270, 3),
                ("04", "08", 410, 2),
                ("05", "02", 350, 1)
            ]
    column_name_list = ["order_id", "prod_id", "unit_price", "qty"]
    generated_order_df = gen_df.generate_dataframe(data_list=data_list,column_name_list=column_name_list)
    # logging dataFrame
    sp_df_logger.log_df(spark_df=generated_order_df,spark_df_name="generated_order_df")

    # Generate product_list dataFrame
    data_list = [
                    ("01", "Scroll Mouse", 250, 20),
                    ("02", "Optical Mouse", 350, 20),
                    ("03", "Wireless Mouse", 450, 50),
                    ("04", "Wireless Keyboard", 580, 50),
                    ("05", "Standard Keyboard", 360, 10),
                    ("06", "16 GB Flash Storage", 240, 100),
                    ("07", "32 GB Flash Storage", 320, 50),
                    ("08", "64 GB Flash Storage", 430, 25)
                ]
    column_name_list = ["prod_id", "prod_name", "list_price", "qty"]
    generated_product_df = gen_df.generate_dataframe(data_list=data_list,column_name_list=column_name_list)
    # logging dataFrame 
    sp_df_logger.log_df(spark_df=generated_product_df,spark_df_name="generated_product_df")
    """Create DataFrame ENDS""" 

    """Join operation STARTS"""
    # initialize df transformation class
    df_t = DataFrameTransformations(spark)
    # initialize df export class
    df_exp = ExportSparkDataFrame(spark)
    # initialize dataframe join classes 
    df_joins = DataFrameJoins(spark)

    #  Handling column ambiguity 
    product_renamed_df = generated_product_df.withColumnRenamed("qty","reorder_qty").withColumnRenamed("prod_id","prod_id2")
    # performing inner join opertion on the two dataFrames
    join_expression = generated_order_df.prod_id == product_renamed_df.prod_id2
    inner_join_df = df_joins.inner_join_df(left_df=generated_order_df,right_df=product_renamed_df,join_expression=join_expression)
    inner_join_df = inner_join_df.select("order_id","prod_id","unit_price","qty","prod_name","list_price","reorder_qty")
    # logging dataFrame
    sp_df_logger.log_df(spark_df=inner_join_df,spark_df_name="inner_join_df")

    #  Handling column ambiguity 
    product_renamed_df = generated_product_df.withColumnRenamed("qty","reorder_qty").withColumnRenamed("prod_id","prod_id2")
    # performing left join operation on the two dataFrames
    join_expression = generated_order_df.prod_id == product_renamed_df.prod_id2
    left_join_df = df_joins.left_join_df(left_df=generated_order_df,right_df=product_renamed_df,join_expression=join_expression)
    left_join_df = left_join_df.select("order_id","prod_id","unit_price","qty","prod_name","list_price","reorder_qty")
    # logging dataFrame
    sp_df_logger.log_df(spark_df=left_join_df,spark_df_name="left_join_df")

    #  Handling column ambiguity 
    product_renamed_df = generated_product_df.withColumnRenamed("qty","reorder_qty").withColumnRenamed("prod_id","prod_id2")
    # performing right join operation on the two dataframes
    join_expression = generated_order_df.prod_id == product_renamed_df.prod_id2
    right_join_df = df_joins.right_join_df(left_df=generated_order_df,right_df=product_renamed_df,join_expression=join_expression)
    right_join_df = right_join_df.select("order_id","prod_id","unit_price","qty","prod_name","list_price","reorder_qty")
    # logging dataFrame
    sp_df_logger.log_df(spark_df=right_join_df,spark_df_name="right_join_df")

    #  Handling column ambiguity 
    product_renamed_df = generated_product_df.withColumnRenamed("qty","reorder_qty").withColumnRenamed("prod_id","prod_id2")
    # performing outer join operation on the two dataFrames
    join_expression = generated_order_df.prod_id == product_renamed_df.prod_id2
    outer_join_df = df_joins.outer_join_df(left_df=generated_order_df,right_df=product_renamed_df,join_expression=join_expression)
    outer_join_df = outer_join_df.select("order_id","prod_id","unit_price","qty","prod_name","list_price","reorder_qty")
    # logging dataFrame
    sp_df_logger.log_df(spark_df=outer_join_df,spark_df_name="outer_join_df")
    """Join operation ENDS"""
    ######################################
    # CREATE A DATAFRAME AND PERFORM DATAFRAME JOIN ON IT ENDS
    ######################################



    # This line is for debugging only comment after <required to see the partitions of spark dataFrame>
    # input("Please enter")
    spark.stop()




    
