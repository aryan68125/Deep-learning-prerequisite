def path_has_csv_files(dbutils, path):
    try:
        files = dbutils.fs.ls(path)
        return any(f.name.lower().endswith(".csv") for f in files)
    except:
        return False


def run_autoloader(spark, dbutils, cfg):
    if not path_has_csv_files(dbutils, cfg["customers_data"]):
        return False

    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", cfg["customer_delta_table_schema_path"])
        .load(cfg["customers_data"])
    )

    (
        df.writeStream
        .option("checkpointLocation", cfg["customer_delta_table_checkpoint_path"])
        .option("mergeSchema", "true")
        .outputMode("append")
        .trigger(availableNow=True)
        .table(cfg["customer_delta_table_location"])
    )

    return True
