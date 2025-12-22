def mkdir_if_not_exists(dbutils, path):
    try:
        dbutils.fs.ls(path)
    except:
        dbutils.fs.mkdirs(path)


def prepare_directories(dbutils, cfg):
    mkdir_if_not_exists(dbutils, cfg["delta_table_schema_store"])
    mkdir_if_not_exists(dbutils, cfg["delta_table_checkpoints"])
    mkdir_if_not_exists(dbutils, cfg["delta_table_path"])

    mkdir_if_not_exists(dbutils, cfg["customers_data"])
    mkdir_if_not_exists(dbutils, cfg["orders_data"])
    mkdir_if_not_exists(dbutils, cfg["sales_data"])
