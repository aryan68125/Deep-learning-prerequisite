import json

def publish_task_output(dbutils, cfg):
    output = {
        "catalog": cfg["catalog"],
        "schema": cfg["schema"],
        "data_ingestion_volume": cfg["data_ingestion_volume"],
        "customers_table": cfg["customers_table"],
        "orders_data": cfg["orders_data"],
        "sales_data": cfg["sales_data"]
    }

    if cfg["is_job"]:
        dbutils.jobs.taskValues.set(key="metadata", value=json.dumps(output))
    else:
        print(output)
