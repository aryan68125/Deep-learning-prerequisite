def is_running_as_job(dbutils):
    try:
        dbutils.widgets.get("catalog")
        return True
    except:
        return False


def load_config(dbutils):
    if is_running_as_job(dbutils):
        return {
            "catalog": dbutils.widgets.get("catalog"),
            "schema": dbutils.widgets.get("schema"),
            "delta_table_volume": dbutils.widgets.get("delta_table_volume"),
            "data_ingestion_volume": dbutils.widgets.get("data_ingestion_volume"),
            "delta_table_schema_store": dbutils.widgets.get("delta_table_schema_store"),
            "delta_table_checkpoints": dbutils.widgets.get("delta_table_checkpoints"),
            "delta_table_path": dbutils.widgets.get("delta_table_path"),
            "customers_data": dbutils.widgets.get("customers_data"),
            "orders_data": dbutils.widgets.get("orders_data"),
            "sales_data": dbutils.widgets.get("sales_data"),
            "customers_table": dbutils.widgets.get("customers_table"),
            "customer_delta_table_schema_path": dbutils.widgets.get("customer_delta_table_schema_path"),
            "customer_delta_table_checkpoint_path": dbutils.widgets.get("customer_delta_table_checkpoint_path"),
            "customer_delta_table_location": dbutils.widgets.get("customer_delta_table_location"),
            "is_job": True
        }

    # LOCAL / DEV
    return {
        "catalog": "job_orchestration",
        "schema": "default",
        "delta_table_volume": "customers_volume",
        "data_ingestion_volume": "job_orchestration_volume",
        "delta_table_schema_store": "/Volumes/job_orchestration/default/customers_volume/schema_store",
        "delta_table_checkpoints": "/Volumes/job_orchestration/default/customers_volume/checkpoints",
        "delta_table_path": "/Volumes/job_orchestration/default/customers_volume/customers_table",
        "customers_data": "/Volumes/job_orchestration/default/job_orchestration_volume/ingest_data/customers_data/",
        "orders_data": "/Volumes/job_orchestration/default/job_orchestration_volume/ingest_data/orders_data/",
        "sales_data": "/Volumes/job_orchestration/default/job_orchestration_volume/ingest_data/sales_data/",
        "customers_table": "customers_table",
        "customer_delta_table_schema_path": "/Volumes/job_orchestration/default/customers_volume/schema_store/",
        "customer_delta_table_checkpoint_path": "/Volumes/job_orchestration/default/customers_volume/checkpoints/",
        "customer_delta_table_location": "job_orchestration.default.customers_table",
        "is_job": False
    }
