import os
import shutil

import time

"""
This class will cleanup the data when the spark application re-runs

The files like logs, metastore , spark-warehouse etc.. will be cleaned up (deleted from the file system)
"""
class CleanupAppFileSystemOnReRun:
    def __init__(self,project_dir):
        self.project_dir = project_dir

    def execute_cleanup(self,clean_logs:bool = False):
        self.derby_logs_cleanup()
        self.spark_warehouse_cleanup()
        self.metastore_cleanup()
        if clean_logs == True:
            self.logs_cleanup()
        # time.sleep(5)

    """This will cleanup the derby.logs"""
    def derby_logs_cleanup(self):
        try:
            derby_logs_dir = os.path.join(self.project_dir, "derby.log")
            if os.path.exists(derby_logs_dir):
                os.remove(derby_logs_dir)
                print(f"Deleted existing derby.log file: {derby_logs_dir}")
            else:
                print(f"derby.log file does not exists: {derby_logs_dir}")
        except Exception as e:
            print(str(e))
            raise
    
    """This will cleanup the spark_warehouse"""
    def spark_warehouse_cleanup(self):
        try:
            spark_warehouse_dir = os.path.join(self.project_dir, "spark-warehouse")
            if os.path.exists(spark_warehouse_dir):
                shutil.rmtree(spark_warehouse_dir)
                print(f"Deleted existing spark-warehouse directory: {spark_warehouse_dir}")
            else:
                print(f"spark-warehouse directory does not exists: {spark_warehouse_dir}")
        except Exception as e:
            print(str(e))
            raise

    """This will cleanup the metastore_db"""
    def metastore_cleanup(self):
        try:
            metastore_dir = os.path.join(self.project_dir, "metastore_db")
            if os.path.exists(metastore_dir):
                shutil.rmtree(metastore_dir)
                print(f"Deleted existing metastore directory: {metastore_dir}")
            else:
                print(f"metastore directory does not exists: {metastore_dir}")
        except Exception as e:
            print(str(e))
            raise

    """This will clean the logs folder"""
    def logs_cleanup(self):
        try:
            log_dir = os.path.join(self.project_dir, "log4j_properties", "logs")
            # delete the log directory
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                print(f"Deleted existing log directory: {log_dir}")
            else:
                print(f"Log directory does not exist: {log_dir}")
        except Exception as e:
            print(str(e))
            raise
            
