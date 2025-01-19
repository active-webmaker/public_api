import os
import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2025, 1, 1),  # 적절히 조정
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

with DAG(
    dag_id='spark_etl_dag',
    default_args=default_args,
    schedule_interval='@daily',  # 매일 1회
    catchup=False
) as dag:

    spark_etl = SparkSubmitOperator(
        task_id='spark_etl_task',
        application=os.path.join(
            '/spark_project/spark_jobs',
            'spark_etl.py'
        ),
        conn_id='spark_default',  # Airflow UI에서 연결 설정
        executor_cores=2,   # Spark 실행코어 수
        executor_memory='2g',
        driver_memory='2g',
        name='spark_etl_job',
        verbose=True
    )

    spark_etl
