from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# 설정 파일 및 PySpark 스크립트 로드
from config import default_args, dag_config, x_cols, y_cols
from pyspark_train import train_and_evaluate_model, filter_target_columns, create_spark_session, load_data


def run_pyspark_job():
    """PySpark 작업 실행"""
    data_path = Variable.get("DATA_PATH")
    
    spark_temp = create_spark_session("temp_spark_for_filtering")
    df_temp = load_data(spark_temp, data_path)
    y_cols_filtered = filter_target_columns(df_temp, y_cols)
    spark_temp.stop()

    train_and_evaluate_model(data_path, y_cols_filtered)


with DAG(
    dag_id=dag_config['dag_id'],
    default_args=default_args,
    schedule_interval=dag_config['schedule_interval'],
    catchup=dag_config['catchup']
) as dag:
    train_task = PythonOperator(
        task_id='train_and_evaluate_models',
        python_callable=run_pyspark_job
    )
