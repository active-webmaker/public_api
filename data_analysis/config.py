from datetime import datetime

# Airflow DAG 설정
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),  # 실제 시작일로 변경
    'retries': 1
}

dag_config = {
    'dag_id': 'pyspark_xgboost_example',
    'schedule_interval': None,
    'catchup': False
}

# PySpark 공통 설정
spark_app_name = "Pyspark_XGBoost_Example"

data_path = "/path/to/your/data.parquet"

# 모델 저장 경로
output_path = "/tmp/xgboost_models"  # 실제 경로로 변경

# 분석에 사용할 컬럼
x_cols = ['avgTa', 'minTa', 'maxTa', 'avgRhm']
y_cols = [
    '가금티푸스', '결핵병', '고병원성조류인플루엔자', '낭충봉아부패병',
    '돼지생식기호흡기증후군', '브루셀라병', '사슴만성소모성질병',
    '아프리카돼지열병', '추백리'
]

# XGBoost 모델 설정
xgb_params = {
    "evalMetric": "auc",
    "numRound": 50,
    "maxDepth": 5,
    "seed": 42
}