# PySpark 관련 라이브러리
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, explode, sequence, lit, expr
from pyspark.sql.types import FloatType, IntegerType

# XGBoost4J-Spark
from sparkxgb import XGBoostClassifier

# 기타
import os
import numpy as np
from airflow.models import Variable

# 설정 파일 로드
from config import x_cols, y_cols, output_path, spark_app_name, xgb_params

# Spark ML 및 유틸리티 함수
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def create_spark_session(app_name):
    """Spark 세션 생성"""
    spark = (
        SparkSession.builder
        .appName(app_name)
        # .master(spark_master)  # 필요 시 설정
        .getOrCreate()
    )
    return spark


def load_data(spark, data_path):
    """데이터 로드"""
    df = spark.read.parquet(data_path)
    return df


def filter_target_columns(df, y_cols, min_count=300):
    """300 이상인 타겟 컬럼만 필터링"""
    sums_exprs = [f"sum({yc}) as {yc}" for yc in y_cols]
    sums_df = df.selectExpr(*sums_exprs).collect()[0].asDict()
    y_cols_filtered = [k for k, v in sums_df.items() if v is not None and v > min_count]
    return y_cols_filtered


def preprocess_features(df, x_cols):
    """독립변수 전처리"""
    df_pre = df
    for xc in x_cols:
        df_pre = df_pre.filter(col(xc).isNotNull())
        df_pre = df_pre.withColumn(xc, col(xc).cast(FloatType()))
        q1, q3 = df_pre.approxQuantile(xc, [0.25, 0.75], 0.0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_pre = df_pre.filter((col(xc) >= lower_bound) & (col(xc) <= upper_bound))

    return df_pre


def prepare_data_for_modeling(df, x_cols, y_col):
    """모델링을 위한 데이터 준비"""
    target_cols = x_cols + [y_col]
    df_y = df.select(*target_cols)
    df_y = df_y.na.fill(value=0, subset=[y_col])
    df_y = df_y.withColumn(y_col, col(y_col).cast(IntegerType()))
    df_y = df_y.withColumn("cnt", when(col(y_col) == 0, lit(1)).otherwise(col(y_col)))
    df_y = df_y.withColumn("rep_seq", sequence(lit(1), col("cnt")))
    df_y = df_y.select(*target_cols, "cnt", explode(col("rep_seq")).alias("rep_item"))
    df_y = df_y.withColumn(y_col, when(col(y_col) > 0, lit(1)).otherwise(col(y_col)))
    df_y = df_y.drop("cnt", "rep_seq", "rep_item")
    df_y = df_y.dropna()
    return df_y


def undersample_data(df, y_col):
    """언더샘플링"""
    count_0 = df.filter(col(y_col) == 0).count()
    count_1 = df.filter(col(y_col) == 1).count()

    if count_0 == 0 or count_1 == 0:
        print(f"[{y_col}] 클래스 불균형이 극단적이어서 스킵합니다.")
        return None

    minority_class = 0 if count_0 < count_1 else 1
    minority_count = min(count_0, count_1)

    df_min = df.filter(col(y_col) == minority_class)
    df_maj = df.filter(col(y_col) != minority_class)

    sample_ratio = minority_count / (count_1 if minority_class == 0 else count_0)
    df_maj_sample = df_maj.sample(False, sample_ratio, seed=42)
    df_resampled = df_min.union(df_maj_sample)
    return df_resampled


def evaluate_model(pred_df, y_col):
    """모델 평가"""
    evaluator_auc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol=y_col,
        metricName="areaUnderROC"
    )
    roc_auc = evaluator_auc.evaluate(pred_df)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=y_col,
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(pred_df)

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=y_col,
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = evaluator_f1.evaluate(pred_df)

    report_str = (
        f"[{y_col}] ROC-AUC: {roc_auc}\n"
        f"[{y_col}] Accuracy: {accuracy}\n"
        f"[{y_col}] F1-Score: {f1_score}\n"
    )

    print(report_str)
    return report_str


def train_and_evaluate_model(data_path, y_cols_filtered):
    """모델 훈련 및 평가"""
    # Spark 세션 생성
    spark = create_spark_session(spark_app_name)

    # 데이터 로드
    df = load_data(spark, data_path)
    df = df.select(*x_cols, *y_cols) #사용할 컬럼만 선택

    # 독립변수 전처리
    df_pre = preprocess_features(df, x_cols)
    df_pre.persist() # 캐싱

    # 스케일링을 위한 파이프라인 생성 및 학습
    assembler = VectorAssembler(inputCols=x_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=True, withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    scaler_model = pipeline.fit(df_pre)

    # 각 타겟 컬럼별 모델 훈련 및 평가
    for yc in y_cols_filtered:
        # 데이터 준비
        df_y = prepare_data_for_modeling(df_pre, x_cols, yc)

        # 언더샘플링
        df_resampled = undersample_data(df_y, yc)
        if df_resampled is None:
            continue

        # Train/Test 분할
        train_df, test_df = df_resampled.randomSplit([0.8, 0.2], seed=42)

        # 스케일링 적용
        train_vec = scaler_model.transform(train_df)
        test_vec = scaler_model.transform(test_df)

        # XGBoost 모델 훈련
        xgb = XGBoostClassifier(
            featuresCol="features_scaled",
            labelCol=yc,
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            **xgb_params
        )
        xgb_model = xgb.fit(train_vec)

        # 예측
        pred_df = xgb_model.transform(test_vec)

        # 평가
        report_str = evaluate_model(pred_df, yc)

        # 모델 및 리포트 저장
        model_path = os.path.join(output_path, f"{yc}_xgboost_model")
        xgb_model.save(model_path)

        with open(os.path.join(output_path, f"{yc}_model_report.txt"), 'w') as f:
            f.write(report_str)

    # Spark 세션 종료
    spark.stop()

if __name__ == "__main__":
    # Airflow에서 Variable로 설정한 값 가져오기
    DATA_PATH = Variable.get("DATA_PATH")

    spark_temp = create_spark_session("temp_spark_for_filtering")
    df_temp = load_data(spark_temp, DATA_PATH)
    y_cols_filtered = filter_target_columns(df_temp, y_cols)
    spark_temp.stop()
    
    train_and_evaluate_model(DATA_PATH, y_cols_filtered)
