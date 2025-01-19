import os
import sys
import requests
import json
from datetime import datetime
from dotenv import dotenv_values
from pyspark.sql import Window
import pyspark.sql.functions as F

# PySpark 관련 import
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, udf, to_date, to_timestamp, when, abs as spark_abs, sum as spark_sum
)
from pyspark.sql.types import StringType, FloatType, IntegerType, StructType, StructField

def load_json_from_env(json_config_key: str):
    """
    .env 파일에서 json_config_key 를 읽어,
    해당 경로의 JSON 파일을 로드하여 dict 형태로 반환
    """
    config = dotenv_values('.env')
    json_path = config[json_config_key]
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_env():
    """
    .env에서 필요한 설정값들을 읽어오기
    """
    config = dotenv_values('.env')
    return {
        'API_URL': config.get('API_URL', ''),
        'API_KEY': config.get('API_KEY', ''),
        'DATA_STORAGE': config.get('DATA_STORAGE', ''),
        'BASE_DATE': config.get('BASE_DATE', ''),
        'MAIN_TABLE': config.get('MAIN_TABLE', ''),
        'TABLE_LIST': config.get('TABLE_LIST', ''),
        'SQLITE_DB': config.get('SQLITE_DB', ''),
        'OBSERVATORY': config.get('OBSERVATORY', '')
    }

def get_obs_df(spark, obs_csv_path):
    """
    관측지점 CSV 불러와서, 종료일이 null 인 행만 필터링 + (지점명, 위도, 경도) 컬럼 선택
    """
    # CSV 스키마는 상황에 맞게 지정하거나, inferSchema=True 사용
    df = spark.read.option("header", True).option("encoding", "UTF-8-SIG").csv(obs_csv_path, inferSchema=True)
    # 종료일이 null인 데이터만 필터링
    df = df.filter(col("종료일").isNull()).select("지점명", "위도", "경도")
    return df

def request_api_data(table, start_date, end_date, api_url, api_key):
    """
    API 호출 (requests 패키지 사용).
    Spark가 직접 HTTP 통신을 하려면 별도 라이브러리가 필요하므로,
    여기서는 일반 Python requests로 JSON을 받아온 뒤 Python list/dict로 처리
    """
    params = {
        'api_key': api_key,
        'start_date': start_date,
        'end_date': end_date,
        'table': table
    }
    try:
        r = requests.get(api_url, params=params, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        else:
            return False, []
    except Exception as e:
        print(f"API request error: {e}")
        return False, []

def transform_with_spark(spark, raw_list):
    """
    raw_list(= 파이썬 list[dict]) 내 "json_data"를 다시 json.loads 하여 Spark DF로 변환
    """
    if not raw_list:
        return None
    
    # 각 원소: {"json_data": "...", "execute_date": "..."}
    # => "json_data" 안에 실제 컬럼이 들어있으므로 파싱 후 Spark DF 생성
    parsed = []
    for row in raw_list:
        json_data_str = row.get("json_data", "{}")
        parsed_json = json.loads(json_data_str)
        parsed_json["execute_date"] = row.get("execute_date", "")
        parsed.append(parsed_json)
    
    # Spark DF로 변환
    df = spark.createDataFrame(parsed)  # RDD -> DataFrame
    return df

def create_col_spark(df, table_conf):
    """
    create_col 함수:
      1) date_col 포맷 맞춰서 newdate 생성
      2) essential_col != reference_col 이면, essential_col에 None을 채움
    """
    tbdate_col_name, tbdate_col_format = table_conf["date_col"]
    essential_col = table_conf["essential_col"]
    reference_col = table_conf["reference_col"]
    
    # 날짜 포맷 변환: to_timestamp() 사용
    # (예) '%Y%m%d' -> to_timestamp(col(tbdate_col_name), 'yyyyMMdd')
    df = df.withColumn(
        "newdate",
        to_timestamp(col(tbdate_col_name), tbdate_col_format.replace('%Y','yyyy').replace('%m','MM').replace('%d','dd'))
    )
    
    if essential_col != reference_col:
        # Pandas에서는 df[essential_col] = None
        # Spark에서는 literal(None).cast(StringType()) 등으로 추가
        df = df.withColumn(essential_col, lit(None).cast(StringType()))
    
    return df

def create_geo_info_spark(spark, df, reference_col, geo_json):
    """
    예시) FARM_LOCPLC 텍스트(= ref_data)에 특정 key가 있으면 위경도 매핑
    원본 create_geo_info()는 여러 DataFrame을 리스트로 처리했지만,
    예시에서는 단일 DF로 단순화(실제 사용 시 join/UDF로 확장).
    """
    # UDF: ref_data 문자열 내에서 geo_json 키 검색 → 일치 시 위도/경도 반환
    # geo_json 구조: geo_json["강원"]["강릉"] = [37.75..., 128.87...]
    def find_lat_lng(ref_data):
        if not ref_data:
            return (None, None)
        for wide_key, wide_val in geo_json.items():
            if wide_key in ref_data:
                for city_key, coords in wide_val.items():
                    if city_key in ref_data:
                        # coords = [lat, lng]
                        return (coords[0], coords[1])
        return (None, None)
    
    udf_find_lat_lng = udf(find_lat_lng, returnType=StructType([
        StructField("lat", FloatType()),
        StructField("lng", FloatType())
    ]))
    
    # DF에 위도/경도 컬럼 추가
    df = df.withColumn("lat_lng", udf_find_lat_lng(col(reference_col)))
    df = df.withColumn("위도", col("lat_lng.lat")).withColumn("경도", col("lat_lng.lng")).drop("lat_lng")
    return df

def match_geo_info_spark(spark, df, obs_df, key_col):
    """
    원본 data_match()와 유사: 위경도 근접 매칭
    - obs_df(지점명, 위도, 경도)
    - df(newdate, stnNm, 위도, 경도) 등에 대해, 거리(절댓값 합)가 가장 작은 지점명을 찾아서 key_col에 할당
    ※ 대용량에선 브로드캐스트 조인, spatial join 등을 고려.
    여기서는 단순 '카트esian 조인 + 최소값' 예시.
    """
    # obs_df를 DF로 두고, df와 crossJoin 후, NS+EW 차이의 절댓값 합이 최소인 obs 지점명 찾기
    
    # obs_df를 temp 뷰로 등록
    obs_df.createOrReplaceTempView("obs_table")
    df.createOrReplaceTempView("main_table")
    
    joined = spark.sql("""
    SELECT 
        m.*,
        o.지점명 AS obs_nm,
        ABS(o.위도 - m.위도) AS cal_ns,
        ABS(o.경도 - m.경도) AS cal_ew
    FROM main_table m
    CROSS JOIN obs_table o
    """)
    
    # cal_news = cal_ns + cal_ew
    joined = joined.withColumn("cal_news", col("cal_ns") + col("cal_ew"))
    
    w = Window.partitionBy("newdate", "stnNm").orderBy("cal_news")
    joined = joined.withColumn("rn", F.row_number().over(w))  # 정렬 후 순번
    best_match = joined.filter(col("rn") == 1).drop("rn")
    
    # best_match DF에서 key_col 업데이트: (예) stnNm -> best obs_nm
    best_match = best_match.withColumn(key_col, col("obs_nm"))
    
    return best_match

def main():
    spark = SparkSession.builder \
        .appName("SparkETLJob") \
        .getOrCreate()
    
    # 1. .env 로드
    env_conf = load_env()
    API_URL = env_conf['API_URL']
    API_KEY = env_conf['API_KEY']
    DATA_STORAGE = env_conf['DATA_STORAGE']
    BASE_DATE = env_conf['BASE_DATE']
    MAIN_TABLE = env_conf['MAIN_TABLE']
    TABLE_LIST = env_conf['TABLE_LIST'].split(',')
    OBSERVATORY = env_conf['OBSERVATORY']

    # 2. columns.json / geo.json 로드
    columns_json = load_json_from_env('COLUMNS_JSON')
    geo_json = load_json_from_env('GEO_JSON')

    # 3. 관측 지점 DF 생성
    obs_df = get_obs_df(spark, OBSERVATORY)
    obs_df.cache()

    # 4. BASE_DATE ~ 오늘 사이의 데이터 API 요청
    start_date = BASE_DATE
    end_date = datetime.today().strftime('%Y%m%d%H%M%S')

    # 5. 메인 테이블(MAIN_TABLE) / 기타 테이블들 처리
    main_table_df = None
    other_tables_df_list = []
    
    for table in TABLE_LIST:
        success, data_list = request_api_data(
            table, start_date, end_date,
            api_url=API_URL, api_key=API_KEY
        )
        if not success:
            print(f"Fail to get data from API for table={table}")
            sys.exit(1)
        
        if data_list:
            # data_transform 단계
            tmp_df = transform_with_spark(spark, data_list)
            if tmp_df is None:
                print(f"No valid data to transform for table={table}")
                continue
            
            # create_col 단계
            table_conf = columns_json[table]
            tmp_df = create_col_spark(tmp_df, table_conf)

            # parquet 임시저장(원본 _org)
            org_path = os.path.join(DATA_STORAGE, f"{table}_org")
            os.makedirs(org_path, exist_ok=True)
            tmp_df.write.mode("append").parquet(
                os.path.join(org_path, f"{table}_{end_date}.parquet"),
                compression="gzip"
            )

            if table == MAIN_TABLE:
                main_table_df = tmp_df
            else:
                other_tables_df_list.append((table, tmp_df))
    
    if main_table_df is None:
        print("No main_table data found. Exiting...")
        sys.exit(1)

    # 6. 메인 테이블(MAIN_TABLE)에 관측지점 obs_df를 merge
    #    => key_col 기준: columns.json[MAIN_TABLE]["essential_col"]
    key_col = columns_json[MAIN_TABLE]["essential_col"]  # ex) "stnNm"

    # Spark에서 메인 테이블 + obs_df Join
    main_table_df.createOrReplaceTempView("main_t")
    obs_df.createOrReplaceTempView("obs_t")

    joined_main = spark.sql(f"""
    SELECT 
        m.*,
        o.지점명 as obs_nm,
        o.위도 as obs_lat,
        o.경도 as obs_lng
    FROM main_t m
    LEFT JOIN obs_t o
      ON m.{key_col} = o.지점명
    """)

    # obs_df 갱신 (중복 제거)
    new_obs_df = joined_main.select("obs_nm", "obs_lat", "obs_lng").dropDuplicates()

    # 7. 기타 테이블들에 대해 create_geo_info + data_match
    final_list = []
    for (tb, df_) in other_tables_df_list:
        # create_geo_info
        df_ = create_geo_info_spark(spark, df_, columns_json[tb]["reference_col"], geo_json)
        # data_match
        matched_df = match_geo_info_spark(spark, df_, new_obs_df.withColumnRenamed("obs_nm","지점명").withColumnRenamed("obs_lat","위도").withColumnRenamed("obs_lng","경도"), key_col)
        final_list.append((tb, matched_df))

    # 8. 최종 메인 DF, 기타 DF들 Parquet 저장
    main_path = os.path.join(DATA_STORAGE, MAIN_TABLE)
    os.makedirs(main_path, exist_ok=True)
    joined_main.write.mode("append").parquet(
        os.path.join(main_path, f"{MAIN_TABLE}_{end_date}.parquet"),
        compression="gzip"
    )

    # cattle 등 기타 테이블 처리
    cattle_df = None
    for (tb, df_) in final_list:
        out_path = os.path.join(DATA_STORAGE, tb)
        os.makedirs(out_path, exist_ok=True)
        df_.write.mode("append").parquet(
            os.path.join(out_path, f"{tb}_{end_date}.parquet"),
            compression="gzip"
        )
        if tb == "cattle":
            cattle_df = df_

    if cattle_df is not None:
        # cattle 피벗: (newdate, stnNm, LKNTS_NM, OCCRRNC_LVSTCKCNT) → sum
        cattle_df = cattle_df.filter(col("LKNTS_NM") != "nodata").na.drop(subset=["LKNTS_NM","OCCRRNC_LVSTCKCNT"])
        cattle_df = cattle_df.withColumn("OCCRRNC_LVSTCKCNT", col("OCCRRNC_LVSTCKCNT").cast(IntegerType()))

        pivoted = (
            cattle_df
            .groupBy("newdate", "stnNm")
            .pivot("LKNTS_NM")
            .agg(spark_sum("OCCRRNC_LVSTCKCNT"))
            .fillna(0)
        )

        # 메인 DF(joined_main)와 merge
        # 시계열 & stnNm 기준
        pivoted.createOrReplaceTempView("cattle_pivot")
        joined_main.createOrReplaceTempView("joined_main")

        merged = spark.sql("""
        SELECT jm.*, cp.*
        FROM joined_main jm
        LEFT JOIN cattle_pivot cp
          ON jm.newdate = cp.newdate
         AND jm.stnNm   = cp.stnNm
        """)

        if merged.count() == 0:
            print("merge_df is empty.")
            sys.exit(1)
        else:
            print("merge_df is exist.")

        # 필요 시 파켓 저장
        merged_path = os.path.join(DATA_STORAGE, "merged_final")
        os.makedirs(merged_path, exist_ok=True)
        merged.write.mode("append").parquet(
            os.path.join(merged_path, f"merged_{end_date}.parquet"),
            compression="gzip"
        )

    # 종료
    spark.stop()
    print("Spark ETL Job is done.")


if __name__ == "__main__":
    main()
