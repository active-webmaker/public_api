from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import glob
from dotenv import dotenv_values
import requests


def data_request(start_date, end_date):
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    API_KEY = config['API_KEY']
    API_URL = config['API_URL']
    LOG_NAME = config['LOG_NAME']
    BASE_DATE = config['BASE_DATE']
    TABLE_LIST = config['TABLE_LIST']
    tables = TABLE_LIST.split(',')

    for table in tables:
        params = {
            'api_key':API_KEY,
            'start_date':start_date,
            'end_date':end_date,
            'table':table
        }

        response = requests.get(API_URL, params=params, timeout=10)
        response_code = response.status_code
        print(response.text)

        if response_code == 200:
            data = response.json()  # JSON 형식인 경우
            print("성공")

        else:
            print("실패")

data_request("20240101", "20240102")

# def data_transform():


# def data_save():



# def data_merge():


"""
# 여러 Parquet 파일 읽기
file_list = glob.glob("data_folder/*.parquet")
df = pd.concat([pd.read_parquet(file) for file in file_list])

# 병합된 파일 저장
df.to_parquet("merged_data.parquet", index=False)
df.to_parquet("test.parquet", engine="pyarrow", compression="gzip")
"""