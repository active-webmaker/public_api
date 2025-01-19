import pandas as pd
import numpy as np
import glob
from dotenv import dotenv_values
import requests
import json
from datetime import datetime
import os
import re
import sys
import sqlite3


def load_json(json_config):
    config = dotenv_values('.env')
    JSON = config[json_config]

    json_dic = {}
    with open(JSON, mode='r', encoding='utf-8') as f:
        json_dic = json.load(f)    
    
    return json_dic



def create_obs_df():
    config = dotenv_values('.env')
    OBSERVATORY = config['OBSERVATORY']
    obs_df = pd.read_csv(OBSERVATORY, encoding='utf-8-sig')
    obs_df = obs_df[obs_df['종료일'].isna()].loc[:,['지점명', '위도', '경도']]
    return obs_df



def data_request(table, start_date, end_date):
    config = dotenv_values('.env')
    API_KEY = config['API_KEY']
    API_URL = config['API_URL']

    params = {
        'api_key':API_KEY,
        'start_date':start_date,
        'end_date':end_date,
        'table':table
    }

    response = requests.get(API_URL, params=params, timeout=10)
    response_code = response.status_code
    result = True
    data = ''

    if response_code == 200:
        data = response.json()  # JSON 형식인 경우

    else:
        result = False

    return (result, data)



def data_transform(data):
    arr = []
    for d in data:
        json_data = d['json_data']
        json_dic = json.loads(json_data)
        json_dic['execute_date'] = d['execute_date']
        arr.append(json_dic)

    df = pd.DataFrame(arr)

    return df



def create_col(df, table_dic):
    tbdate_col_name = table_dic['date_col'][0]
    tbdate_col_format = table_dic['date_col'][1]
    df['newdate'] = pd.to_datetime(df[tbdate_col_name], format=tbdate_col_format)

    essential_col = table_dic['essential_col']
    reference_col = table_dic['reference_col']
    
    if essential_col != reference_col:
        df[essential_col] = None

    return df



def search_dic(data, data_dic):
    dic_keys = data_dic.keys()
    result = [None, None]
    complete = False
    for key in dic_keys:
        if data.find(key) >=0:
            secon_keys = data_dic[key].keys()
            for skey in secon_keys:
                if data.find(skey) >=0:
                    result = [key, skey]
                    complete = True
                    break
        if complete:
            break

    return result



def create_geo_info(other_item, json_dic, geo_json):
    result = []
    for df_item in other_item:
        table = df_item[0]
        df = df_item[1]
        reference_col = json_dic[table]['reference_col']
        df['위도'] = None
        df['경도'] = None
        df.dropna(subset=[reference_col], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    
        for i in range(len(df)):
            ref_data = df.loc[i, reference_col]
            df_i = df.loc[i]
            (key, skey) = search_dic(ref_data, geo_json)
            if key != None:
                df.loc[i, '위도'] = geo_json[key][skey][0]
                df.loc[i, '경도'] = geo_json[key][skey][1]
            else:
                pass
            
        result.append([table, df])
    
    return result



def match_geo_info(serz, key_df, key_col):
    result = None
    
    key_df['위도'] = key_df['위도'].fillna(np.nan)
    key_df['경도'] = key_df['경도'].fillna(np.nan)
    ns = serz['위도']
    ew = serz['경도']

    key_df['cal_ns'] = key_df['위도'] - ns
    key_df['cal_ew'] = key_df['경도'] - ew
    key_df['cal_ns'] = key_df['cal_ns'].abs()
    key_df['cal_ew'] = key_df['cal_ew'].abs()
    key_df['cal_news'] = key_df['cal_ns'] + key_df['cal_ew']

    min_num = min(key_df['cal_news'])
    if type(min_num) == float or type(min_num) == int:
        result_df = key_df[key_df['cal_news']==min_num].reset_index(drop=True).copy()
        result = result_df.loc[0, '지점명']

    return result



def data_match(obs_df, other_item, key_col):
    df_list = []
    ref_df = obs_df.loc[:,['지점명', '위도', '경도']].copy()
    
    for item in other_item:
        ref_df.copy()
        table = item[0]
        df = item[1]
        df.reset_index(drop=True, inplace=True)
        newdf = df.copy()
        for i in range(len(newdf)):
            row = newdf.loc[i].copy()
            match_val = match_geo_info(row, ref_df, key_col)
            newdf.loc[i, key_col] = match_val
        
        df_list.append([table, newdf])
        
    return df_list
            

def save_to_sqlite(table_name, df, sql_db):
  conn = sqlite3.connect(sql_db)  # 데이터베이스 연결
  df.to_sql(table_name, conn, if_exists='append', index=False)  # 데이터 저장
  conn.close()  # 연결 종료



def export_date(db_name, table_name, date_col):
    start_date = None
    end_date = None

    conn = sqlite3.connect(db_name)
    query = f"""
        SELECT * 
        FROM {table_name} 
        ORDER BY {date_col} 
    """
    order_query = query + "DESC LIMIT 1"
    df = pd.read_sql_query(order_query, conn)  # 데이터 추출
    if not df.empty:
        df[date_col] = pd.to_datetime(df[date_col])
        end_date = df.loc[0, 'newdate'].strftime('%Y%m%d%H%M%S')

    order_query = query + "ASC LIMIT 1"
    df = pd.read_sql_query(order_query, conn)  # 데이터 추출
    if not df.empty:
        df[date_col] = pd.to_datetime(df[date_col])
        start_date = df.loc[0, 'newdate'].strftime('%Y%m%d%H%M%S')
    conn.close()

    return (start_date, end_date)



def export_to_parquet(db_name, table_name, date_col, start_date, end_date, DATA_STORAGE):
    conn = sqlite3.connect(db_name)
    query = f"""
        SELECT * 
        FROM {table_name}
    """
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    start_year = start_datetime.year
    end_year = end_datetime.year
    period = end_year - start_year

    for year in range(period):
        start = f"{start_year + year}-01-01 00:00:00"
        end = f"{start_year + year + 1}-01-01 00:00:00"

        new_query = query +f"""
            WHERE {date_col} >= '{start}' AND {date_col} < '{end}'
        """
        df = pd.read_sql_query(new_query, conn)  # 데이터 추출

        fpath = f'{DATA_STORAGE}/{table_name}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fname = f'{fpath}/{table_name}_{start_year + year}.parquet'
        df.to_parquet(fname, engine="pyarrow", compression="gzip")
    
    cursor = conn.cursor()
    start = f"{start_year}-01-01 00:00:00"
    end = f"{end_year}-01-01 00:00:00"

    new_query = f"""
        DELETE FROM {table_name}
        WHERE {date_col} >= '{start}' AND {date_col} < '{end}'
    """
    cursor.execute(new_query)
    conn.commit()
    conn.close()



def file_exist(table_name, base_date):
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    fpath = f'{DATA_STORAGE}/{table_name}_org/*.parquet'
    days_file_list = glob.glob(fpath)
    result = base_date

    if days_file_list:
        days_file_list.sort(reverse=True)
        fname = days_file_list[0]
        ymd = re.search(r'_\d{14}', fname).group()
        ymd = ymd.replace('_', '')
        result = ymd

    return result



def envset():
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    BASE_DATE = config['BASE_DATE']

    MAIN_TABLE = config['MAIN_TABLE']
    TABLE_LIST = config['TABLE_LIST']
    SQLITE_DB = config['SQLITE_DB']
    result = (
        config, DATA_STORAGE, BASE_DATE, 
        MAIN_TABLE, TABLE_LIST, SQLITE_DB
    )
    return result


def main():
    json_dic = load_json('COLUMNS_JSON')
    geo_json = load_json('GEO_JSON')
    obs_df = create_obs_df()

    (config, DATA_STORAGE, BASE_DATE, MAIN_TABLE, TABLE_LIST, SQLITE_DB) = envset()
    key_col = json_dic[MAIN_TABLE]['essential_col']
    tables = TABLE_LIST.split(',')

    start_date = file_exist(MAIN_TABLE, BASE_DATE)
    end_date = datetime.today().strftime('%Y%m%d%H%M%S')
    key_item = ''

    df_list = []
    for table in tables:
        (result, data) = data_request(table, start_date, end_date)
        if result:
            print("Success: 정상적으로 서버에 연결되었습니다.")
            print(f'{table} 테이블로부터, {start_date}부터 {end_date}까지의 데이터를 요청합니다.')
        else:
            print("Fail: 서버에 연결하지 못했습니다.")
            sys.exit()

        if data:
            print("Data exist: 데이터 처리를 시작합니다.\n")
            df = data_transform(data)
            table_dic = json_dic[table]
            fname = f'{DATA_STORAGE}/{table}_org/{table}_{end_date}'
            df.to_parquet(f'{fname}.parquet', engine="pyarrow", compression="gzip")
            if table == MAIN_TABLE:
                pass
            df = create_col(df, table_dic)
            if table == MAIN_TABLE:
                key_item = [table, df]
            else:
                df_list.append([table, df])
        
        else:
            print("Data Not exist: 처리할 데이터가 없습니다.")
            sys.exit()
    
    key_item[1] = pd.merge(key_item[1], obs_df, left_on=key_col, right_on='지점명', how='left')
    obs_df = key_item[1].loc[:,[key_col, '지점명', '위도', '경도']].drop_duplicates(subset=[key_col,])
    obs_df = obs_df.loc[:,['지점명', '위도', '경도']]
    df_list = create_geo_info(df_list, json_dic, geo_json)
    df_list = data_match(obs_df, df_list, key_col)

    table = key_item[0]
    fname = f'{DATA_STORAGE}/{table}/{table}_{end_date}'
    key_item[1].to_parquet(f'{fname}.parquet', engine="pyarrow", compression="gzip")

    cattle = ''
    for item in df_list:
        table = item[0]
        df = item[1]
        fname = f'{DATA_STORAGE}/{table}/{table}_{end_date}'
        df.to_parquet(f'{fname}.parquet', engine="pyarrow", compression="gzip")
        if table == 'cattle':
            cattle = df.copy()

    cattle = cattle.loc[:, ['newdate', 'stnNm', 'LKNTS_NM', 'OCCRRNC_LVSTCKCNT']]
    cattle = cattle[cattle['LKNTS_NM']!='nodata']
    cattle.dropna(subset=["LKNTS_NM", "OCCRRNC_LVSTCKCNT"], inplace=True)
    cattle['OCCRRNC_LVSTCKCNT'] = cattle['OCCRRNC_LVSTCKCNT'].astype(int)
    cattle = cattle.pivot_table(
        index=['newdate', 'stnNm'],
        columns='LKNTS_NM',
        values='OCCRRNC_LVSTCKCNT',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    merge_df = key_item[1]

    # 날짜 형식 통일
    cattle.loc[:, 'newdate'] = pd.to_datetime(cattle['newdate'])
    merge_df.loc[:, 'newdate'] = pd.to_datetime(merge_df['newdate'])

    # 공백 제거
    cattle.loc[:,'stnNm'] = cattle['stnNm'].str.strip()
    merge_df.loc[:,'stnNm'] = merge_df['stnNm'].str.strip()


    # 병합 키 조합 비교
    cattle_keys = set(cattle[['newdate', 'stnNm']].itertuples(index=False, name=None))
    merge_df_keys = set(merge_df[['newdate', 'stnNm']].itertuples(index=False, name=None))
    missing_keys = cattle_keys - merge_df_keys
    if missing_keys:
        print(f'cattle의 키 조합 수: {len(cattle_keys)}')
        print(f'merge_df의 키 조합 수: {len(merge_df_keys)}')
        print(f'merge_df에 없는 키: {len(missing_keys)}')

    try:
        merge_df = pd.merge(merge_df, cattle, on=['newdate', 'stnNm'], how='left')

    except Exception as e:
        print(f'Error: {e}')
        print(f'merge_df.columns:\n{merge_df.columns}')
        print(f'cattle_df.columns:\n{cattle.columns}')
        sys.exit()
    
    
    if merge_df.empty:
        print("merge_df is empty: 병합된 데이터프레임에 데이터가 없습니다.")
        sys.exit()

    else:
        print('merge_df is exist: 데이터프레임에 데이터가 존재합니다.')

    save_to_sqlite('total_df', merge_df, SQLITE_DB)

    if os.path.exists(SQLITE_DB):
        (start_date, end_date) = export_date(SQLITE_DB, 'total_df', 'newdate')
        export_to_parquet(SQLITE_DB, 'total_df', 'newdate', start_date, end_date, DATA_STORAGE)


if __name__ == "__main__":
    main()
