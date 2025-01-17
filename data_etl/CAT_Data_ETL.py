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
    # print("obs_df(0:10):\n", obs_df.loc[0:10])
    # print("obs_df_len:", len(obs_df))
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
    # print(len(response.text))
    result = True
    data = ''

    if response_code == 200:
        # print("Success")
        data = response.json()  # JSON 형식인 경우

    else:
        # print("Fail")
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
    # print(df.loc[0])
    # print(df.loc[10])

    return df



def create_col(df, table_dic):
    tbdate_col_name = table_dic['date_col'][0]
    tbdate_col_format = table_dic['date_col'][1]
    df['newdate'] = pd.to_datetime(df[tbdate_col_name], format=tbdate_col_format)

    essential_col = table_dic['essential_col']
    reference_col = table_dic['reference_col']
    
    if essential_col != reference_col:
        df[essential_col] = None

    if 'newdate' in  df.columns:
        print('newdate_exist')
    else:
        print('newdate_not_exist')

    return df



def search_dic(data, data_dic):
    dic_keys = data_dic.keys()
    result = [None, None]
    complete = False
    for key in dic_keys:
        # print(f'key:{key}')
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
        # print(f'table: {table}')
        # print(f'reference_col: {reference_col}')
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
                # print(f'df_i[FARM_NM]: {df_i['FARM_NM']}')
                # print(f'df_i[LKNTS_NM]: {df_i['LKNTS_NM']}')
                # print(f'df_i[FARM_LOCPLC]: {df_i['FARM_LOCPLC']}')
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
    print(f'obs_df_type: {type(obs_df)}')
    print(f'obs_df_head:\n {obs_df.head()}')
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
        
    print(f'data_match df_list[0][1]_col_head:\n {df_list[0][1].loc[:,[key_col, '위도', '경도']].head()}')
    return df_list
            


def data_split_save(table_name, df, json_dic, merge_bool=False):
    if merge_bool:
        date_col = 'newdate'
    else:
        tbdate_col_data = json_dic['date_col'][0]
        tbdate_col_format = json_dic['date_col'][1]
        df['tbdate'] = pd.to_datetime(tbdate_col_data, format=tbdate_col_format)
        date_col = 'tbdate'

    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=[date_col], inplace=True)
    start_date = df.loc[0, date_col]
    end_date = df.loc[len(df)-1, date_col]
    
    df_list = []
    group_date = start_date
    group_num = 0
    for i in range(len(df)):
        split_date = df.loc[i, date_col]

        if split_date == group_date:
            if (i == len(df) - 1):
                new_df = df.iloc[group_num:i+1].copy()
                df_list.append([new_df, group_date])

        else:
            new_df = df.iloc[group_num:i].copy()
            df_list.append([new_df, group_date])
            group_date = split_date
            group_num = i

            if (i == len(df) - 1):
                new_df = df.iloc[i:i+1].copy()
                df_list.append([new_df, group_date])
            

    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    today = datetime.today()
    today_year = today.year
    today_month = today.month

    if merge_bool:
        fpath = f'{DATA_STORAGE}/{table_name}/days'
    else:
        fpath = f'{DATA_STORAGE}/{table_name}/{today_year}/{today_month}'

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fname = f'{fpath}/{table_name}'

    for df_item in df_list:
        df = df_item[0]
        df_date = df_item[1]
        table_date = df_date.strftime('%Y%m%d')
        new_fname = f'{fname}_{table_date}.parquet'
        if os.path.exists(new_fname):
            old_df = pd.read_parquet(new_fname)
            df = pd.concat([old_df, df])
        df.to_parquet(new_fname, engine="pyarrow", compression="gzip")



def file_merge(table_name, load_period, load_format, sav_period, sav_num):
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    days_file_list = glob.glob(f'{DATA_STORAGE}/{table_name}/{load_period}/*.parquet').sort()

    if days_file_list:
        start_file = days_file_list[0]
        ymd = re.search(load_format, start_file).group()
        ymd = ymd.replace('_', '')[0:sav_num]

        def file_sav(fname_list, sav_date):
            df_list = []
            for file_name in fname_list:
                df = pd.read_parquet(file_name)
                df_list.append(df)
            
            df = pd.concat(df_list)
            fpath = f'{DATA_STORAGE}/{table_name}/{sav_period}/{table_name}_{sav_date}.parquet'
            df.to_parquet(fpath, engine="pyarrow", compression="gzip")

            for file_name in fname_list:
                os.remove(file_name)


        fname_list = []
        last_file = days_file_list[len(days_file_list)-1]
        for file_name in days_file_list:
            if file_name.find(ymd) >= 0:
                fname_list.append(file_name)
                if file_name == last_file and len(fname_list) > 0:
                    file_sav(fname_list, ymd)
            
            elif fname_list:
                file_sav(fname_list, ymd)
                fname_list = []
                fname_list.append(file_name)
                ymd = re.search(load_format, file_name).group()
                ymd = ymd.replace('_', '')[0:sav_num]




def file_exist(table_name, base_date):
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    fpath = f'{DATA_STORAGE}/{table_name}_org/*.parquet'
    print(f'fpath: {fpath}')
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
    return (config, DATA_STORAGE, BASE_DATE, MAIN_TABLE, TABLE_LIST)


def main():
    json_dic = load_json('COLUMNS_JSON')
    geo_json = load_json('GEO_JSON')
    obs_df = create_obs_df()
    print(f'no.2 obs_df_type: {type(obs_df)}')

    (config, DATA_STORAGE, BASE_DATE, MAIN_TABLE, TABLE_LIST) = envset()
    key_col = json_dic[MAIN_TABLE]['essential_col']
    tables = TABLE_LIST.split(',')

    # start_date = file_exist(MAIN_TABLE, BASE_DATE)
    start_date = BASE_DATE
    end_date = datetime.today().strftime('%Y%m%d%H%M%S')
    key_item = ''

    df_list = []
    for table in tables:
        (result, data) = data_request(table, start_date, end_date)
        if result:
            print("Success: 정상적으로 서버에 연결되었습니다.")
            print((table, start_date, end_date))
        else:
            print("Fail: 서버에 연결하지 못했습니다.")
            sys.exit()

        if data:
            print("Data exist: 데이터 처리를 시작합니다.")
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
    df_list = create_geo_info(df_list, json_dic, geo_json)
    df_list = data_match(obs_df, df_list, key_col)
    print(f'no.1 cattle.head:\n {df_list[0][1].head()}')

    table = key_item[0]
    fname = f'{DATA_STORAGE}/{table}/{table}_{end_date}'
    key_item[1].to_parquet(f'{fname}.parquet', engine="pyarrow", compression="gzip")
    print(f'key_item[1].colomns:\n {key_item[1].columns}')

    cattle = ''
    for item in df_list:
        table = item[0]
        df = item[1]
        fname = f'{DATA_STORAGE}/{table}/{table}_{end_date}'
        df.to_parquet(f'{fname}.parquet', engine="pyarrow", compression="gzip")
        if table == 'cattle':
            cattle = df.copy()

    # cattle = cattle.loc[:, ['newdate', 'stnNm', 'LKNTS_NM', '위도', '경도']]
    cattle = cattle.loc[:, ['newdate', 'stnNm', 'LKNTS_NM', 'OCCRRNC_LVSTCKCNT']]
    cattle = cattle[cattle['LKNTS_NM']!='nodata']
    cattle.dropna(subset=["LKNTS_NM", "OCCRRNC_LVSTCKCNT"], inplace=True)
    cattle['OCCRRNC_LVSTCKCNT'] = cattle['OCCRRNC_LVSTCKCNT'].astype(int)
    # print(f'no.2 cattle.head:\n {cattle[cattle['LKNTS_NM']!='nodata'].head()}')
    # print(f'cattle-date-20240119:\n {cattle[cattle['newdate']==pd.to_datetime("20240119", format="%Y%m%d")]}')
    cattle = cattle.pivot_table(
        index=['newdate', 'stnNm'],
        columns='LKNTS_NM',
        values='OCCRRNC_LVSTCKCNT',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    print(f'cattle-date-20240119:\n {cattle[cattle['newdate']==pd.to_datetime("20240119", format="%Y%m%d")]}')
    # print(f'no.3 cattle.head:\n {cattle.head()}')
    # print(f'no.4 cattle.head:\n {cattle[cattle['결핵병'] > 0].head()}')
    merge_df = key_item[1]

    # 날짜 형식 통일
    cattle.loc[:, 'newdate'] = pd.to_datetime(cattle['newdate'])
    merge_df.loc[:, 'newdate'] = pd.to_datetime(merge_df['newdate'])

    # 공백 제거
    cattle.loc[:,'stnNm'] = cattle['stnNm'].str.strip()
    merge_df.loc[:,'stnNm'] = merge_df['stnNm'].str.strip()


    # 병합 키 조합 비교
    # cattle_keys = set(cattle[['newdate', 'stnNm']].itertuples(index=False, name=None))
    # merge_df_keys = set(merge_df[['newdate', 'stnNm']].itertuples(index=False, name=None))
    # missing_keys = cattle_keys - merge_df_keys
    # print(f'cattle의 키 조합 수: {len(cattle_keys)}')
    # print(f'merge_df의 키 조합 수: {len(merge_df_keys)}')
    # print(f'merge_df에 없는 키: {missing_keys}')

    # print(f'cattle.colomns:\n {cattle.columns}')
    # print(f'merge_df.colomns:\n {merge_df.columns}')
    # print(f'merge_df.head:\n {merge_df.loc[:10, :]}')
    # print(f'cattle.head:\n {cattle.loc[:10, :]}')
    # print(f'cattle:\n {cattle.loc[(cattle['브루셀라병'] > 0) | (cattle['결핵병'] > 0), ['결핵병', '브루셀라병']]}')
    # try:
    #     merge_df = pd.merge(merge_df, cattle, on=['newdate', 'stnNm'], how='left')

    # except Exception as e:
    #     print(f'Error: {e}')
    #     print(f'merge_df.columns:\n{merge_df.columns}')
    #     print(f'cattle_df.columns:\n{cattle.columns}')
    #     sys.exit()
    
    # if merge_df.empty:
    #     print("merge_df is empty: 병합된 데이터프레임에 데이터가 없습니다.")
    #     sys.exit()

    # print(f'merge_df.head:\n {merge_df.loc[(merge_df["결핵병"] > 0), ['결핵병']].head()}')
    # print(f'merge_df.head:\n {merge_df.loc[(merge_df["브루셀라병"] > 0), ['브루셀라병']].head()}')
    # data_split_save("total_df", merge_df, json_dic, merge_bool=True)
    # file_merge("total_df", 'days', r'_\d{8}', 'months', 6)
    # file_merge("total_df", 'months', r'_\d{6}', 'years', 4)


if __name__ == "__main__":
    main()
