import requests
from datetime import datetime, timedelta
import pymysql
import time
from dotenv import dotenv_values
import json
import sys


def log_recode(log_txt, log_var, extra_var=None):
    print(log_var)
    if extra_var:
        log_var += ', ' + str(extra_var)
    log_txt += log_var + '\n'
    return log_txt

# 현재 시간을 토대로 '키워드_날짜_시간' 형태의 파일 이름을 만드는 함수
def filename_maker(filename='output', file_form='txt'):
    print('파일 이름 생성 함수 호출')
    today = datetime.now()
    today_Y = "%04d" % today.year
    today_M = "%02d" % today.month
    today_D = "%02d" % today.day
    today_h = "%02d" % today.hour
    today_m = "%02d" % today.minute
    today_s = "%02d" % today.second
    filename = f'{filename}_{today_Y}_{today_M}_{today_D}_{today_h}_{today_m}_{today_s}.{file_form}'
    return filename


log_txt = log_recode('', 'PROGRAM START\n')

config = dotenv_values('.env')
USERID = config['USERID']
PW = config['PW']
DBNAME = config['DBNAME']
API_KEY = config['API_KEY']
API_URL = config['API_URL']
API_SECTION = config['API_SECTION']
DATA_STORAGE = config['DATA_STORAGE']
fname = filename_maker(f'{DATA_STORAGE}/log_txt')
log_txt = log_recode(log_txt, 'add dotenv_values')


today_date = datetime.now()
today_date_str = today_date.strftime("%Y%m%d")
upload_date = today_date - timedelta(days=14)
log_txt = log_recode(log_txt, f'today_date: {today_date}, upload_date: {upload_date}')

request_date = datetime.strptime('20240101', "%Y%m%d")
log_txt = log_recode(log_txt, f'base_request_date: {request_date}')

# MySQL 연결 설정
conn = pymysql.connect(host='localhost', user=USERID, password=PW, db=DBNAME, charset='utf8', cursorclass=pymysql.cursors.DictCursor)

try:
    # 커서 생성
    curs = conn.cursor()

    # SQL 쿼리 실행
    sql = "SELECT * FROM cattle ORDER BY JSON_EXTRACT(json_data, '$.OCCRRNC_DE') DESC LIMIT 1"
    curs.execute(sql)
    row = curs.fetchone() # 하나의 결과만 가져오기

    if row != None:
        json_data = row['json_data']
        json_data = json.loads(json_data)
        request_date = json_data['OCCRRNC_DE']
        log_txt = log_recode(log_txt, f'last_request_date: {request_date}')
        request_date = datetime.strptime(str(request_date), "%Y%m%d") + timedelta(days=1)
        log_txt = log_recode(log_txt, f'update_request_date: {request_date}')

    else:
        log_txt = log_recode(log_txt, f'MySQL {DBNAME} cattle empty')

    conn.close()
    

except Exception as e:
    conn.close()

    log_txt = log_recode(log_txt, f"Error: {e}")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(log_txt)

    sys.exit()



TYPE = "json"
max_count = 1000
all_data = []
item_num = 1000
result_code = "INFO-000"

while request_date <= upload_date:
    totalCnt = 2000
    page = 1
    end_index = max_count
    wnum = 1

    while totalCnt > end_index:
        try:
            if page >= 10 or wnum >= 10:
                log_txt = log_recode(
                    log_txt, 
                    f'page: {page}, wnum: {wnum},'
                    + 'page >= 10 or wnum >= 10, break'
                )
                break

            log_txt = log_recode(log_txt, f'request_date: {request_date}, page: {page}')
            
            start_index = (page-1) * max_count + 1
            end_index = page * max_count
            log_txt = log_recode(log_txt, f'start_index: {start_index}, end_index: {end_index}')

            request_date_str = request_date.strftime("%Y%m%d")
            log_txt = log_recode(log_txt, f'request_date: {request_date}, request_date_str: {request_date_str}')
            
            api_url = f"{API_URL}/{API_KEY}/{TYPE}/{API_SECTION}/{start_index}/{end_index}"
            params = {
                "OCCRRNC_DE": request_date_str
            }
            response = requests.get(api_url, params=params, timeout=10)
        
            if response.status_code == 200:
                data = response.json()  # JSON 형식인 경우
                result_code = data[API_SECTION]['result']['code']
                row = data[API_SECTION]['row']
                totalCnt = int(data[API_SECTION]['totalCnt'])
                log_txt = log_recode(log_txt, f'result_code: {result_code}, totalCnt: {totalCnt}')
                
                if result_code != "INFO-000":
                    log_txt = log_recode(log_txt, 'Not INFO-000, break')
                    break

                if totalCnt == 0:
                    row = [{"LKNTS_NM":"no_data", "OCCRRNC_DE":request_date_str},]
                    log_txt = log_recode(log_txt, 'response_data: no_data')
                else:
                    log_txt = log_recode(log_txt, 'response_data: exist')
                
                log_txt = log_recode(log_txt, f'data_length: {len(data)}')
                all_data.extend(row)
                log_txt = log_recode(log_txt, 'data extend\n', row)

            else:
                log_txt = log_recode(log_txt, f"Error: {response.status_code}")
                break  # 오류 발생 시 중단
        
        except Exception as e:
            log_txt = log_recode(log_txt, f"Error: {e}")
            break  # 오류 발생 시 중단
        
        page += 1
        wnum += 1
        time.sleep(3)

    if result_code != "INFO-000":
        break

    request_date = request_date + timedelta(days=1)


# MySQL 연결 설정
conn = pymysql.connect(host='localhost', user=USERID, password=PW, db=DBNAME, charset='utf8', cursorclass=pymysql.cursors.DictCursor)

try:
    # 커서 생성
    curs = conn.cursor()
    log_txt = log_recode(log_txt, 'MySql Data Insert Start')
    sql = "\
        INSERT INTO cattle (json_data, execute_date)\
        VALUES(%s, %s)\
    "
    fornum = 1
    for data in all_data:
        # SQL 쿼리 실행
        json_string = json.dumps(data, ensure_ascii=False)
        log_txt = log_recode(log_txt, f'{fornum}번째 데이터 기록', (json_string, today_date))
        curs.execute(sql, (json_string, today_date))
        fornum += 1
    conn.commit()

except Exception as e:
    log_txt = log_recode(log_txt, f"Error: {e}")
    conn.rollback()

finally:
    # 연결 닫기 (필수)
    conn.close()

log_txt = log_recode(log_txt, 'PROGRAM END')
with open(fname, 'w', encoding='utf-8') as f:
    f.write(log_txt)

print('log_txt save')