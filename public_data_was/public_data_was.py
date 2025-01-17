from datetime import datetime
import pymysql
from dotenv import dotenv_values
import json
from flask import Flask, request
import pandas as pd


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


def mysql_connect(table, start_date=None, end_date=None):
    start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    config = dotenv_values('.env')
    USERID = config['USERID']
    PW = config['PW']
    DBNAME = config['DBNAME']
    TABLE_LIST = config['TABLE_LIST']
    tables = TABLE_LIST.split(',')

    log_txt = log_recode('', 'add dotenv_values')

    # MySQL 연결 설정
    conn = pymysql.connect(
        host='localhost', user=USERID, password=PW, 
        db=DBNAME, charset='utf8', cursorclass=pymysql.cursors.DictCursor
    )

    row = ''

    try:
        # 커서 생성
        curs = conn.cursor()
        vars = []

        # 쿼리 생성
        sql = "SELECT * FROM "      
        if table in tables:
            sql += table
        else:
            raise Exception("Wrong table name")
        
        if start_date:
            sql += " WHERE execute_date > %s"
            vars.append(start_date)
            if end_date:
                sql += " AND execute_date < %s"
                vars.append(end_date)
        sql += " ORDER BY id DESC, execute_date DESC"
        print(f'sql: {sql}, vars: {vars}')

        # SQL 쿼리 실행
        curs.execute(sql, vars)
        row = curs.fetchall()

    except Exception as e:
        log_txt = log_recode(log_txt, f"Error: {e}")

    finally:
        if conn:
            conn.close()

    return (row, log_txt)



app = Flask(__name__)
tables = ['cattle', 'temperature']

@app.route('/api')
def home():
    log_txt = log_recode('', 'request\n')
    result = ''
    config = dotenv_values('.env')
    DATA_STORAGE = config['DATA_STORAGE']
    LOG_NAME = config['LOG_NAME']
    API_KEY = config['API_KEY']
    fname = filename_maker(f'{DATA_STORAGE}/{LOG_NAME}')

    try:
        api_key = request.args.get('api_key')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        table = request.args.get('table')
        start_datetime = datetime.strptime(start_date, "%Y%m%d%H%M%S")
        end_datetime = datetime.strptime(end_date, "%Y%m%d%H%M%S")

        if api_key != API_KEY:
            raise Exception("Wrong api_key")
        elif start_datetime > end_datetime:
            raise Exception("start_datetime bigger than end_datetime")
        elif not table in tables:
            raise Exception("Wrong table name")
        else:
            (row, add_log) = mysql_connect(table, start_datetime, end_datetime)
            log_txt += add_log
            df = pd.DataFrame(row)
            json_data = df.to_json(orient='records', date_format='iso')
            result = (json_data, 200)
        
    except Exception as e:
        log_txt = log_recode(log_txt, f"error: {e}")
        error_msg = {"error": "Internal Server Error", "code": 500}
        result = (json.dumps(error_msg, ensure_ascii=False), 500)
    
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(log_txt)
        
    return result


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # Flask는 내부에서만 실행