from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from dotenv import dotenv_values
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# 환경변수 로드
config = dotenv_values('.env')
DATA_PATH = config['DATA_PATH']

# 데이터 불러오기
df = pd.read_parquet(DATA_PATH)

# 데이터 분석에 사용할 컬럼 선언
x_cols = [
    'avgTa', 'minTa', 'maxTa', 'avgRhm',
]

y_cols = [
    '가금티푸스', '결핵병', '고병원성조류인플루엔자', '낭충봉아부패병', 
    '돼지생식기호흡기증후군', '브루셀라병', '사슴만성소모성질병', 
    '아프리카돼지열병', '추백리'
]

columns = x_cols + y_cols

# 데이터 분석에 사용되는 독립변수와 종속변수 컬럼만 슬라이싱
df = df.loc[:, columns]

# 종속변수 데이터 크기 확인(300 이상 되는 데이터만 분석)
df_sum = df[y_cols].sum()
df_sum_verty = df_sum > 300
df_sum_verty_index = df_sum[df_sum > 300].index


# 1. 독립변수 결측치 및 이상치 처리
# 독립변수 결측치 처리
df_pre_droped = df.copy()
df_pre_droped.dropna(subset=x_cols, inplace=True)
df_pre_droped.reset_index(drop=True, inplace=True)

# 독립 변수 타입 처리
for xc in x_cols:
    df_pre_droped.loc[:, xc] = pd.to_numeric(df_pre_droped[xc], errors='coerce')

df_pre_droped[x_cols] = df_pre_droped[x_cols].astype(float)


# 독립변수 이상치 처리
for xc in x_cols:
    Q1 = df_pre_droped[xc].quantile(0.25)
    Q3 = df_pre_droped[xc].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_low = df_pre_droped[df_pre_droped[xc] < lower_bound]
    df_high = df_pre_droped[df_pre_droped[xc] > upper_bound]
    df_pre_droped = df_pre_droped[(df_pre_droped[xc] >= lower_bound) & (df_pre_droped[xc] <= upper_bound)]
    df_pre_droped.reset_index(drop=True, inplace=True)


# 2. 종속변수 별 데이터 분석

# 종속 변수 분석 대상 리스트 생성
y_cols_droped = list(df_sum_verty_index)

# 반복문을 통한 종속 변수 별 분석 수행
for yc in y_cols_droped:
    df_droped = df_pre_droped[x_cols + [yc]]

    # 종속 변수 결측치 처리
    df_droped[yc] = df_droped[yc].fillna(0)

    # 종속 변수 타입 처리
    df_droped[yc] = df_droped[yc].astype(int)
    df_droped = df_droped[x_cols + [yc]].dropna().reset_index(drop=True)

    # 감염 수만큼 종속 변수 행 복제
    df_droped['cnt'] = df_droped[yc]
    df_droped.loc[df_droped['cnt']==0, 'cnt'] = 1
    df_droped = df_droped.loc[df_droped.index.repeat(df_droped['cnt'])]
    df_droped = df_droped.drop(columns=['cnt']).reset_index(drop=True)
    df_droped.loc[df_droped[yc] > 0, yc] = 1

    # 전체 결측치 처리
    df_droped.dropna(inplace=True)
    df_droped.reset_index(drop=True, inplace=True)


    # 언더샘플링
    x = df_droped.drop(columns=[yc])
    y = df_droped[yc]

    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    print(f'x, y count: {len(x_resampled)}, {len(y_resampled)}')

    # 데이터 분리 (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # 데이터 표준화(정규화, 스케일링)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # XGBoost 모델 학습
    model = XGBClassifier(eval_metric='auc')
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 확률 예측

    # 성능 지표 저장
    report = classification_report(y_test, y_pred)
    roc_auc = "ROC-AUC Score:", roc_auc_score(y_test, y_prob)
    model_report = str(report) + '\n' + str(roc_auc)
    with open(f'{yc}_model_report.txt', 'w') as f:
        f.write(model_report)

    # 모델 저장
    joblib.dump(model, f'{yc}_XGBoost_moder.pkl')