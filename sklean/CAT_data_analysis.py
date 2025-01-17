from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import glob


"""

# 데이터 불러오기
df = pd.read_parquet('data.parquet')

# 특징량과 타겟 변수 분리
X = df.drop('target_variable', axis=1)
y = df['target_variable']

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
"""




"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 특성
y = iris.target  # 라벨

# 이진 분류를 위해 클래스 2개만 선택 (예: 클래스 0과 1)
X = X[y != 2]
y = y[y != 2]

# 2. 데이터 분리 (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 예측
y_pred = model.predict(X_test)

# 6. 성능 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
ROC-AUC 또는 PR-AUC

# 7. 특성 중요도 출력 (로지스틱 회귀의 계수)
print("Coefficients:\n", model.coef_)
print("Intercept:\n", model.intercept_)
"""



"""
데이터 탐색:

df.describe()는 데이터의 기술 통계량(평균, 표준 편차 등)을 출력하여 데이터의 분포를 파악하는 데 도움을 줍니다.
df.corr()는 변수 간의 상관 관계를 계산하여 변수 간의 관계를 파악하는 데 도움을 줍니다.
sns.pairplot(df)는 각 변수 쌍에 대한 산점도를 그려 변수 간의 관계를 시각적으로 보여줍니다.
선형성 확인:

sns.regplot(x='A', y='B', data=df)는 변수 A와 B 간의 산점도와 함께 회귀선을 그려 선형적인 관계를 시각적으로 보여줍니다.
추가 분석:

다른 변수 쌍에 대해 sns.regplot()을 사용하여 선형성을 확인할 수 있습니다.
잔차 분석을 수행하여 선형 모델의 적합성을 평가할 수 있습니다.
변수 변환(로그 변환, 제곱근 변환 등)을 통해 선형성을 개선할 수 있습니다.
"""


"""
import pandas as pd
import numpy as np

# 예제 데이터 생성
data = {
    'age': [25, 30, 28, np.nan, 22, 27, 35, 29, 24, 120],
    'income': [50000, 60000, 55000, 70000, 48000, 52000, 65000, 58000, np.nan, 60000]
}
df = pd.DataFrame(data)

# 1. 결측치 처리

# (1) 결측치 확인
print(df.isnull().sum())

# (2) 결측치 제거
df_dropped = df.dropna()  # 모든 결측치가 있는 행 제거
print(df_dropped)

# (3) 결측치 대체
df_filled = df.fillna(df.mean())  # 평균값으로 대체
print(df_filled)

# 2. 이상치 처리

# (1) 이상치 확인 (Boxplot 시각화)
import matplotlib.pyplot as plt
plt.boxplot(df['age'])
plt.show()

# (2) IQR을 이용한 이상치 제거
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outlier = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]
print(df_no_outlier)

# (3) 이상치 대체 (최대/최소값으로 대체)
def replace_outlier(data, column):
  Q1 = data[column].quantile(0.25)
  Q3 = data[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  data.loc[data[column] < lower_bound, column] = lower_bound
  data.loc[data[column] > upper_bound, column] = upper_bound
  return data

df_replaced = replace_outlier(df.copy(), 'age')
print(df_replaced)
"""