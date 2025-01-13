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
