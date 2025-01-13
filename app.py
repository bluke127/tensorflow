import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 공부 시간 (랜덤 데이터 생성)
# np.random.seed()를 설정하는 이유는 **난수 생성의 재현성(reproducibility)**을 보장하기 위해서입니다. 재현성이란, 동일한 코드와 데이터를 사용할 때 항상 같은 결과를 얻을 수 있도록 하는 것을 의미합니다.
np.random.seed(42)
# 첫번째 파라미터는 난수의 최소값
# 두번짼느 최댓값
# 세번째는 몇개생성할지
study_hours = np.random.uniform(1, 10, 100)  # 공부 시간: 1~10시간
print(study_hours)
print(np)
# 중간고사 성적 (노이즈 추가)
scores = 5 * study_hours + np.random.normal(0, 5, size=100)  # y = 5x + noise

# 데이터 시각화
plt.scatter(study_hours, scores)
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Scores")
plt.show()
# Sequential 모델 생성
model = Sequential([
    Dense(1, input_shape=(1,), activation='linear')  # 입력 1개, 출력 1개
])

# Dense:
# 완전 연결층(fully connected layer)을 정의합니다.
# 모든 입력 노드가 모든 출력 노드와 연결됩니다.
# 파라미터 설명:
# 1 (units):
#
# 이 레이어의 출력 노드 수를 지정합니다.
# 여기서는 출력이 1개인 모델을 만듭니다.
# (즉, 선형 회귀 모델에서는 단 하나의 값(예측 값)을 출력합니다.)
# input_shape=(1,):
#
# 입력 데이터의 **형태(shape)**를 정의합니다.
# (1,)은 입력으로 **하나의 특성(feature)**만 사용한다는 뜻입니다.
# 예를 들어, 공부 시간(x) 하나만 입력으로 받는 상황입니다.
# activation='linear':
#
# 활성화 함수(activation function)를 지정합니다.
# linear는 입력값 그대로 출력값을 전달합니다.
# 수식으로는
# 𝑓
# (
# 𝑥
# )
# =
# 𝑥
# f(x)=x.
# 선형 회귀 모델은 출력에 비선형 변환을 하지 않기 때문에 linear 활성화 함수를 사용합니다.

# 모델 컴파일
model.compile(optimizer='sgd', loss='mse')  # 옵티마이저: SGD, 손실 함수: MSE

history = model.fit(study_hours, scores, epochs=50, verbose=1)