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

# 모델 컴파일
model.compile(optimizer='sgd', loss='mse')  # 옵티마이저: SGD, 손실 함수: MSE

history = model.fit(study_hours, scores, epochs=50, verbose=1)