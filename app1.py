# 판다스
# 데이터를 표로 만드는 라이브러리
# - 파일 읽어오기 : pd.read_csv(경로)
# - 모양 확인하기 : print(데이터.shape)
# - 칼럼 선택하기 : 데이터[['칼럼명1', '칼럼명2', '칼럼명3']]
# - 칼럼 이름 출력하기 : print(데이터.columns)
import tensorflow as tf
import pandas as pd
파일경로 = "https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv"
데이터 = pd.read_csv(파일경로)
# //기본적으로 5개 출력
데이터.head()
독립 = 데이터[['온도']]
종속 = 데이터[['판매량']]
# .shape는 Dataframe(행과열로 구성)의 행과열을 반환
print(데이터.columns)
print(독립.shape, 종속.shape)
#독립변수의 개수
x = tf.keras.layers.Input(shape=[1])
#종속변수의 개수
y = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(x, y)
#lose값은 모델이 얼마나 좋냐 안좋냐로 보면됨
#0에 가까워야 정답을 맞힘
model.compile(loss='mse')
#epochs 시도횟수
#verbose 출력횟수
# 처음에 10만 햇을때는 loss가 1000이었지만 10000 돌린후 다시 확인해며면 2.23로 상당히 낮아진걸 볼 수 있다
model.fit(독립,종속,epochs=10000,verbose=0)
model.predict(독립)
model.predict([15])