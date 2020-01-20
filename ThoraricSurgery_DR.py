
# 딥러닝을 구동하는 데 필요한 케라스 함수 호출
from keras.models import Sequential # 딥러닝 구조를 한층 한층 쉽게 쌓아올릴 수 있게 해준다 model.add()
from keras.layers import Dense # 각 층의 옵션 설정

#필요한 라이브러리 호출
# numpy : 수치 계산 라이브러리, 데이터 분석에 많이 사용됨
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 수술환장 데이터 부르기
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = Data_set[:,0:17]  # 속성 : 항목
Y = Data_set[:,17]    # 클래스 : 결과값

# 딥러닝 구조 결정 ( 모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

# 결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(X, Y)[1]))
