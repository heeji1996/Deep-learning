
# 선형회귀_평균제곱근오차

import numpy as np

# 기울기 a와 y절편
ab = [3, 76]

# x, y의 데이터 값
data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]  # 2,4,6,8
y = [i[1] for i in data]

# y=ax+b에 a와 b 값을 대입하여 결과를 출력하는 함수 : 예측값
def predict(x):
    return ab[0]*x + ab[1]

# RMSE(평균제곱근오차) 함수
def rmse(p, a):  # 예측치(predict_result), 실제값(y)
    return np.sqrt(((p-a) ** 2).mean())

# RMSE함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

# 예측값이 들어갈 빈 리스트
predict_result = []

# 모든 x값을 한 번씩 대입하여
for i in range(len(x)): # 4
    # prdict_result 리스트 완성
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

# 최종 RMSE 출력
print("rmse 최종값 : " + str(rmse_val(predict_result, y)))