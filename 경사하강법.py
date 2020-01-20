
# 경사하강법

import tensorflow as tf

#x, y의 데이터 값
data = [[2,81], [4,93], [6,91], [8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# 기울가 a와 y절편b의 값을 임의로 정한다.
# 단, 기울기의 범위는 0~10사이이며, y절편은 0~100사이에서 변하게한다.
# Variable() : 변수값을 정할 때 사용하는 함수
# 임의의 기울기 = 리스트 요소 한개의 0~10사이의 임의의 숫자 발생, tf.random_uniform([1], 0, 10,
#   데이터타입 = 실수64bit, 실행시 0과 같은 값이 나올 수 있게 설정 dtype=tf.float64, seed=0
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# y(예측값)에 대한 일차방정식 ax+b의 식을 세운다
y = a * x_data + b

# 텐서플로우 RMSE(평균제곱근오차) 함수
# tf.sqrt(x) : x의 제곱근 계산
# tf.reduce_mean(x) : x의 평균계산
# tf.square(x) : x의 제곱계산
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로하는 값 찾기 (학습률과 평균제곱근 오차를 인자로 줌)
# GradientDescentOptimizer() : 경사하강법
# minimize(rmse) : rmse의 가장 작은 값 찾아가는 함수
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로우를 이용한 학습 = 실행
# with 블럭 탈출 시 session이 종료됨
# 학습 할 수록 기울기 a는 0에 가깝게 출력됨
# sess.run() : 실행
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # 2001번 실행(0번째를 포함하므로)
    for step in range(2001):
        sess.run(gradient_descent)
        # 100번마다 결과 출력
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y절편b = %.4f"
                  % (step, sess.run(rmse), sess.run(a), sess.run(b)))
