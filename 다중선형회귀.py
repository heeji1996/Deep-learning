
# 다중선형회기 (Multi limear regression)
# 과외횟수(x2) 추가됨

import tensorflow as tf

# x1, x2, y의 데이터 값
data =[[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]  # 실제값

# 임의의 기울기a 와 y의절편 b 값
# 단, 기울기의 범위는 0~10 사이며 y절편은 0~100사이에서 변한다
a1 = tf.Variable(tf.random_uniform([1],0,10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1],0,10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1],0,100, dtype=tf.float64, seed=0))

# 새로운 방정식 : x인자가 하나 더 추가 되었으므로 기울기도 하나 더 추가해야한다.
y = a1*x1 + a2*x2 + b   # 예측값

# 텐서플로우 RMSE 함수 (평균오차제곱근)
rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))

# 학습률
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기 (경사하강법)
# GradientDescentOptimizer() : 경사하강법
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 학습이 시작되는 부분
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_descent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y절편b = %.4f"
                  % (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))
