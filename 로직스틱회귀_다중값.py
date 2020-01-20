
# 다중값을 갖는 로직스틱 회귀
# 과외 횟수 추가됨

import tensorflow as tf
import numpy as np

# 실행할 때마다 같은 결과를 출력하기 위한 seed값을 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x, y의 데이터 값
# reshape(7,1) : 리스트 7행 1열로 바꾸기
x_data = np.array([[2,3], [4,3], [6,4], [8,6], [10,7], [12,8], [14,9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7,1)

# 입력 값을 플레이스 홀더에 저장
# placeholder('데이터형', '행렬의 차원', '이름') : run()동작 시 사용할 변수의 형태를 미지 지정함
# shape=[None, 2] : 행 상과없이 2 열씩 들어감
# 실제 들어갈 때 값은 2.0f, 3.0f... 이런 식으로 들어감
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기a와 바이어스b의 값을 임의로 정함
# [2,1] 의미 : 들어오는 값은 2개, 나가는 값은 1개
a = tf.Variable(tf.random_uniform([2,1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# y(예상값)시그모이드 함수의 방정식
# matmul() : 행렬곱 함수
# a1x1 + a2x2
y = tf.sigmoid(tf.matmul(X,a) + b)

# 오차(loss)를 구하는 함수
# tf.reduce_mean(x) : x의 평균계산
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1-y))

# 학습률
learning_rate = 0.1

# 오차(loss) 최소로하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 예견치 = 예상치(y)가 0.5보다 크면 실수형으로 형변환하여 변수에 저장 (0.5는 T,F의 기준 값)
predicted = tf.cast(y > 0.5, dtype=tf.float64)
# 예견치(predicted)와 실제값(Y)이 일치하면 실수형으로 형변환하여 오차함수 적용
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))


# 학습(실행)
# feed_dict={X:x_data, Y:y_data} : 튜플형식으로 키:값 데이터 넣기
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], feed_dict={X:x_data, Y:y_data})
        if (i+1) % 300 == 0:
            print("step:%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))

    new_x = np.array([7, 6]).reshape(1, 2)
    new_y = sess.run(y, feed_dict={X: new_x})

    # 활용
    # [7, 6] 은 각각 공부한 시간과 과외 수업 횟수
    print("공부한 시간: %d, 과외 수업 횟수: %d" % (new_x[:, 0], new_x[:, 1]))
    print("합격 가능성: %6.2f %%" % (new_y * 100))



