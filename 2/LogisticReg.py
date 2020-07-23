
import tensorflow as tf
import numpy as np

# [털, 날개] , [기타, 포유류, 조류] 데이터20개
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1],[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1],[1, 0], [1, 1],	[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],  
    [0, 1, 0],  
    [0, 0, 1],  
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],  
    [0, 0, 1], 
    [1, 0, 0],  
    [0, 1, 0],  
    [0, 0, 1],  
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 신경망 모델 구성

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#[입력-2 , 히든레이어1-10]
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
#[ 히든레이어1-10, 히든레이어2-20]
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
#[ 히든레이어2-20, 히든레이어3-10]
W3 = tf.Variable(tf.random_uniform([20, 10], -1., 1.))
#[ 히든레이어3-10, 출력-3]
W4 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))


# 편향
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([10]))
b4 = tf.Variable(tf.zeros([3]))

# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

L3 = tf.add(tf.matmul(L2, W3), b3)
L3 = tf.nn.relu(L3)

model = tf.add(tf.matmul(L3, W4), b4)

#비용 함수
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)



# 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


# 결과 확인

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
