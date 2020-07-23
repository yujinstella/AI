import tensorflow as tf

height = [ 150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170, 180]
weight = [42.5, 43.4, 44.2, 45.1, 45.9, 46.8, 47.6, 48.5, 49.3, 50.2, 51.0, 51.9, 52.7, 53.6, 54.4, 55.3, 56.1, 57.0, 57.8, 58.7, 59.5, 68.0]

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -100, 100))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00003)
train_op = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: height, Y: weight})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    print("X: 180, Y:", sess.run(hypothesis, feed_dict={X: 180}))
    print("X: 150, Y:", sess.run(hypothesis, feed_dict={X: 150}))