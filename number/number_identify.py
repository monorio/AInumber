import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_connection():
    mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.1))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))

    y_predict = tf.matmul(x, weights) + bias

    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(error)

    equal_list = tf.equal(
        tf.argmax(y_predict, axis=1),
        tf.argmax(y_true, axis=1)
    )

    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            image_batch, label_batch = mnist.train.next_batch(100)
            _, error_value, accuracy_value = sess.run([optimizer, error, accuracy],
                                                      feed_dict={x: image_batch, y_true: label_batch})

            print("Train %d, loss %f: , Accurancy is : %f" % (i + 1, error_value, accuracy_value))
    return None


if __name__ == "__main__":
    print()
    full_connection()
