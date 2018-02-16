import tensorflow as tf
C_1 = tf.constant(5.0, name = "cool")
C_2 = tf.constant(1.0)
C_3 = tf.constant(2.0)

golden_ratio = (tf.sqrt(C_1) + C_2)/C_3

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    print sess.run(golden_ratio)
    writer.close()
