import tensorflow as tf

input_x = tf.placeholder(tf.int32, [None, 256, 256, 3], name='input_x')
input_y = tf.placeholder(tf.float32, [None, 8], name='input_y')

conv = tf.layers.conv2d(
    inputs=input_x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

fc = tf.layers.dense(conv, 128, name='fc1')
fc = tf.contrib.layers.dropout(fc, 0.2)
fc = tf.nn.relu(fc)

# 分类器
fc2 = tf.layers.dense(fc, 8, name='fc2')
y_pred_coords = tf.nn.relu(fc2)
