import tensorflow as tf

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, 1, 1], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target_prediction')

# Trivial linear model
y_ = tf.identity(tf.layers.dense(x, 1), name='output')

# Optimize loss
loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

# Write the graph out to a file.
with open('graph.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
