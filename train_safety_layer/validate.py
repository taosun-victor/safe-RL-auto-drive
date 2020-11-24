import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

c_old, act0, act1, c_new = np.loadtxt("cte.log", unpack=True)
action = np.column_stack((act0, act1))
c_old = np.asarray(c_old).reshape(-1, 1)
c_new = np.asarray(c_new).reshape(-1, 1)
#print ("action\n",action,"\n", "c_old\n", c_old, "\n", "c_new\n", c_new, "\n")
#print (action.shape, c_old.shape, c_new.shape)

with open("obs.log") as f:
    obs = [line.split() for line in f]
obs = np.asarray(obs).astype(float)
#print ("obs\n", obs)


tf_obs = tf.placeholder(tf.float32, obs.shape)
tf_c_old = tf.placeholder(tf.float32, c_old.shape)
tf_action = tf.placeholder(tf.float32, action.shape)
tf_c_new = tf.placeholder(tf.float32, c_new.shape)

hidden1 = tf.layers.dense(tf_obs, 64, tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 16, tf.nn.relu)
output = tf.layers.dense(hidden2, 2)

correction = tf.reduce_sum(tf.math.multiply(output, tf_action), 1)
#print (correction.shape)
correction = tf.reshape(correction, tf_c_old.shape)
c_pred = tf_c_old + correction

loss = tf.losses.mean_squared_error(tf_c_new, c_pred)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "../saved_params/safe_layer")


feed_dict = {tf_obs: obs, tf_c_old: c_old, tf_action: action, tf_c_new: c_new}


l = sess.run(loss, feed_dict)
print (l)
