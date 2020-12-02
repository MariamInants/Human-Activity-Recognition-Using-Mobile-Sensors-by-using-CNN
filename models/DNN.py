from .BaseNN import *
class DNN(BaseNN):


# wrap of conv1d
    def conv1d(self,x, W, b, stride):
        x = tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='SAME')
        x = tf.add(x, b)
        return tf.nn.relu(x)


# wrap of maxpool1d
    def maxpool1d(self,x, kernel_size, stride):
        return tf.nn.max_pool(x, ksize=[1, kernel_size, 1, 1], strides=[1, stride, 1, 1], padding='VALID')


# network definition
    def network(self, X):
        W = {
            'wc1': tf.Variable(tf.random_normal([1, 64, 9, 32])),
            'wc2': tf.Variable(tf.random_normal([1, 64, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([32 * 32 * 2, 1000])),
            'wd2': tf.Variable(tf.random_normal([1000, 500])),
            'wd3': tf.Variable(tf.random_normal([500, 300])),
            'out': tf.Variable(tf.random_normal([300, 6]))
        }

        b = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1000])),
            'bd2': tf.Variable(tf.random_normal([500])),
            'bd3': tf.Variable(tf.random_normal([300])),
            'out': tf.Variable(tf.random_normal([6]))
        }
        dropout= 0.8

        conv1 = self.conv1d(X, W['wc1'], b['bc1'], 1)
        conv1 = self.maxpool1d(conv1, 2, stride=2)
        conv2 = self.conv1d(conv1, W['wc2'], b['bc2'], 1)
        conv2 = self.maxpool1d(conv2, 2, stride=2)
        conv2 = tf.reshape(conv2, [-1, W['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(conv2, W['wd1']), b['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
        fc2 = tf.add(tf.matmul(fc1, W['wd2']), b['bd2'])
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob=dropout)
        fc3 = tf.add(tf.matmul(fc2, W['wd3']), b['bd3'])
        fc3 = tf.nn.relu(fc3)
        fc3 = tf.nn.dropout(fc3, keep_prob=dropout)
        out = tf.add(tf.matmul(fc3, W['out']), b['out'])
        return out

      
       
    def metrics(self, Y, Y_pred):
        y_pred_softmax = tf.nn.softmax(Y_pred, name="y_pred_softmax")
        correct_pred = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(Y, 1))
       # tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
       # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
       # loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(Y,Y_pred))
       
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

       # tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return loss, accuracy , y_pred_softmax



