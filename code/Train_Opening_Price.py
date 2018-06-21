import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
import os

data = np.load("50.npy")
data_normal = (data - data.min())/(data.max()-data.min())
# data_re = 1.521356 * (data.max()-data.min()) + data.min()
data_open_n = data_normal[:1200,0]

train_keep_prob = 0.6
max_steps = 600
batch_size = 2
num_steps = 50
num_classes = 1
learning_rate = 1e-4
num_layers = 3
lstm_size = 256
grad_clip = 5

tf.reset_default_graph()

def get_batches(arr, n_seqs, n_steps):
    
    '''
    arr: data to be divided
    n_seqs: batch-size, # of input sequences
    n_steps: timestep, # of characters in a input sequences
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

## create RNN model
# input and target
with tf.name_scope('inputs'):
    inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 1), name='inputs')
    targets = tf.placeholder(tf.float32, shape=(batch_size, num_steps), name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# create LSTM cell
def get_a_cell(lstm_size, keep_prob):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    drop = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
    return drop

# create muti rnn cell
with tf.name_scope('LSTMRNN'):
    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    RNN_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    seq_output = tf.concat(RNN_outputs, 1)
    x = tf.reshape(seq_output, [-1, lstm_size])  
    
# output
with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal([lstm_size, num_classes], stddev=0.01))
    softmax_b = tf.Variable(tf.zeros(num_classes))

logits = tf.matmul(x, softmax_w) + softmax_b
proba_prediction = logits
# proba_prediction = tf.nn.softmax(logits, name='predictions')

# loss
with tf.name_scope('loss'):
#     y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(targets, [100,1])
    loss_nr = tf.square(y_reshaped - logits)
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss_nr)

# optimizer
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
train_op = tf.train.AdamOptimizer(learning_rate)
optimizer = train_op.apply_gradients(zip(grads, tvars))


cross_entropy_loss = np.zeros(max_steps)
train_error = np.zeros(max_steps)

session = tf.Session()        
with session as sess:
    sess.run(tf.global_variables_initializer())
    # Train network
    step = 0
    initial_state = cell.zero_state(batch_size, tf.float32)
    new_state = sess.run(initial_state)
    for epoch in range(0, max_steps):
        for x, y in get_batches(data_open_n, batch_size, num_steps):
            x = np.reshape(x, [batch_size, num_steps, 1])
            feed = {inputs: x, targets: y, keep_prob: train_keep_prob, initial_state: new_state}
            cross_entropy_loss[epoch],loss_n , pre_train, new_state, _ = sess.run([loss, proba_prediction, proba_prediction, final_state, optimizer], feed_dict=feed)

    #save model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('/home/xxxxxx', 'Final_model_lstm'))
    
# Cross Entropy
plt.figure()
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.title('Learning Curve')
plt.plot(cross_entropy_loss)
plt.savefig('Final_Cross_Entropy_LSTM')
plt.show()
