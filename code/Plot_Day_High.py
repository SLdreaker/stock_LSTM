import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

data = np.load("50.npy")
data_normal = (data - data.min())/(data.max()-data.min())
# data_re = 1.521356 * (data.max()-data.min()) + data.min()
data_open_n = data_normal[:1200,1]

train_keep_prob = 1.0
max_steps = 250
batch_size = 1
num_steps = 50
num_classes = 1
learning_rate = 1e-4
num_layers = 3
lstm_size = 256
grad_clip = 5

tf.reset_default_graph()

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
    y_reshaped = tf.reshape(targets, [batch_size*num_steps,1])
    loss_nr = tf.square(y_reshaped - logits)
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss_nr)


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
        
re_ = np.zeros(max_steps)
saver = tf.train.Saver()
session = tf.Session() 
with session as sess:
    initial_state = cell.zero_state(batch_size, tf.float32)
    new_state = sess.run(initial_state)
    saver.restore(session, '/home/xxxxxx/Final_model_lstm_hight')
    
    
    train_batches = get_batches(data_open_n, batch_size, num_steps)
    x, y = next(train_batches)
    save_x = x
    for epoch in range(0,max_steps):
        x = np.reshape(x, [batch_size, num_steps, 1])
        feed = {inputs: x, targets: y, keep_prob: train_keep_prob, initial_state: new_state}
        pre_train, new_state = sess.run([proba_prediction, final_state],feed_dict=feed)
        pre_train = np.reshape(np.array(pre_train), [batch_size, num_steps])
        pre_train = pre_train[0,batch_size]
        re_[epoch] = pre_train * (data.max()-data.min()) + data.min()
        x[0,:num_steps-1,0] = x[0,1:num_steps,0]
        x[0,num_steps-1,0] = pre_train
        
np.save("DayHigh.npy",re_)
plt.plot(re_, label='predict')
plt.plot(data[num_steps:max_steps+num_steps,1], label='real')
plt.legend(loc='upper right')
plt.title('Yuan Da Taiwan 50 - Day High')
plt.xlabel('day')
plt.ylabel('price')
plt.show()
