# Basic implementation of "One-shot Learning with Memory-Augmented Neural Networks",
# Santoro et al., 2016 (https://arxiv.org/abs/1605.06065)
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(dim):
    initial = tf.constant(0., shape=(dim,))
    return tf.Variable(initial)


def one_hot_constant(shape, dtype=tf.float32):
    initial = np.zeros(shape)
    initial[..., 0] = 1.
    return tf.constant(initial, dtype)


class MANN(object):
    def __init__(self, output_size, batch_size, controller_size, memory_shape,
        dtype=tf.float32, gamma=.95):
        assert len(memory_shape) == 2
        
        self.dtype = dtype
        self.output_size = output_size
        self.batch_size = batch_size
        self.controller_size = controller_size
        self.memory_shape = memory_shape
        self.gamma = gamma

    def __call__(self, x):
        controller = tf.nn.rnn_cell.BasicLSTMCell(self.controller_size)
        initial_state = controller.zero_state(self.batch_size, self.dtype)

        # Read head.
        W_k = weight_variable((self.controller_size, self.memory_shape[1]))
        b_k = bias_variable(self.memory_shape[1])

        # Output for a single write head. We differ from the paper here.
        W_a = weight_variable((self.controller_size, self.memory_shape[1]))
        b_a = bias_variable(self.memory_shape[1])

        W_o = weight_variable((self.controller_size + self.memory_shape[1], self.output_size))
        b_o = bias_variable(self.output_size)

        W_g = weight_variable((self.controller_size, 1))
        b_g = bias_variable(1)
        
        def step(prev, x, eps=1e-6):
            prev_output, prev_state, prev_M, prev_r, prev_w_r, _, prev_w_u, prev_w_lu = prev
            output, state = controller(tf.concat(1, [x, prev_r]), prev_state)

            # Compute similarity and w_r.
            k = tf.nn.tanh(tf.matmul(output, W_k) + b_k)  # (batch_size, memory_shape[1])
            unnorm = tf.squeeze(tf.batch_matmul(prev_M, tf.expand_dims(k, -1)))
            k_norm2 = tf.reduce_sum(k ** 2, reduction_indices=-1, keep_dims=True)
            m_norm2 = tf.reduce_sum(prev_M ** 2, reduction_indices=-1)
            w_r = tf.nn.softmax(unnorm / tf.sqrt(k_norm2 * m_norm2 + eps))  # (batch_size, memory_shape[0])  # eq. (3)
            
            # Read from memory.
            r = tf.squeeze(tf.batch_matmul(tf.expand_dims(w_r, 1), prev_M))  # (batch_size, memory_shape[1])  # eq. (4)
            
            # Write to memory.
            alpha = tf.matmul(output, W_g) + b_g
            g = tf.nn.sigmoid(alpha)  # (batch_size, 1)
            w_w = g * prev_w_r + (1. - g) * prev_w_lu  # (batch_size, memory_shape[0])
            w_u = self.gamma * prev_w_u + w_r + w_w  # (batch_size, memory_shape[0])
            
            # Update w_lu.
            _, sorted_indexes = tf.nn.top_k(w_u, k=memory_shape[0], sorted=True)
            w_lu = tf.one_hot(sorted_indexes[:, -1], self.memory_shape[0])
            
            # Set all least recently used memory locations to zero.
            prev_M = prev_M * (1. - tf.tile(tf.expand_dims(prev_w_lu, -1), [1, 1, self.memory_shape[1]]))
            
            # Write back to memory.
            M = []
            a = tf.nn.tanh(tf.matmul(output, W_a) + b_a)  # (batch_size, memory_shape[1])
            for idx in range(self.memory_shape[0]):
                M.append(prev_M[:, idx, :] + tf.expand_dims(w_w[:, idx], -1) * a)
            M = tf.transpose(tf.pack(M), perm=[1, 0, 2])
            
            # Combine into output.
            o = tf.matmul(tf.concat(1, [output, r]), W_o) + b_o  # (batch_size, self.output_size)
            return [o, state, M, r, w_r, w_w, w_u, w_lu]

        init = [
            tf.zeros((self.batch_size, self.output_size)),              # outputs
            initial_state,                                              # states
            tf.zeros((self.batch_size,) + self.memory_shape),           # M
            tf.zeros((self.batch_size, self.memory_shape[1])),          # r
            one_hot_constant((self.batch_size, self.memory_shape[0])),  # w_r
            tf.zeros((self.batch_size, self.memory_shape[0])),          # w_w
            one_hot_constant((self.batch_size, self.memory_shape[0])),  # w_u
            tf.zeros((self.batch_size, self.memory_shape[0])),          # w_lu
        ]
        return tf.scan(step, x, initializer=init)


def one_hot(xs, nb_classes):
    encoded_xs = []
    for batch in xs:
        encoded_xs.append([])
        for x in batch:
            e = np.zeros(nb_classes)
            if x > 0:
                e[x - 1] = 1.
            encoded_xs[-1].append(e)
    return np.array(encoded_xs)


def fill(xs, nb_classes):
    encoded_xs = []
    for batch in xs:
        encoded_xs.append([])
        for x in batch:
            e = np.zeros(nb_classes)
            e[:x] = 1.
            encoded_xs[-1].append(e)
    return np.array(encoded_xs)


def generate_data(batch_size, nb_classes, input_size, length):
    mapping = np.random.random_integers(nb_classes, size=(batch_size, input_size))
    xs = np.random.random_integers(input_size, size=(batch_size, length))
    ys = []
    for batch_idx in range(batch_size):
        ys.append([mapping[batch_idx, x - 1] for x in xs[batch_idx, :]])
    ys = np.array(ys)
    assert xs.shape == ys.shape

    xs = fill(xs, nb_classes=input_size)
    ys = one_hot(ys, nb_classes=nb_classes)
    prev_ys = np.hstack([np.zeros((batch_size, 1, nb_classes)), ys[:, :-1]])
    assert prev_ys.shape == ys.shape

    axes = [1, 0, 2]
    return np.transpose(xs, axes), np.transpose(ys, axes), np.transpose(prev_ys, axes)


input_size = 5
nb_classes = 5
batch_size = 16
controller_size = 200
memory_shape = (128, 40)
mann = MANN(nb_classes, batch_size, controller_size, memory_shape)
x = tf.placeholder(tf.float32, shape=(None, batch_size, input_size))
y = tf.placeholder(tf.float32, shape=(None, batch_size, nb_classes))
prev_y = tf.placeholder(tf.float32, shape=(None, batch_size, nb_classes))
all_variables = mann(tf.concat(2, [x, prev_y]))
outputs = all_variables[0]

# Add softmax over output.
y_ = tf.nn.softmax(outputs)

# Compute loss.
diff = tf.reduce_sum(y * tf.log(y_), reduction_indices=[2, 0])
cross_entropy = -tf.reduce_mean(diff)

# Measure accuracy.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training.
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(100000):
        xs, ys, prev_ys = generate_data(batch_size=batch_size, nb_classes=nb_classes, input_size=input_size, length=10)
        feed = {
            x: xs,
            y: ys,
            prev_y: prev_ys,
        }
        _, loss, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict=feed)
        print('step {}: loss={}, accuracy={}'.format(i, loss, acc))
