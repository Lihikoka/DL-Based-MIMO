import tensorflow as tf
import numpy as np


def batch_norm(x, n_out, is_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                        name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                        name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, 0, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                    mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)
    return normed


def disH(H, shape1, shape2, vbit, is_train): #  May have to be redesigned
    """
    Discretize the input
    H: 2-dimensional real-valued channel matrix (e.g. [[H_re, -H_im],
                                                       [H_im, H_re ]])
    vbit: quantize to v-bit values
    is_train: bool variable to indicate whether is training
    """
    # Length of shape1 and shape2 must be 2
    assert len(shape1) == 2 && len(shape2) == 2
    with tf.variable_scope('disH'):
        # Flatten the 2d channel matrix to a 1d channel vector
        H_flat = tf.reshape(H, [-1])

        # Shapes of layer 1 and layer 2
        shape1 = [H_flat.size, H_flat.size / 2]
        shape2 = [H_flat.size / 2, vbit]

        # Layer 1: FC/ReLU
        W_fc1 = tf.get_variable("W_fc1", shape1,
            initializer=tf.random_normal_initializer())
        b_fc1 = tf.get_variable("b_fc1", shape1[1],
            initializer=tf.constant_initializer(1.0))
        relu_fc1 = tf.nn.relu(tf.add(tf.matmul(H_flat_real, W_fc1), b_fc1))
        bn_fc1 = batch_norm(relu_fc1, shape1[1], is_train)

        #Layer 2: FC/Softmax
        W_fc2 = tf.get_variable("W_fc2", shape2,
            initializer=tf.random_normal_initializer())
        b_fc2 = tf.get_variable("b_fc2", shape2[1],
            initializer=tf.constant_initializer(1.0))
        sm_fc2 = tf.nn.softmax(tf.add(tf.matmul(bn_fc1, W_fc2), b_fc2))
        # Convert softmax output to one-hot vector
        H_v = tf.one_hot(tf.argmax(sm_fc2, dimension=0), depth=vbit)
        return H_v


def txenc(s, H, is_train):
    """
    s: 2^k-dimension one-hot message vector (k=2 for QPSK, k=4 for 16-QAM, etc)
    H: real-valued channel matrix
    is_train: bool variable to indicate whether is training
    """
    vbit = 2
    shape_fc1 = [H.size + s.size, H.size + s.size] # [16 + 4, 16 + 4] for 2 x 2 MIMO
    shape_fc2 = [H.size + s.size, 4]

    with tf.variable_scope("disH"):
        H_v = disH(H, 2)
    # Layer 1: Concatenate H_v and x
    input_concat = tf.concat(0, [H_v, s])

    # Layer 2: FC/ReLU
    W_fc1 = tf.get_variable("W_fc1", shape_fc1,
        initializer=tf.random_normal_initializer())
    b_fc1 = tf.get_variable("b_fc1", shape_fc1[1],
        initiailizer=tf.constant_initializer(1.0))
    relu_fc1 = tf.nn.relu(tf.add(tf.matmul(input_concat, W_fc1), b_fc1))
    bn_fc1 = batch_norm(relu_fc1, shape_fc1[1], is_train)

    # Layer 3: FC/Linear
    W_fc2 = tf.get_variable("W_fc2", shape_fc2,
        initializer=tf.random_normal_initializer())
    b_fc2 = tf.get_variable("b_fc2", shape_fc2[1],
        initiailizer=tf.constant_initializer(1.0))
    linear = tf.add(tf.matmul(bn_fc1, W_fc2), b_fc2)
    # Layer 4: Normalization
    mean, var = tf.moments(linear, 0, name='moments')
    x = tf.batch_normalization(linear, mean, var, tf.constant(0.0),
                               tf.constant(1.0), 0.001)
    return x


def channel(x, H, snrdB):
    """
    x: transmitted symbol, [x_re, x_im], E[xx*] = I
    H: 2-dimensional real-valued channel matrix (e.g. [[H_re, -H_im],
                                                       [H_im, H_re ]])
    snrdB: SNR in dB
    """
    Hx = tf.matmul(x, H)
    snr = tf.pow(10, snrdB/10)
    noise = tf.random_normal(Hx.shape, 0, 1/snr)
    y = tf.add(Hx, noise)
    return y


def rxdec(y):
    """
    y: received (4 x 1 for 2 x 2 MIMO)
    """
    shape1 = [4, 8]
    shape2 = [8, 4]

    W_fc1 = tf.get_variable("W_fc1", shape1,
        initializer=tf.random_normal_initializer())
    b_fc1 = tf.get_variable("b_fc1", shape1[1],
        initializer=tf.constant_initializer(1.0))
    relu = tf.nn.relu(tf.add(tf.matmul(H_flat_real, W_fc1), b_fc1))

    #Layer 2: FC/Softmax
    W_fc2 = tf.get_variable("W_fc2", shape2,
        initializer=tf.random_normal_initializer())
    b_fc2 = tf.get_variable("b_fc2", shape2[1],
        initializer=tf.constant_initializer(1.0))
    s_est = tf.nn.softmax(tf.add(tf.matmul(relu, W_fc2), b_fc2))
    return s_est


def train(s, H, s_hat, model_location, summary_location,
          training_epochs, batch_size=32, LR=0.0001, traintestsplit=0.1):
    """
    s: one-hot vector of the transmitted symbol
    H: complex channel matrix, dtype=np.complex64
    """
    def complex2Real(H):
        H_re = H.real
        H_im = H.imag
        H_row1 = np.concatenate((H_re, -H_im), axis=1)
        H_row2 = np.concatenate((H_im, H_re), axis=1)
        H_r = np.concatenate((H_row1, H_row2), axis=0)
        return H_r
    H_real = complex2Real(H)

    n_total = s.shape[0]
    n_train = int(n_total * traintestsplit)
    n_test = n_total - n_train

    n_input = s.shape[1]

    s = tf.placeholder(tf.float32, [None, n_input])
    s_ = tf.placeholder(tf.float32, [None, n_input])
    H_ = tf.placeholder(tf.float32, [None] + H.shape)
    is_train = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
