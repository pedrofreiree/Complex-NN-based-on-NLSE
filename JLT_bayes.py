# imports we know we'll need
from __future__ import print_function
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer
import numpy as np
import scipy.io as sio
import BER_calc
from datetime import datetime
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
from scipy import special

tf.disable_v2_behavior()

top = 100
dim_delay = Integer(low=1, high=50, name='lr')
dim_num_dense_layers11 = Integer(low=1, high=top, name='node_layer11')
dim_num_dense_layers12 = Integer(low=1, high=top, name='node_layer12')
dim_num_dense_layers13 = Integer(low=1, high=top, name='node_layer13')
dim_num_dense_layers21 = Integer(low=1, high=top, name='node_layer21')
dim_num_dense_layers22 = Integer(low=1, high=top, name='node_layer22')
dim_num_dense_layers23 = Integer(low=1, high=top, name='node_layer23')
# dim_batch_size = Integer(low=200, high=200, name='batch_size')

dimensions = [dim_delay,
              dim_num_dense_layers11,
              dim_num_dense_layers12,
              dim_num_dense_layers13,
              dim_num_dense_layers21,
              dim_num_dense_layers22,
              dim_num_dense_layers23,
              # dim_batch_size
              ]

default_parameters = [20, top, top, top, top, top, top]

Fiber = 'LEAF_6x80km_DP-'
Modulation = '64QAM_'
QAM = 64
MATLAB_file_name = Fiber + Modulation + 'P=3.0dBm.mat'
print(MATLAB_file_name)
MATLAB_refs = sio.loadmat(MATLAB_file_name, squeeze_me=True)
In_raw_complex = (MATLAB_refs['train_in_unnorm'])  # Baseband SSFM input
Out_raw_complex = (MATLAB_refs['train_des'])  # Baseband SSFM input
x_pol_in_raw_complex = In_raw_complex[:, 0]
y_pol_in_raw_complex = In_raw_complex[:, 1]
x_pol_des_raw_complex = Out_raw_complex[:, 0]
y_pol_des_raw_complex = Out_raw_complex[:, 1]
Params_ch = MATLAB_refs['Params_ch']

Max_dist = np.amax(np.abs(x_pol_in_raw_complex))
dist_contelation = Max_dist / ((np.sqrt(QAM) - 1) * np.sqrt(2))
# Divide the raw data between train and test datasets
length_raw_complex = int(len(x_pol_in_raw_complex))
train_data_margin = 0.9  # The margin of the raw data sent to the training dataset
guard_band = 10 ** 3
train_range = range(guard_band, int(length_raw_complex * train_data_margin - guard_band / 2.))
test_range = range(int(length_raw_complex * train_data_margin + guard_band / 2.),
                   length_raw_complex - guard_band)

x_pol_in_train_complex = x_pol_in_raw_complex[train_range]
y_pol_in_train_complex = y_pol_in_raw_complex[train_range]
x_pol_des_train_complex = x_pol_des_raw_complex[train_range]
y_pol_des_train_complex = y_pol_des_raw_complex[train_range]

x_pol_in_test_complex = x_pol_in_raw_complex[test_range]
y_pol_in_test_complex = y_pol_in_raw_complex[test_range]
x_pol_des_test_complex = x_pol_des_raw_complex[test_range]
y_pol_des_test_complex = y_pol_des_raw_complex[test_range]


@use_named_args(dimensions=dimensions)
def fitness(lr, node_layer11, node_layer12, node_layer13,
            node_layer21, node_layer22, node_layer23):
    # start the model making process and create our first layer
    tf.reset_default_graph()  # Erase all the info from TF models
    # print('################################## Neurons number - ', neurons)
    learning_rate = 1e-3
    seedd = 1234
    training_epochs = 5000
    batch_size = 1000
    n_taps = lr  # Number of forward and backward taps
    n_sym = 2 * n_taps + 1
    Best_BER_NN = 1

    # Construct NMSE estimator
    def NMSE_est(x, x_ref):
        return 20. * np.log10(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))

    def BER_est(x, x_ref):
        QAM_order = QAM
        return BER_calc.QAM_BER_gray(x, x_ref, QAM_order)

    def ph_norm(x, x_ref):
        nl_shift = np.dot(np.transpose(np.conjugate(x_ref)), x_ref) / np.dot(np.transpose(np.conjugate(x_ref)), x)
        return x * nl_shift, nl_shift

    # Define dataset constructor out of input raw data
    def create_dataset(x_pol_in_complex, y_pol_in_complex, x_pol_des_complex, n_sym1, batch_size1):

        raw_size = x_pol_in_complex.shape[0]
        dataset_size = raw_size - 2 * n_sym1
        dataset_range = n_sym1 + np.arange(dataset_size)

        dataset_x_pol_des = x_pol_des_complex[dataset_range]

        dataset_x_pol_in = np.empty([dataset_size, n_sym1], dtype='complex128')
        dataset_x_pol_in[:] = np.nan + 1j * np.nan

        dataset_y_pol_in = np.empty([dataset_size, n_sym1], dtype='complex128')
        dataset_y_pol_in[:] = np.nan + 1j * np.nan

        bnd_vec = int(np.floor(n_sym1 / 2))
        for vec_idx, center_vec in enumerate(dataset_range):
            local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
            if np.any(local_range < 0) or np.any(local_range > raw_size):
                ValueError('Local range steps out of the data range during dataset creation!!!')
            else:
                dataset_x_pol_in[vec_idx, :] = x_pol_in_complex[local_range]
                dataset_y_pol_in[vec_idx, :] = y_pol_in_complex[local_range]

        if np.any(np.isnan(dataset_x_pol_in)) or np.any(np.isnan(dataset_y_pol_in)):
            ValueError('Dataset matrix wasn''t fully filled by data!!!')

        # Cut the excessive datapoint not fitting into batches
        batches_limit = range(int(np.floor(dataset_size / batch_size1) * batch_size1))
        dataset_x_pol_in = dataset_x_pol_in[batches_limit]
        dataset_y_pol_in = dataset_y_pol_in[batches_limit]
        dataset_x_pol_des = dataset_x_pol_des[batches_limit]
        dataset_x_pol_in, dataset_y_pol_in, dataset_x_pol_des = shuffle(dataset_x_pol_in, dataset_y_pol_in,
                                                                        dataset_x_pol_des)
        return dataset_x_pol_in, dataset_y_pol_in, dataset_x_pol_des

    # Create the testing dataset
    dataset_x_pol_in_test, dataset_y_pol_in_test, dataset_x_pol_des_test = create_dataset(x_pol_in_test_complex,
                                                                                          y_pol_in_test_complex,
                                                                                          x_pol_des_test_complex, n_sym,
                                                                                          batch_size)
    # Create the train dataset
    dataset_x_pol_in_train, dataset_y_pol_in_train, dataset_x_pol_des_train = create_dataset(x_pol_in_train_complex,
                                                                                             y_pol_in_train_complex,
                                                                                             x_pol_des_train_complex,
                                                                                             n_sym, batch_size)

    # tf Graph input
    tf_x_pol_in = tf.placeholder(tf.complex128, [None, n_sym])
    tf_y_pol_in = tf.placeholder(tf.complex128, [None, n_sym])
    tf_x_pol_des = tf.placeholder(tf.complex128, [None, 1])
    # Y2 = tf.placeholder(tf.complex128, [None, 1])
    weight_init_mean = 0
    weight_init_std = 0.1
    # Store layers weight & bias
    tf.set_random_seed(seedd)
    neurons_l11 = node_layer11  # number of neurons layer 11
    neurons_l12 = node_layer12  # number of neurons layer 12
    neurons_l13 = node_layer13  # number of neurons layer 12
    neurons_l21 = node_layer21  # number of neurons layer 21
    neurons_l22 = node_layer22  # number of neurons layer 22
    neurons_l23 = node_layer23  # number of neurons layer 22

    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                              dtype=tf.dtypes.float64)),
        'h2': tf.Variable(tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                              dtype=tf.dtypes.float64)),
        'h3': tf.Variable(
            tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h4': tf.Variable(
            tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h5': tf.Variable(
            tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h6': tf.Variable(
            tf.truncated_normal([n_sym, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h7': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l11], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h8': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l11], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h9': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l11], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h10': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l11], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h11': tf.Variable(
            tf.truncated_normal([neurons_l11, neurons_l12], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h12': tf.Variable(
            tf.truncated_normal([neurons_l11, neurons_l12], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h13': tf.Variable(
            tf.truncated_normal([neurons_l12, neurons_l13], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h14': tf.Variable(
            tf.truncated_normal([neurons_l12, neurons_l13], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h15': tf.Variable(
            tf.truncated_normal([neurons_l13, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h16': tf.Variable(
            tf.truncated_normal([neurons_l13, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h17': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l21], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h18': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l21], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h19': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l21], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h20': tf.Variable(
            tf.truncated_normal([n_sym, neurons_l21], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h21': tf.Variable(
            tf.truncated_normal([neurons_l21, neurons_l22], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h22': tf.Variable(
            tf.truncated_normal([neurons_l21, neurons_l22], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h23': tf.Variable(
            tf.truncated_normal([neurons_l22, neurons_l23], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h24': tf.Variable(
            tf.truncated_normal([neurons_l22, neurons_l23], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h25': tf.Variable(
            tf.truncated_normal([neurons_l23, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),
        'h26': tf.Variable(
            tf.truncated_normal([neurons_l23, 1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                dtype=tf.dtypes.float64)),

    }
    weights_complex = {
        'w1': tf.complex(weights['h1'], weights['h2']),
        'w2': tf.complex(weights['h3'], weights['h4']),
        'w3': tf.complex(weights['h5'], weights['h6']),
        'w4': tf.complex(weights['h7'], weights['h8']),
        'w5': tf.complex(weights['h9'], weights['h10']),
        'w6': tf.complex(weights['h11'], weights['h12']),
        'w7': tf.complex(weights['h13'], weights['h14']),
        'w8': tf.complex(weights['h15'], weights['h16']),
        'w9': tf.complex(weights['h17'], weights['h18']),
        'w10': tf.complex(weights['h19'], weights['h20']),
        'w11': tf.complex(weights['h21'], weights['h22']),
        'w12': tf.complex(weights['h23'], weights['h24']),
        'w13': tf.complex(weights['h25'], weights['h26']),
    }
    x_gamma_accum_r = tf.Variable(-
                                  tf.truncated_normal([1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                                      dtype=tf.dtypes.float64))
    x_gamma_accum_i = tf.Variable(-
                                  tf.truncated_normal([1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                                      dtype=tf.dtypes.float64))
    y_gamma_accum_r = tf.Variable(-
                                  tf.truncated_normal([1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                                      dtype=tf.dtypes.float64))
    y_gamma_accum_i = tf.Variable(-
                                  tf.truncated_normal([1], seed=seedd, mean=weight_init_mean, stddev=weight_init_std,
                                                      dtype=tf.dtypes.float64))

    # Define model
    def multilayer_perceptron(x_pol_tapped, y_pol_tapped):
        # Estimates nonlinear distortion made onto X-pol by X- and Y-pols
        def act_cust(x):
            return tf.nn.tanh(x)

        log_x_pol_adap_tap = tf.math.log(tf.matmul(x_pol_tapped, weights_complex['w1']))

        abs_sq_x_pol_adap_tap = tf.pow(tf.math.abs(tf.matmul(x_pol_tapped, weights_complex['w2'])), 2)
        abs_sq_y_pol_adap_tap = tf.pow(tf.math.abs((tf.matmul(y_pol_tapped, weights_complex['w3']))), 2)

        x_pol_nl_ph_rot = tf.multiply(tf.cast(abs_sq_x_pol_adap_tap, tf.complex128),
                                      tf.cast(tf.complex(x_gamma_accum_r, x_gamma_accum_i), tf.complex128))
        y_pol_nl_ph_rot = tf.multiply(tf.cast(abs_sq_y_pol_adap_tap, tf.complex128),
                                      tf.cast(tf.complex(y_gamma_accum_r, y_gamma_accum_i), tf.complex128))

        layer_21_linear = tf.matmul(x_pol_tapped, weights_complex['w4']) + tf.matmul(y_pol_tapped,
                                                                                     weights_complex['w5'])
        layer_21_real = act_cust(tf.math.real(layer_21_linear))
        layer_21_imag = act_cust(tf.math.imag(layer_21_linear))
        layer_21 = tf.complex(layer_21_real, layer_21_imag)

        layer_22_linear = tf.matmul(layer_21, weights_complex['w6'])
        layer_22_real = act_cust(tf.math.real(layer_22_linear))
        layer_22_imag = act_cust(tf.math.imag(layer_22_linear))
        layer_22 = tf.complex(layer_22_real, layer_22_imag)

        layer_23_linear = tf.matmul(layer_22, weights_complex['w7'])
        layer_23_real = (act_cust(tf.math.real(layer_23_linear)))
        layer_23_imag = (act_cust(tf.math.imag(layer_23_linear)))
        Xi = tf.complex(layer_23_real, layer_23_imag)

        layer_11_linear = tf.matmul(x_pol_tapped, weights_complex['w9']) + tf.matmul(y_pol_tapped,
                                                                                     weights_complex['w10'])
        layer_11_real = act_cust(tf.math.real(layer_11_linear))
        layer_11_imag = act_cust(tf.math.imag(layer_11_linear))
        layer_11 = tf.complex(layer_11_real, layer_11_imag)
        layer_12_linear = tf.matmul(layer_11, weights_complex['w11'])
        layer_12_real = act_cust(tf.math.real(layer_12_linear))
        layer_12_imag = act_cust(tf.math.imag(layer_12_linear))
        layer_12 = tf.complex(layer_12_real, layer_12_imag)

        layer_13_linear = tf.matmul(layer_12, weights_complex['w12'])
        layer_13_real = (act_cust(tf.math.real(layer_13_linear)))
        layer_13_imag = (act_cust(tf.math.imag(layer_13_linear)))
        Theta = tf.complex(layer_13_real, layer_13_imag)

        # Output
        out_layer1 = tf.exp(log_x_pol_adap_tap + x_pol_nl_ph_rot + y_pol_nl_ph_rot + tf.matmul(Theta, weights_complex[
            'w13'])) + tf.matmul(
            Xi, weights_complex['w8'])
        return out_layer1

    # Construct model
    tf_x_pol_pred = multilayer_perceptron(tf_x_pol_in, tf_y_pol_in)
    # Define loss and optimizer
    MSE = tf.pow(tf.abs(tf_x_pol_pred - tf_x_pol_des), 2)
    loss_op = tf.math.reduce_mean(MSE)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # NMSE estimation
        NMSE_train_noNN_norm = np.zeros(training_epochs)
        NMSE_train_NN_norm = np.zeros(training_epochs)
        NMSE_train_NN = np.zeros(training_epochs)

        NMSE_test_noNN_norm = np.zeros(training_epochs)
        NMSE_test_NN_norm = np.zeros(training_epochs)
        NMSE_test_NN = np.zeros(training_epochs)

        # BER estimation
        BER_train_noNN_norm = np.zeros(training_epochs)
        BER_train_NN_norm = np.zeros(training_epochs)
        BER_train_NN = np.zeros(training_epochs)

        BER_test_noNN_norm = np.zeros(training_epochs)
        BER_test_NN_norm = np.zeros(training_epochs)
        BER_test_NN = np.zeros(training_epochs)

        # Training cycle
        for epoch_idx in range(training_epochs):

            # Train the NN on batches
            dataset_length_train = np.arange(2 ** 18)
            dataset_length_test = np.arange(int(0.2 * 2 ** 18))
            batch_num_train = int(np.floor(len(dataset_length_train) / batch_size))

            dataset_x_pol_pred_train = np.empty(batch_num_train * batch_size, dtype='complex128')
            dataset_x_pol_pred_train[:] = np.nan + 1j * np.nan

            dataset_x_pol_in_train, dataset_y_pol_in_train, dataset_x_pol_des_train = shuffle(dataset_x_pol_in_train,
                                                                                              dataset_y_pol_in_train,
                                                                                              dataset_x_pol_des_train)
            dataset_x_pol_in_trainn = dataset_x_pol_in_train[dataset_length_train]
            dataset_y_pol_in_trainn = dataset_y_pol_in_train[dataset_length_train]
            dataset_x_pol_des_trainn = dataset_x_pol_des_train[dataset_length_train]

            dataset_x_pol_in_testt = dataset_x_pol_in_test[dataset_length_test]
            dataset_y_pol_in_testt = dataset_y_pol_in_test[dataset_length_test]

            for batch_idx in range(batch_num_train):
                batch_range = batch_idx * batch_size + np.arange(batch_size)
                batch_x_pol_in_train = dataset_x_pol_in_trainn[batch_range]
                batch_y_pol_in_train = dataset_y_pol_in_trainn[batch_range]
                batch_x_pol_des_train = dataset_x_pol_des_trainn[batch_range]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, _, batch_x_pol_pred_train = sess.run([train_op, loss_op, tf_x_pol_pred],
                                                        feed_dict={tf_x_pol_in: batch_x_pol_in_train,
                                                                   tf_y_pol_in: batch_y_pol_in_train,
                                                                   tf_x_pol_des: batch_x_pol_des_train.reshape(-1, 1)
                                                                   })
                batch_x_pol_pred_train = np.squeeze(batch_x_pol_pred_train)
                dataset_x_pol_pred_train[batch_range] = batch_x_pol_pred_train

            if np.any(np.isnan(dataset_x_pol_pred_train)):
                ValueError('Training prediction vector wasn''t fully filled!!!')

            # Estimate the NN on batches
            batch_num_test = int(np.floor(len(dataset_length_test) / batch_size))
            dataset_x_pol_pred_test = np.zeros(batch_num_test * batch_size, dtype='complex128')
            dataset_x_pol_pred_test[:] = np.nan + 1j * np.nan

            for batch_idx in range(batch_num_test):
                batch_range = batch_idx * batch_size + np.arange(batch_size)
                batch_x_pol_in_test = dataset_x_pol_in_testt[batch_range]
                batch_y_pol_in_test = dataset_y_pol_in_testt[batch_range]

                # Run optimization op (backprop) and cost op (to get loss value)
                batch_x_pol_pred_test, best_w = sess.run([tf_x_pol_pred, weights_complex],
                                                         feed_dict={tf_x_pol_in: batch_x_pol_in_test,
                                                                    tf_y_pol_in: batch_y_pol_in_test
                                                                    })

                batch_x_pol_pred_test = np.squeeze(batch_x_pol_pred_test)
                dataset_x_pol_pred_test[batch_range] = batch_x_pol_pred_test

            if np.any(np.isnan(dataset_x_pol_pred_test)):
                ValueError('Testing prediction vector wasn''t fully filled!!!')

            x_pol_norm_noNN_train, _ = ph_norm(
                dataset_x_pol_in_trainn[np.arange(len(dataset_x_pol_pred_train)), n_taps],
                dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])
            x_pol_norm_noNN_test, _ = ph_norm(dataset_x_pol_in_testt[np.arange(len(dataset_x_pol_pred_test)), n_taps],
                                              dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])
            x_pol_norm_NN_train, _ = ph_norm(dataset_x_pol_pred_train,
                                             dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])
            x_pol_norm_NN_test, _ = ph_norm(dataset_x_pol_pred_test,
                                            dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])

            # NMSE estimation
            NMSE_train_noNN_norm[epoch_idx] = NMSE_est(x_pol_norm_noNN_train, dataset_x_pol_des_train[
                np.arange(len(dataset_x_pol_pred_train))])
            NMSE_train_NN_norm[epoch_idx] = NMSE_est(x_pol_norm_NN_train,
                                                     dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])
            NMSE_train_NN[epoch_idx] = NMSE_est(dataset_x_pol_pred_train,
                                                dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])

            NMSE_test_noNN_norm[epoch_idx] = NMSE_est(x_pol_norm_noNN_test,
                                                      dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])
            NMSE_test_NN_norm[epoch_idx] = NMSE_est(x_pol_norm_NN_test,
                                                    dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])
            NMSE_test_NN[epoch_idx] = NMSE_est(dataset_x_pol_pred_test,
                                               dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])

            # BER estimation
            BER_train_noNN_norm[epoch_idx] = BER_est(x_pol_norm_noNN_train,
                                                     dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])
            BER_train_NN_norm[epoch_idx] = BER_est(x_pol_norm_NN_train,
                                                   dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])
            BER_train_NN[epoch_idx] = BER_est(dataset_x_pol_pred_train,
                                              dataset_x_pol_des_train[np.arange(len(dataset_x_pol_pred_train))])

            BER_test_noNN_norm[epoch_idx] = BER_est(x_pol_norm_noNN_test,
                                                    dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])
            BER_test_NN_norm[epoch_idx] = BER_est(x_pol_norm_NN_test,
                                                  dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])
            BER_test_NN[epoch_idx] = BER_est(dataset_x_pol_pred_test,
                                             dataset_x_pol_des_test[np.arange(len(dataset_x_pol_pred_test))])

            if BER_test_NN[epoch_idx] < Best_BER_NN:
                Best_BER_NN = BER_test_NN[epoch_idx]
                Best_Epoch_NN = epoch_idx
                Best_NMSE_NN = NMSE_test_NN[epoch_idx]
                Best_BER_no_NN = BER_test_noNN_norm[epoch_idx]
                Best_NMSE_no_NN = NMSE_test_noNN_norm[epoch_idx]
            elif BER_test_NN[epoch_idx] == Best_BER_NN:
                if NMSE_test_NN[epoch_idx] < Best_NMSE_NN:
                    Best_BER_NN = BER_test_NN[epoch_idx]
                    Best_Epoch_NN = epoch_idx
                    Best_NMSE_NN = NMSE_test_NN[epoch_idx]
                    Best_BER_no_NN = BER_test_noNN_norm[epoch_idx]
                    Best_NMSE_no_NN = NMSE_test_noNN_norm[epoch_idx]
            optt = (Best_BER_no_NN - Best_BER_NN) * 100 / Best_BER_no_NN
            print('##################  Improvement', np.round(optt, 3), ' % ########################')

    optt = (Best_BER_no_NN - Best_BER_NN) * 100 / Best_BER_no_NN
    sess.close()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    if optt > 0:
        Q_prop_ref2 = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * Best_BER_NN))
        print('######################################################################', dt_string)
        print('##################  Improvement', np.round(optt, 3), ' % ########################')
        print("NMSE x  with NN", Best_NMSE_NN, "BER x  with NN", Best_BER_NN, "NMSE x  without NN", Best_NMSE_no_NN,
              "BER x  without NN", Best_BER_no_NN, "best Epoch", Best_Epoch_NN)
        print("delay=  ", n_taps, "n11= ",
              node_layer11, "n12 =", node_layer12, "n13 =", node_layer13, "n21 =", node_layer21, "n22  =", node_layer22,
              "n23  =", node_layer23)
        print('Best Qfactor', Q_prop_ref2)
        print('######################################################################')

    else:
        print('######################################################################', dt_string)
        print('##################  NO Improvement ')
        print('######################################################################')

    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -optt


gp_result = gp_minimize(func=fitness,
                        dimensions=dimensions,
                        n_calls=50,
                        noise=0.01,
                        n_jobs=-1,
                        kappa=5,
                        x0=default_parameters)
