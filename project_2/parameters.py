JSON_SEPARATORS = (',', ': ')

import tensorflow as tf

class Parameters:
    class Attention:
        class Experiment1:
            #conditions
            BATCH_SIZE = 30
            LEARNING_RATE = 0.00001
            LABEL_COLUMN_NAME = 'listens'
            N_CLASS = 3
            EPOCH_NUM = 10
            WINDOW_NUM = 10

            Regularizer = tf.contrib.layers.l2_regularizer(LEARNING_RATE)
            Activation = tf.nn.relu
            Initializer = tf.contrib.layers.xavier_initializer(LEARNING_RATE)
            Optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            LOSS_FUNCTION = tf.losses.softmax_cross_entropy

            # Parameters
            RNN_TYPE = tf.contrib.rnn.BasicLSTMCell
            EMBEDDING_DIMENSION = 128
            ATTENTION_LENGTH = 10
            NUM_RNN_LAYERS = 2

            NUM_FC_HIDDEN_UNITS = 512

            def __init__(self):
                pass

        def __init__(self):
            pass
