JSON_SEPARATORS = (',', ': ')

import tensorflow as tf

class Constant:
    class Project1:
        METADATA_PATH = 'dataset/track_metadata.csv'
        LABEL_COLUMN_NAME = 'track_genre_top'
        SAVE_CHECKPOINT_PATH = 'checkpoint'

        def __init__(self):
            pass

class Data:
    class ChormaStftHop4096:
        FEATURE_NAME = 'chroma_stft'
        IMAGE_HEIGHT = 20
        IMAGE_WIDTH = 313

        def __init__(self):
            pass

    def __init__(self):
        pass


class AutoEncoder:
    class Experiment1:
        class Conditions:
            BATCH_SIZE = 32
            EPOCH_NUM = 100
            LEARNING_RATE = 0.00001

            Regularizer = tf.contrib.layers.l2_regularizer(LEARNING_RATE)
            Activation = tf.nn.relu
            Initializer = tf.contrib.layers.xavier_initializer(LEARNING_RATE)

            def __init__(self):
                pass

        class Parameters:
            N_CLASS = 8
            STRIDES = 1

            ENCODING_LAYER_1_FILTER_WIDTH = 5
            ENCODING_LAYER_1_FILTER_HEIGHT = 5
            ENCODING_LAYER_1_FILTER_NUMBERS = 32

            ENCODING_LAYER_2_FILTER_WIDTH = 5
            ENCODING_LAYER_2_FILTER_HEIGHT = 5
            ENCODING_LAYER_2_FILTER_NUMBERS = 64

            AFTER_ENCODING_OUTPUT_WIDTH = Data.ChormaStftHop4096.IMAGE_WIDTH + 8
            AFTER_ENCODING_OUTPUT_HEIGHT = Data.ChormaStftHop4096.IMAGE_HEIGHT + 8
            AFTER_ENCODING_OUTPUT_DEPTH = ENCODING_LAYER_2_FILTER_NUMBERS

            DECODING_LAYER_1_FILTER_WIDTH = 5
            DECODING_LAYER_1_FILTER_HEIGHT = 5
            DECODING_LAYER_1_FILTER_NUMBERS = 32

            DECODING_LAYER_2_FILTER_WIDTH = 5
            DECODING_LAYER_2_FILTER_HEIGHT = 5
            DECODING_LAYER_2_FILTER_NUMBERS = 1

            FULLY_CONNECTED_LAYER_UNITS = 128


            def __init__(self):
                pass

        def __init__(self):
            pass
    def __init__(self):
        pass
