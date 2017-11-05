import tensorflow as tf

class Graph:
    def __init__(self, ExperimentVariables, DataConstant):
        self.Parameters = ExperimentVariables.Parameters
        self.Conditions = ExperimentVariables.Conditions
        self.DataConstant = DataConstant

    def build_graph(self):
        self.X = tf.placeholder(tf.float32
                                , [None
                                    , self.DataConstant.IMAGE_HEIGHT
                                    , self.DataConstant.IMAGE_WIDTH]
                                , name="X")

        self.y = tf.placeholder(tf.float32, [None, self.Parameters.N_CLASS], name="y")

        reshape_X = tf.reshape(self.X
                               , [-1
                                   , self.DataConstant.IMAGE_HEIGHT
                                   , self.DataConstant.IMAGE_WIDTH
                                   , 1]
                               )

        encoding_layer_1 = self._build_encoding_layer(
            reshape_X
            , self.Parameters.ENCODING_LAYER_1_FILTER_NUMBERS
            , self.Parameters.ENCODING_LAYER_1_FILTER_HEIGHT
            , self.Parameters.ENCODING_LAYER_1_FILTER_WIDTH
            , self.Parameters.STRIDES
            , name="EncodingLayer1"
        )

        encoding_layer_2 = self._build_encoding_layer(
            encoding_layer_1
            , self.Parameters.ENCODING_LAYER_2_FILTER_NUMBERS
            , self.Parameters.ENCODING_LAYER_2_FILTER_HEIGHT
            , self.Parameters.ENCODING_LAYER_2_FILTER_WIDTH
            , self.Parameters.STRIDES
            , name="EncodingLayer2"
        )

        # Reconstruction
        decoding_layer_1 = self._build_decoding_layer(
            encoding_layer_2
            , self.Parameters.DECODING_LAYER_1_FILTER_NUMBERS
            , self.Parameters.DECODING_LAYER_1_FILTER_HEIGHT
            , self.Parameters.DECODING_LAYER_1_FILTER_WIDTH
            , self.Parameters.STRIDES
            , name="DecodingLayer1"
        )

        self.decoding_layer_2 = self._build_decoding_layer(
            decoding_layer_1
            , self.Parameters.DECODING_LAYER_1_FILTER_NUMBERS
            , self.Parameters.DECODING_LAYER_1_FILTER_HEIGHT
            , self.Parameters.DECODING_LAYER_1_FILTER_WIDTH
            , self.Parameters.STRIDES
            , name="DecodingLayer1"
        )

        # Classification
        flat = tf.reshape(self.encoding_layer_2, [-1
            , sOUTPUT_HEIGHT*OUTPUT_WIDTH*OUTPUT_DEPTH])
        fully_connected_layer = tf.layers.dense(flat
                                   , self.Parameters.FULLY_CONNECTED_LAYER_UNITS
                                   , activation=self.Conditions.activation)
        self.classification_logit = tf.layers.dense(fully_connected_layer, N_CLASS)

    def _build_encoding_layer(self, input_tensor, FILTER_NUMBERS, FILTER_HEIGHT, FILTER_WIDTH, STRIDES, name):
        def add_padding_for_encoding(input_tensor, FILTER_NUMBERS):
            return tf.pad(input_tensor
                          , [[0,0]
                              , [FILTER_NUMBERS-1, FILTER_NUMBERS-1]
                              , [FILTER_NUMBERS-1, FILTER_NUMBERS-1]
                              ,[0,0]]
                          )

        return tf.layers.conv2d(add_padding_for_encoding(input_tensor, FILTER_NUMBERS)
                                , FILTER_NUMBERS
                                , (FILTER_HEIGHT, FILTER_WIDTH)
                                , STRIDES
                                , padding="valid"
                                , activation=self.Conditions.Activation
                                , kernel_regularizer=self.Conditions.Regularizer
                                , kernel_initializer=self.Conditions.Initializer
                                , name=name
                                )

    def _build_decoding_layer(self, input_tensor, FILTER_NUMBERS, FILTER_HEIGHT, FILTER_WIDTH, STRIDES, name):
        return tf.layers.conv2d(input_tensor
                                , FILTER_NUMBERS
                                , (FILTER_HEIGHT, FILTER_WIDTH)
                                , STRIDES
                                , "valid"
                                , activation=self.Conditions.Activation
                                , kernel_initializer=self.Conditions.Initializer
                                , name=name
                                )

    def calculate_classification_loss(self):
        self.classification_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(
                self.y
                , self.classification_logit
            )
        )

    def calculate_reconstruction_loss(self):
        self.reconstruction_loss = tf.reduce_mean(
            tf.losses.mean_squared_error(
                self.y
                , self.decoding_layer_2
            )
        )

    def define_loss(self):
        self.mean_loss = tf.reduce_mean(
            self.classification_loss + self.reconstruction_loss
        )

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.mean_loss)

    def define_accuracy(self):
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.classification_logit, 1), tf.argmax(self.y, 1))
                , "float"
            )
        )

