import tensorflow as tf

def build_graph(model_parameters, data_constant):
    def set_X(IMAGE_HEIGHT, IMAGE_WIDTH):
        return tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name="X")

    def set_y(N_CLASS):
        return tf.placeholder(tf.float32, [None, N_CLASS], name="y")

    def reshape_X(X, IMAGE_HEIGHT, IMAGE_WIDTH):
        return tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    def build_encoding_layer(input_tensor, FILTER_NUMBERS, FILTER_HEIGHT, FILTER_WIDTH, STRIDES, name):
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
                                , activation=model_parameters.Activation
                                , kernel_regularizer=model_parameters.Regularizer
                                , kernel_initializer=model_parameters.Initializer
                                , name=name
                                )

    def encode(X, data_constant, model_parameters):
        return build_encoding_layer(
            build_encoding_layer(
                reshape_X(
                    X
                    , data_constant.IMAGE_HEIGHT
                    , data_constant.IMAGE_WIDTH
                )
                , model_parameters.ENCODING_LAYER_1_FILTER_NUMBERS
                , model_parameters.ENCODING_LAYER_1_FILTER_HEIGHT
                , model_parameters.ENCODING_LAYER_1_FILTER_WIDTH
                , model_parameters.STRIDES
                , name="EncodingLayer1"
            )
            , model_parameters.ENCODING_LAYER_2_FILTER_NUMBERS
            , model_parameters.ENCODING_LAYER_2_FILTER_HEIGHT
            , model_parameters.ENCODING_LAYER_2_FILTER_WIDTH
            , model_parameters.STRIDES
            , name="EncodingLayer2"
        )

    def build_decoding_layer(input_tensor, FILTER_NUMBERS, FILTER_HEIGHT, FILTER_WIDTH, STRIDES, name):
        return tf.layers.conv2d(input_tensor
                                , FILTER_NUMBERS
                                , (FILTER_HEIGHT, FILTER_WIDTH)
                                , STRIDES
                                , "valid"
                                , activation=model_parameters.Activation
                                , kernel_initializer=model_parameters.Initializer
                                , name=name
                                )

    def decode(after_encode_output, model_parameters):
        return build_decoding_layer(
            build_decoding_layer(
                after_encode_output
                , model_parameters.DECODING_LAYER_1_FILTER_NUMBERS
                , model_parameters.DECODING_LAYER_1_FILTER_HEIGHT
                , model_parameters.DECODING_LAYER_1_FILTER_WIDTH
                , model_parameters.STRIDES
                , name="DecodingLayer1"
            )
            , model_parameters.DECODING_LAYER_2_FILTER_NUMBERS
            , model_parameters.DECODING_LAYER_2_FILTER_HEIGHT
            , model_parameters.DECODING_LAYER_2_FILTER_WIDTH
            , model_parameters.STRIDES
            , name="DecodingLayer2"
        )

    def calculate_reconstruction_loss(y, after_encode_output, model_parameters):
        return tf.reduce_mean(
            tf.losses.mean_squared_error(
                y
                , decode(
                    after_encode_output
                    , model_parameters
                )
            )
        )

    def classify(after_encode_output, model_parameters):
        return calculate_output_logit(
            build_fully_connected_layer(
                reshape_after_encode_output(
                    after_encode_output
                    , model_parameters.AFTER_ENCODING_OUTPUT_HEIGHT
                    , model_parameters.AFTER_ENCODING_OUTPUT_WIDTH
                    , model_parameters.AFTER_ENCODING_OUTPUT_DEPTH
                )
                , model_parameters.FULLY_CONNECTED_LAYER_UNITS
                , activation=model_parameters.Activation
            )
            , model_parameters.N_CLASS
        )

    def reshape_after_encode_output(after_encode_output
                                    , OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_DEPTH):
        return tf.reshape(after_encode_output
                          , [-1, OUTPUT_HEIGHT*OUTPUT_WIDTH*OUTPUT_DEPTH])

    def build_fully_connected_layer(input_tensor, FULLY_CONNECTED_LAYER_UNITS, activation):
        return tf.layers.dense(input_tensor
                               , FULLY_CONNECTED_LAYER_UNITS
                               , activation=activation)

    def calculate_output_logit(input_tensor, N_CLASS):
        return tf.layers.dense(input_tensor, N_CLASS)


    def calculate_classification_loss(y, classification_logit):
        return tf.reduce_mean(
            tf.losses.softmax_cross_entropy(
                y
                , classification_logit
            )
        )

    def calculate_total_loss(y, after_encode_output, classification_logit, model_parameters):
        return calculate_classification_loss(y, classification_logit) \
               + calculate_reconstruction_loss(y, after_encode_output, model_parameters)

    def optimize(mean_loss):
        return tf.train.AdamOptimizer(0.0001).minimize(mean_loss)

    def calculate_accuracy(y, classification_logit):
        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(classification_logit, 1), tf.argmax(y, 1))
                , "float"
            )
        )

    X = set_X(
        data_constant.IMAGE_HEIGHT
        , data_constant.IMAGE_WIDTH
    )
    y = set_y(model_parameters.N_CLASS)

    after_encode_output = encode(
        X
        , data_constant
        , model_parameters
    )

    classification_logit = classify(after_encode_output, model_parameters)

    # Loss and optimizer
    mean_loss = tf.reduce_mean(
        calculate_total_loss(
            y
            , after_encode_output
            , classification_logit
            , model_parameters
        )
    )
    optimizer = optimize(mean_loss)
    accuracy = calculate_accuracy(y, classification_logit)

    return X, y, mean_loss, optimizer, accuracy



