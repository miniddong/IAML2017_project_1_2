import tensorflow as tf

def build_graph(ExperimentVariables, DataConstant):
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
                              , [FILTER_HEIGHT-1, FILTER_WIDTH-1]
                              , [FILTER_HEIGHT-1, FILTER_WIDTH-1]
                              ,[0,0]]
                          )

        return tf.layers.conv2d(add_padding_for_encoding(input_tensor, FILTER_NUMBERS)
                                , FILTER_NUMBERS
                                , (FILTER_HEIGHT, FILTER_WIDTH)
                                , STRIDES
                                , padding="valid"
                                , activation=ExperimentVariables.Conditions.Activation
                                , kernel_regularizer=ExperimentVariables.Conditions.Regularizer
                                , kernel_initializer=ExperimentVariables.Conditions.Initializer
                                , name=name
                                )


    def build_decoding_layer(input_tensor, FILTER_NUMBERS, FILTER_HEIGHT, FILTER_WIDTH, STRIDES, name):
        return tf.layers.conv2d(input_tensor
                                , FILTER_NUMBERS
                                , (FILTER_HEIGHT, FILTER_WIDTH)
                                , STRIDES
                                , "valid"
                                , activation=ExperimentVariables.Conditions.Activation
                                , kernel_initializer=ExperimentVariables.Conditions.Initializer
                                , name=name
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
        return tf.losses.softmax_cross_entropy(
            y, classification_logit
        )

    def calculate_reconstruction_loss(y, after_decode_output):
        return tf.losses.mean_squared_error(
            y, after_decode_output
        )

    def calculate_mean_loss(classification_loss, reconstruction_loss):
        return tf.reduce_mean(classification_loss) + tf.reduce_mean(reconstruction_loss)

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
        DataConstant.IMAGE_HEIGHT
        , DataConstant.IMAGE_WIDTH
    )

    after_reshape_X =reshape_X(
        X
        , DataConstant.IMAGE_HEIGHT
        , DataConstant.IMAGE_WIDTH
    )

    after_encode_output = build_encoding_layer(
        build_encoding_layer(
            after_reshape_X
            , ExperimentVariables.Parameters.ENCODING_LAYER_1_FILTER_NUMBERS
            , ExperimentVariables.Parameters.ENCODING_LAYER_1_FILTER_HEIGHT
            , ExperimentVariables.Parameters.ENCODING_LAYER_1_FILTER_WIDTH
            , ExperimentVariables.Parameters.STRIDES
            , name="EncodingLayer1"
        )
        , ExperimentVariables.Parameters.ENCODING_LAYER_2_FILTER_NUMBERS
        , ExperimentVariables.Parameters.ENCODING_LAYER_2_FILTER_HEIGHT
        , ExperimentVariables.Parameters.ENCODING_LAYER_2_FILTER_WIDTH
        , ExperimentVariables.Parameters.STRIDES
        , name="EncodingLayer2"
    )

    after_decode_output = build_decoding_layer(
        build_decoding_layer(
            after_encode_output
            , ExperimentVariables.Parameters.DECODING_LAYER_1_FILTER_NUMBERS
            , ExperimentVariables.Parameters.DECODING_LAYER_1_FILTER_HEIGHT
            , ExperimentVariables.Parameters.DECODING_LAYER_1_FILTER_WIDTH
            , ExperimentVariables.Parameters.STRIDES
            , name="DecodingLayer1"
        )
        , ExperimentVariables.Parameters.DECODING_LAYER_2_FILTER_NUMBERS
        , ExperimentVariables.Parameters.DECODING_LAYER_2_FILTER_HEIGHT
        , ExperimentVariables.Parameters.DECODING_LAYER_2_FILTER_WIDTH
        , ExperimentVariables.Parameters.STRIDES
        , name="DecodingLayer2"
    )

    y = set_y(ExperimentVariables.Parameters.N_CLASS)

    classification_logit = calculate_output_logit(
        build_fully_connected_layer(
            reshape_after_encode_output(
                after_encode_output
                , ExperimentVariables.Parameters.AFTER_ENCODING_OUTPUT_HEIGHT
                , ExperimentVariables.Parameters.AFTER_ENCODING_OUTPUT_WIDTH
                , ExperimentVariables.Parameters.AFTER_ENCODING_OUTPUT_DEPTH
            )
            , ExperimentVariables.Parameters.FULLY_CONNECTED_LAYER_UNITS
            , activation=ExperimentVariables.Conditions.Activation
        )
        , ExperimentVariables.Parameters.N_CLASS
    )

    classification_loss = calculate_classification_loss(y, classification_logit)
    reconstruction_loss = calculate_reconstruction_loss(y, classification_logit)
    print(X.shape)
    print(after_reshape_X.shape)
    print(after_encode_output.shape)
    print(after_decode_output.shape)
    #print(reconstruction_loss.shape)

    # Loss and optimizer
    mean_loss = calculate_mean_loss(
        classification_loss
        , reconstruction_loss
    )

    optimizer = optimize(mean_loss)
    accuracy = calculate_accuracy(y, classification_logit)

    return X, y, after_encode_output, after_decode_output, mean_loss, optimizer, accuracy



