import tensorflow as tf

def set_X(IMAGE_HEIGHT, window_size):
    return tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, window_size])

def set_window_size(IMAGE_WIDTH, WINDOW_NUM):
    return int(IMAGE_WIDTH / WINDOW_NUM)


def set_y(N_CLASS):
    return tf.placeholder(tf.float32, [None, N_CLASS])


def unstack_X_by_timestep(X, window_size):
    return tf.unstack(X, num=window_size, axis=2)


def set_rnn_cell(RNN_TYPE, EMBEDDING_DIMENSION):
    return RNN_TYPE(EMBEDDING_DIMENSION, state_is_tuple=False)


def wrap_attention(rnn_cell, attention_length):
    return tf.contrib.rnn.AttentionCellWrapper(
        rnn_cell, attention_length
        , state_is_tuple=False
    )

def stack_rnn_cells(make_rnn_cell, num_layers):
    return list(map(
        lambda i: make_rnn_cell
        , range(num_layers)
    ))


def layer_rnn_cells(stacked_rnn_cells):
    return tf.nn.rnn_cell.MultiRNNCell(
        stacked_rnn_cells
        , state_is_tuple=False
    )


def run_rnn(rnn_cells, X):
    return tf.contrib.rnn.static_rnn(
        rnn_cells, X, dtype=tf.float32
    )

def run_bidirectional_rnn(forward_rnn_cells, backward_rnn_cells, X):
    return tf.contrib.rnn.stack_bidirectional_rnn(
        forward_rnn_cells
        , backward_rnn_cells
        , X
        , dtype=tf.float32
    )


def set_fully_connected_layer(X, num_hidden_units, activation):
    return tf.layers.dense(X, num_hidden_units, activation=activation)


def get_logits(X, N_CLASS):
    return tf.layers.dense(X, N_CLASS)


def calculate_losses(LOSS_FUNCTION, labels, logits):
    return LOSS_FUNCTION(labels, logits)


def calculate_mean_loss(losses):
    return tf.reduce_mean(losses)


def set_train_step(Optimizer, target_loss):
    return Optimizer.minimize(target_loss)


def build_attention_model(data_constant, experiment_parameters):
    X = set_X(
        data_constant.IMAGE_HEIGHT
        , set_window_size(
            data_constant.IMAGE_WIDTH
            , experiment_parameters.WINDOW_NUM
        )
    )

    y = set_y(experiment_parameters.N_CLASS)

    # unstack X
    unstacked_X = unstack_X_by_timestep(
        X
        , set_window_size(
            data_constant.IMAGE_WIDTH
            , experiment_parameters.WINDOW_NUM
        )
    )

    # stack rnn cells
    stacked_rnn_cells = []
    for i in range(experiment_parameters.NUM_RNN_LAYERS):
        stacked_rnn_cells.append(
            wrap_attention(
                set_rnn_cell(
                    experiment_parameters.RNN_TYPE
                    , experiment_parameters.EMBEDDING_DIMENSION
                )
                , experiment_parameters.ATTENTION_LENGTH
            )
        )

    # run rnn
    _, encoding = run_rnn(
        layer_rnn_cells(stacked_rnn_cells)
        , unstacked_X
    )

    # classification
    logits = get_logits(
        set_fully_connected_layer(
            encoding
            , experiment_parameters.NUM_FC_HIDDEN_UNITS
            , experiment_parameters.Activation
        )
        , experiment_parameters.N_CLASS
    )

    # loss
    mean_loss = calculate_mean_loss(
        calculate_losses(experiment_parameters.LOSS_FUNCTION
                         , labels=y, logits=logits)
    )
    train_step = set_train_step(experiment_parameters.Optimizer, mean_loss)

    # accuracy
    prediction_indices = tf.argmax(logits, 1)
    # accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

    return X, y, logits, mean_loss, train_step, prediction_indices

def initialize_variables(sess):
    return sess.run(tf.global_variables_initializer())

def initialize_graph():
    return tf.Graph().as_default()

def initialize_session():
    return tf.Session()

def build_bidirectional_attention_rnn_model(data_constant, experiment_parameters):
    X = set_X(
        data_constant.IMAGE_HEIGHT
        , set_window_size(
            data_constant.IMAGE_WIDTH
            , experiment_parameters.WINDOW_NUM
        )
    )

    y = set_y(experiment_parameters.N_CLASS)

    # unstack X
    unstacked_X = unstack_X_by_timestep(
        X
        , set_window_size(
            data_constant.IMAGE_WIDTH
            , experiment_parameters.WINDOW_NUM
        )
    )

    # stack rnn cells
    forward_rnn_cells = []
    for i in range(experiment_parameters.NUM_RNN_LAYERS):
        forward_rnn_cells.append(
            wrap_attention(
                set_rnn_cell(
                    experiment_parameters.RNN_TYPE
                    , experiment_parameters.EMBEDDING_DIMENSION
                )
                , experiment_parameters.ATTENTION_LENGTH
            )
        )

    backward_rnn_cells = []
    for i in range(experiment_parameters.NUM_RNN_LAYERS):
        backward_rnn_cells.append(
            wrap_attention(
                set_rnn_cell(
                    experiment_parameters.RNN_TYPE
                    , experiment_parameters.EMBEDDING_DIMENSION
                )
                , experiment_parameters.ATTENTION_LENGTH
            )
        )

    # run rnn
    _, encoding = run_bidirectional_rnn(
        layer_rnn_cells(forward_rnn_cells)
        , layer_rnn_cells(backward_rnn_cells)
        , unstacked_X
    )

    # classification
    logits = get_logits(
        set_fully_connected_layer(
            encoding
            , experiment_parameters.NUM_FC_HIDDEN_UNITS
            , experiment_parameters.Activation
        )
        , experiment_parameters.N_CLASS
    )

    # loss
    mean_loss = calculate_mean_loss(
        calculate_losses(experiment_parameters.LOSS_FUNCTION
                         , labels=y, logits=logits)
    )
    train_step = set_train_step(experiment_parameters.Optimizer, mean_loss)

    # accuracy
    prediction_indices = tf.argmax(logits, 1)
    # accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

    return X, y, logits, mean_loss, train_step, prediction_indices
