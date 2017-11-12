from dataloader import DataLoader
import tensorflow as tf

def set_X(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS):
    return tf.placeholder([None, IMAGE_HEIGHT, int(IMAGE_WIDTH/NUM_STEPS)])

def set_y(N_CLASS):
    return tf.placeholder([None, N_CLASS])

def lstm(LSTM_SIZE):
    return tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)

def initialize_state(BATCH_SIZE, state_size):
    return tf.zeros([BATCH_SIZE, state_size])

def initialize_hidden_and_current_state(BATCH_SIZE, state_size):
    return initialize_state(BATCH_SIZE, state_size)\
        , initialize_state(BATCH_SIZE, state_size)

