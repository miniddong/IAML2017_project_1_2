import os, sys, inspect
import numpy as np
import tensorflow as tf
from time import localtime, time
from tensorflow.python.platform import gfile

def get_parent_directory_path():
    return os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    )

def add_directory_to_sys_path(directory_path):
    return sys.path.insert(-1, directory_path)

add_directory_to_sys_path(get_parent_directory_path())
from dataloader import DataLoader

def get_numbers_of_batch(dataloader):
    return dataloader.num_batch

def load_dataloader(METADATA_PATH, BATCH_SIZE, LABEL_COLUMN_NAME, EXTRACTED_FEATURE_NAME=None, is_training=False):
    return DataLoader(file_path=METADATA_PATH
                      , batch_size=BATCH_SIZE
                      , label_column_name=LABEL_COLUMN_NAME
                      , is_training=is_training
                      , use_extracted_feature=EXTRACTED_FEATURE_NAME
                      )


def get_batch(train_dataloader):
    return train_dataloader.next_batch()


def chunk_batch_X(batch_X, chunk_size, timestep):
    return list(
        map(
            lambda X: X[:, timestep * chunk_size:(timestep + 1) * chunk_size]
            , batch_X
        )
    )


def split_X(batch_X, split_indices):
    """
    params @batch_X [a list of ndarray] (length : batch_size)
    params @split_indices [a list of integer]

    """

    def debug():
        def test(X):
            return list(map(lambda X: X.shape, np.hsplit(X, split_indices)))

        print(list(map(lambda X: test(X), batch_X)))

    # debug()

    return list(map(lambda X: np.hsplit(X, split_indices), batch_X))


def get_split_indices(IMAGE_WIDTH, WINDOW_NUM):
    return list(map(lambda i: i * int(IMAGE_WIDTH / WINDOW_NUM), range(1, WINDOW_NUM + 1)))


def get_current_chunk(batch_X_chunks, selected_window_index):
    return list(
        map(
            lambda X_chunks: X_chunks[selected_window_index]
            , batch_X_chunks
        )
    )


def get_context_chunk(batch_X_chunks, selected_window_index, context_window_size=1):
    """
    params @batch_X_chunks[a list of a list of ndarrays]
    params @selected_window
    """

    def _index_clamp_to_positive(index):
        return 0 if index < 0 else index

    def _get_before_chunk_indices():
        return _index_clamp_to_positive(selected_window_index - context_window_size), selected_window_index

    def _get_after_chunk_indices():
        return selected_window_index + 1, selected_window_index + 1 + context_window_size

    return list(
        map(
            lambda X_chunks: X_chunks[_get_before_chunk_indices()[0]:_get_before_chunk_indices()[1]] \
                             + X_chunks[_get_after_chunk_indices()[0]:_get_after_chunk_indices()[1]]
            , batch_X_chunks
        )
    )

def get_batch_predictions(batch_votings, labels):
    return np.equal(np.argmax(batch_votings, 1), np.argmax(labels, 1))

def add_votings(batch_votings, prediction_indices):
    batch_votings[np.arange(prediction_indices.shape[0]), prediction_indices] += 1
    return batch_votings

def calculate_accuracy(batch_predictions):
    return np.mean(batch_predictions)

def initialize_batch_votings(BATCH_SIZE, N_CLASS):
    return np.zeros([BATCH_SIZE, N_CLASS])

# save model
def set_saver():
    return tf.train.Saver()

def set_output_directory_path(CHECKPOINT_PATH):
    output_directory_path = CHECKPOINT_PATH + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
    create_output_directory_if_not_exist(output_directory_path)
    return output_directory_path

def create_output_directory_if_not_exist(output_directory_path):
    if not gfile.Exists(output_directory_path):
        gfile.MakeDirs(output_directory_path)

def save_model(saver, sess, output_directory_path):
    return saver.save(sess, output_directory_path)
