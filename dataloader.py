import numpy as np
import pandas as pd
import features
import pickle

'''

'''
class DataLoader():
    def __init__(self, file_path, batch_size, label_column_name, use_extracted_feature, is_training = True):
        '''

        :param file_path: file path for track_metadata.csv
        :param batch_size:
        :param label_column_name: column name of label (project 1: track_genre_top, project 2: listens)
        :param is_training: training / validation mode
        '''
        self.batch_size = batch_size
        self.token_stream = []
        self.file_path = file_path
        self.is_training = is_training
        self.label_column_name = label_column_name
        self.use_extracted_feature = use_extracted_feature
        self.create_batches()

    def create_batches(self):
        '''

        :return:
        '''
        self.metadata_df = pd.read_csv(self.file_path)
        
        if self.is_training:
            mode = 'training'
        else:
            mode = 'validation'


        if self.use_extracted_feature:
            with open('{}_{}.pkl'.format(self.use_extracted_feature, mode), 'rb') as f:
                self.extracted_tid_feature_pairs = pickle.load(f)
            tids, batch_x = zip(*self.extracted_tid_feature_pairs)
            self.metadata_df = self.metadata_df[self.metadata_df['track_id'].isin(tids)]
            print(len(tids))
            assert len(tids) == len(self.metadata_df.index)
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['set_split'] == mode]
                                
        self.pointer = 0
        self.num_batch = int(len(self.metadata_df) / self.batch_size)
        self.label_dict = {k: v for v,k in enumerate(set(self.metadata_df[self.label_column_name].values))}

    def next_batch(self):
        '''

        :return: feature array, label array (one-hot encoded)
        '''
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size
        end_pos = (start_pos+self.batch_size)
        
        meta_df = self.metadata_df.iloc[start_pos:end_pos]
        # TODO: load features
       
        if self.use_extracted_feature:
            tids, batch_x = zip(*self.extracted_tid_feature_pairs[start_pos:end_pos])
            assert meta_df['track_id'].values.tolist() == list(tids)
            return list(batch_x), self.convert_labels(meta_df)
        else:
            track_ids = meta_df['track_id'].values
            return features.compute_mfcc_example(track_ids), self.convertLabels(meta_df)

    def reset_pointer(self):
        self.pointer = 0

    def convert_labels(self, meta_df):
        '''

        :param meta_df: metadata (as pandas DataFrame)
        :return: numpy array with (batch_size, number of genres) shape. one-hot encoded
        '''
        # create one-hot encoded array
        label_array = np.zeros((len(meta_df.index), len(self.label_dict)))
        labels = meta_df[self.label_column_name].values
        for i, label in enumerate(labels):
            label_pos = self.label_dict.get(label)
            label_array[i, label_pos] = 1
        return label_array


if __name__ == "__main__":
    # for test
    training_loader = DataLoader('dataset/track_metadata.csv', 32, 'track_genre_top', is_training=True)
    valid_loader = DataLoader('dataset/track_metadata.csv', 32, 'track_genre_top', is_training=False)

    for _ in range(training_loader.num_batch):
        track_ids, label_onehot = training_loader.next_batch()

    for _ in range(valid_loader.num_batch):
        track_ids, label_onehot = valid_loader.next_batch()
        print(track_ids, label_onehot)



