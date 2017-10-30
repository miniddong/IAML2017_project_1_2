from time import localtime, time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import *

from utilities.experiments import Constant, AutoEncoder, Data
import autoencoder

from dataloader import DataLoader

### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)

ExperimentConditions = AutoEncoder.Experiment1
Constant = Constant.Project1
Data = Data.ChormaStftHop4096
is_train_mode = True

X, y, mean_loss_op, optimizer_op, accuracy_op = autoencoder.build_graph(
    ExperimentConditions
    , Data
)

# Train and evaluate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    PRINT_EVERY = 10


    if is_train_mode:
        train_dataloader = DataLoader(file_path='dataset/track_metadata.csv'
                                      , batch_size=ExperimentConditions.BATCH_SIZE
                                      , label_column_name=Constant.LABEL_COLUMN_NAME
                                      , use_extracted_feature=Data.FEATURE_NAME
                                      , is_training=True
                                      )

        for epoch in tnrange(ExperimentConditions.EPOCH_NUM):
            total_batch = train_dataloader.num_batch
            epoch_loss, epoch_accuracy = [0, 0]
            batch_losses, batch_accuracies = [[], []]

            for i in range(total_batch):
                batch_xs, batch_ys = train_dataloader.next_batch()
                # TODO:  do some train step code here
                # print(convolution_layer1.eval(feed_dict).shape)
                # print(convolution_layer2.eval(feed_dict).shape)
                # print(convolution_layer3.eval(feed_dict).shape)
                # print(flat.eval(feed_dict).shape)
                # print(fc_layer.eval(feed_dict).shape)
                # print(output_layer.eval(feed_dict).shape)
                batch_loss, _, batch_accuracy = sess.run(
                    [mean_loss_op, optimizer_op, accuracy_op]
                    , feed_dict={X: batch_xs, y: batch_ys}
                )

                epoch_loss += batch_loss / total_batch
                epoch_accuracy += batch_accuracy / total_batch

                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_accuracy)

                if is_train_mode and (i % PRINT_EVERY) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                          .format(i, batch_loss, batch_accuracy))
            print("Epoch {0}, Overall loss = {1:.3g} and accuracy of {2:.3g}".format(epoch + 1, epoch_loss,
                                                                                     epoch_accuracy))

            plt.plot(batch_losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(epoch + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()

        print('Training finished !')
        output_dir = Constant.SAVE_CHECKPOINT_PATH + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        saver.save(sess, output_dir)
        print('Model saved in file : %s' % output_dir)
    else:
        # skip training and restore graph for validation test
        saver.restore(sess, Constant.LOAD_CHECKPOINT_PATH)

        # Validation
        validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv'
                                           , batch_size=ExperimentConditions.BATCH_SIZE
                                           , label_column_name=Constant.LABEL_COLUMN_NAME
                                           , use_extracted_feature=Data.FEATURE_NAME
                                           , is_training=False)

        validation_accuracy, validation_accuracies = [0, []]
        total_batch = validation_dataloader.num_batch

        for i in range(total_batch):
            batch_xs, batch_ys = validation_dataloader.next_batch()
            feed_dict = {X: batch_xs, y: batch_ys}
            # print(convolution_layer1.eval(feed_dict).shape)
            # print(convolution_layer2.eval(feed_dict).shape)
            # print(convolution_layer3.eval(feed_dict).shape)
            # print(flat.eval(feed_dict).shape)
            # print(fc_layer.eval(feed_dict).shape)
            # print(output_layer.eval(feed_dict).shape)
            datum_accuracy = accuracy_op.eval(feed_dict=feed_dict)

            validation_accuracy += datum_accuracy / total_batch
            validation_accuracies.append(datum_accuracy)

            if (i % PRINT_EVERY) == 0:
                print("Iteration {0}: with minibatch validation accuracy of {1:.2g}" \
                      .format(i, datum_accuracy))
        print("Validation accuracy of {0:.3g}".format(validation_accuracy))

        plt.plot(validation_accuracies)
        plt.grid(True)
        plt.title('Validation Accuracy')
        plt.xlabel('minibatch number')
        plt.ylabel('minibatch accuracy')
        plt.show()