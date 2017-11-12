from tqdm import *
import telegram_send

import utilities
import attention as model

def train(project_constant, data_constant, experiment_parameters):
    try:
        train_dataloader = utilities.load_dataloader(
            project_constant.METADATA_PATH
            , experiment_parameters.BATCH_SIZE
            , project_constant.LABEL_COLUMN_NAME
            , data_constant.FEATURE_NAME
            , is_training=True
        )

        num_batches = utilities.get_numbers_of_batch(train_dataloader)

        with model.initialize_graph():
            X, y, logits, mean_loss, train_step, prediction_indices = model.build_attention_model(
                data_constant
                , experiment_parameters
            )

            with model.initialize_session() as sess:
                model.initialize_variables(sess)
                epoch_losses, epoch_accuracies = list(), list()
                # epoch
                for epoch in range(experiment_parameters.EPOCH_NUM):
                    epoch_loss, epoch_accuracy = 0, 0

                    # batch
                    for batch_index in tqdm(range(num_batches)):
                        batch_loss = 0
                        batch_X, batch_y = utilities.get_batch(train_dataloader)
                        batch_votings = utilities.initialize_batch_votings(
                            experiment_parameters.BATCH_SIZE
                            , experiment_parameters.N_CLASS
                        )
                        batch_X_chunks = utilities.split_X(
                            batch_X
                            , utilities.get_split_indices(
                                data_constant.IMAGE_WIDTH
                                , experiment_parameters.WINDOW_NUM
                            )
                        )

                        # chunk
                        for selected_window_index in tqdm(range(experiment_parameters.WINDOW_NUM)):
                            feed_dict = {X: utilities.get_current_chunk(batch_X_chunks, selected_window_index)
                                , y: batch_y}
                            _, mean_loss_, prediction_indices_ = sess.run([train_step, mean_loss, prediction_indices]
                                                                          , feed_dict)
                            batch_votings = utilities.add_votings(batch_votings, prediction_indices_)
                            batch_loss += mean_loss_ / experiment_parameters.WINDOW_NUM

                        batch_accuracy = utilities.calculate_accuracy(
                            utilities.get_batch_predictions(batch_votings, labels=batch_y)
                        )

                        epoch_loss += batch_loss / num_batches
                        epoch_accuracy += batch_accuracy / num_batches

                        message = "epoch: {3}/{4}, batch: {0}/{1}, batch_accuracy: {2}".format(batch_index
                                                                                               , num_batches
                                                                                               , batch_accuracy
                                                                                               , epoch
                                                                                               , experiment_parameters.EPOCH_NUM
                                                                                               )
                        telegram_send.send(messages=[message])

                    epoch_losses += epoch_loss
                    epoch_accuracies += epoch_accuracy
                    message = "epoch: {0}/{3}, epoch_loss: {1}, batch_accuracy: {2}".format(epoch
                                                                                            , epoch_loss
                                                                                            , epoch_accuracy
                                                                                            , experiment_parameters.EPOCH_NUM
                                                                                            )
                    telegram_send.send(messages=[message])

                # save model
                output_directory_path = utilities.set_output_directory_path(project_constant.SAVE_CHECKPOINT_PATH)
                utilities.save_model(utilities.set_saver(), sess, output_directory_path)
                telegram_send.send(messages=['Model saved in file : %s' % output_directory_path])

    except Exception as e:
        telegram_send.send(messages=[str(e)])