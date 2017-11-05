import autoencoder_fp
from utilities.experiments_1 import AutoEncoder

autoencoder_fp.build_graph(
    AutoEncoder.Experiment1
    , Data.ChormaStftHop4096
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    PRINT_EVERY = 10
    train()
    validate()
