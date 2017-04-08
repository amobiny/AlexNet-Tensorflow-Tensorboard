import tensorflow as tf
import os
import alexnet_model as model
import load_input as loader
import numpy as np
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
now = datetime.now()

tf.app.flags.DEFINE_integer('training_epoch', 100, "training epoch")
tf.app.flags.DEFINE_integer('batch_size', 100, "batch size")
tf.app.flags.DEFINE_integer('validation_interval', 200, "validation interval")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, "dropout keep prob")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate")
tf.app.flags.DEFINE_float('rms_decay', 0.9, "rms optimizer decay")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "l2 regularization weight decay")
# tf.app.flags.DEFINE_integer('validation_size', 10000, "validation size in training data")
tf.app.flags.DEFINE_string('save_name', os.getcwd() + '/var.ckpt', "path to save variables")
tf.app.flags.DEFINE_boolean('is_train', True, "True for train, False for test")
#tf.app.flags.DEFINE_string('test_result', 'result.csv', "test file path")
tf.app.flags.DEFINE_string('log_path', "./Graph/" + now.strftime("%Y%m%d-%H%M%S"), "path to save summaries")

# MNIST database
image_size = 28		# size of images
image_channel = 1	# number of channels (1 for black & white)
label_cnt = 10		# number of classes


def train():
    # build graph
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(image_size, image_channel, label_cnt)
    logits = model.inference(inputs, dropout_keep_prob, label_cnt)

    accuracy = model.accuracy(logits, labels)
    loss = model.loss(logits, labels)
    train = tf.train.RMSPropOptimizer(learning_rate, FLAGS.rms_decay).minimize(loss)
#    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # ready for summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_path + '/train/', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.log_path + '/validation/')

    # tf saver
    saver = tf.train.Saver()
    if os.path.isfile(FLAGS.save_name):
        saver.restore(sess, FLAGS.save_name)

    # load mnist data
    train_dataset, train_labels, validation_dataset, validation_labels = loader.load_train_data(image_size,
                                                                                                image_channel,
                                                                                                label_cnt)

    i = 0
    print("Initialized")
    cur_learning_rate = FLAGS.learning_rate
    for epoch in range(FLAGS.training_epoch):
        step = 0
        if epoch % 10 == 0 and epoch > 0:
            cur_learning_rate /= 10
        for start, end in zip(range(0, len(train_dataset), FLAGS.batch_size),
                              range(FLAGS.batch_size, len(train_dataset), FLAGS.batch_size)):
            step += 1
            batch_data = train_dataset[start:end]
            batch_labels = train_labels[start:end]
            print(step)
            feed_dict_batch = {inputs: batch_data, labels: batch_labels,
                               dropout_keep_prob: FLAGS.dropout_keep_prob,
                               learning_rate: cur_learning_rate}
            if i % FLAGS.validation_interval == 0:
                summary, _, loss_batch, acc_batch = sess.run([merged, train, loss, accuracy], feed_dict=feed_dict_batch)
                train_writer.add_summary(summary, i)

                summary_valid, acc_valid, loss_valid = sess.run([merged, accuracy, loss],
                                                                feed_dict={inputs: validation_dataset,
                                                                           labels: np.squeeze(validation_labels),
                                                                           dropout_keep_prob: 1.0})
                validation_writer.add_summary(summary_valid, i)
                print('---------------------------------')
                print('epoch:', epoch + 1, ' , step:', step)
                print("Batch loss: {:.2f}".format(loss_batch))
                print("Batch accuracy: {0:.01%}".format(acc_batch))

                print("Validation loss: {:.2f}".format(loss_valid))
                print("Validation accuracy: {0:.01%}".format(acc_valid))
            else:
                sess.run(train, feed_dict=feed_dict_batch)
            i += 1

    train_writer.close()
    validation_writer.close()


def main(_):
    if FLAGS.is_train:
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()






