# Copyright 2017 Jin Fagang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
import tensorflow as tf
from models.rnn_model_no_state import RNNModel
import logging
import os
import numpy as np
import jieba
import pickle

logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s %(filename)s line:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
jieba.default_logger.disabled = True


tf.app.flags.DEFINE_integer('batch_size', 360, 'batch size.')
tf.app.flags.DEFINE_integer('n_steps', 6, 'length of inputs columns.')
tf.app.flags.DEFINE_integer('n_units', 4, 'number units in rnn cell.')
tf.app.flags.DEFINE_integer('n_layers', 2, 'number of layer of stack rnn model.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
# set this to 'main.py' relative path
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints/fiction/', 'checkpoints save path.')

tf.app.flags.DEFINE_string('model_prefix', 'fiction', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 10050, 'train how many epochs.')
tf.app.flags.DEFINE_boolean('is_restore', True, 'to restore from previous or not.')

FLAGS = tf.app.flags.FLAGS


def running(is_train=True):
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)

    # get fiction and dump it into pkl
    # fiction = Fiction(batch_size=FLAGS.batch_size, n_steps=FLAGS.n_steps)
    with open('./datasets/Fiction/fiction.pkl', 'rb') as f:
        fiction = pickle.load(f)
    vocab_size = fiction.vocab_size

    if is_train:
        inputs = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.n_steps])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])
    else:
        inputs = tf.placeholder(tf.int32, [1, FLAGS.n_steps])
        labels = None
    model = RNNModel(inputs, labels, n_units=FLAGS.n_units, n_layers=FLAGS.n_layers,
                     lr=FLAGS.learning_rate, vocab_size=vocab_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if FLAGS.is_restore:
            if os.path.exists(checkpoint):
                saver.restore(sess, checkpoint)
                logging.info("restore from the checkpoint {0}".format(checkpoint))
                start_epoch += int(checkpoint.split('-')[-1])
        if is_train:
            logging.info('training start...')
            epoch = 0
            try:
                for epoch in range(start_epoch, FLAGS.epochs):
                    for batch in range(fiction.n_chunks):
                        batch_x, batch_y = fiction.next_batch()
                        loss, _ = sess.run([model.loss, model.train_op],
                                           feed_dict={inputs: batch_x, labels: batch_y})
                        logging.info('epoch: %s,  batch: %s, loss: %s' % (epoch, batch, loss))
                    if epoch % 60 == 0:
                        saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
                logging.info('optimization done! enjoy your Fiction composer!')
            except KeyboardInterrupt:
                logging.info('interrupt manually, try saving checkpoint for now...')
                saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
                logging.info('last epoch were saved, next time will start from epoch {}.'.format(epoch))
        else:
            logging.info('I am thinking to compose  Jin Yong fiction novel...')
            if checkpoint:
                saver.restore(sess, checkpoint)
                logging.info("restore from the checkpoint {0}".format(checkpoint))
                start_epoch += int(checkpoint.split('-')[-1])

            vocab_int_map = fiction.vocab_int_map
            start_sentence = input('please input a sentence in Chinese: ')
            sent = process_sent(start_sentence, vocab_int_map, FLAGS.n_steps)

            next_words = []
            for i in range(2000):
                outputs = sess.run([model.outputs], feed_dict={inputs: sent})
                next_word_index = np.argmax(outputs)
                next_words.append(next_word_index)
                sent = np.append(sent, next_word_index)
                sent = np.array([sent[1:]])
            drama_text = [{v: k for k, v in vocab_int_map.items()}[i] for i in next_words]
            drama_text.insert(0, start_sentence)
            pretty_print(drama_text)


def process_sent(sent, vocab_int, steps):
    """
    this file token sentence and make it into numpy array, return a fixed length 2d array
    :param sent: 
    :param vocab_int: 
    :param steps: 
    :return: 
    """
    sent_list = jieba.lcut(sent)
    # if words not in vocab dict then let this word be a random index which maybe other words
    index_list = [vocab_int[i] if i in vocab_int.keys() else np.random.randint(0, 90) for i in sent_list]
    if len(index_list) < steps:
        index_list = np.hstack((index_list, np.random.randint(0, 90, steps - len(index_list))))
    else:
        index_list = index_list[0: steps]
    return np.array([index_list])


def pretty_print(words_list):
    """
    print the words list sentence by sentence
    :param words_list: 
    :return: 
    """
    print('I am thinking to compose Jin Yong fiction, hand on a minute...')
    all_punctuations = ['。', '？', '！', '，', ',']
    enter_punctuations = ['。', '？', '！', '.']
    token = 'TOKEN'
    add_token = [i+token if i in enter_punctuations else i for i in words_list]
    split_token = ''.join(add_token).split(token)
    drop_extra = [i for i in split_token if i not in enter_punctuations]
    print('Here is what I got: ')
    for i in drop_extra:
        print(i)


def main(is_train):
    running(is_train)

if __name__ == '__main__':
    tf.app.run()