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
from tensorflow.contrib import rnn


class RNNModel(object):
    def __init__(self, inputs, labels, n_units, n_layers, lr, vocab_size):
        self.inputs = inputs
        self.labels = labels
        self.n_units = n_units
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.lr = lr

        self.outputs, self.states = self.rnn_model()

        if self.labels is not None:
            self.train_op, self.loss = self.update()

    def rnn_model(self):
        cell = rnn.BasicLSTMCell(num_units=self.n_units)
        multi_cell = rnn.MultiRNNCell([cell]*self.n_layers)
        # we only need one output so get it wrapped to out one value which is next word index
        cell_wrapped = rnn.OutputProjectionWrapper(multi_cell, output_size=1)

        # get input embed
        embedding = tf.Variable(initial_value=tf.random_uniform([self.vocab_size, self.n_units], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        # what is inputs dim??

        # add initial state into dynamic rnn, if I am not result would be bad, I tried, don't know why
        if self.labels is not None:
            initial_state = cell_wrapped.zero_state(int(inputs.get_shape()[0]), tf.float32)
        else:
            initial_state = cell_wrapped.zero_state(1, tf.float32)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=inputs, dtype=tf.float32, initial_state=initial_state)
        outputs = tf.reshape(outputs, [int(outputs.get_shape()[0]), int(inputs.get_shape()[1])])

        w = tf.Variable(tf.truncated_normal([int(inputs.get_shape()[1]), self.vocab_size]))
        b = tf.Variable(tf.zeros([self.vocab_size]))

        logits = tf.nn.bias_add(tf.matmul(outputs, w), b)
        return logits, states

    def update(self):
        labels_one_hot = tf.one_hot(tf.reshape(self.labels, [-1]), depth=self.vocab_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=self.outputs)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=loss)
        return train_op, total_loss








