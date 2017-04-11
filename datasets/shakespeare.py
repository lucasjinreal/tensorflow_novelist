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
"""
process Shakespeare corpora and generate RNN model need batched data
"""
import numpy as np
from utils.corpus_cleaner_cn import CorpusCleanerCN
import jieba
import collections


class Shakespeare(object):
    def __init__(self, batch_size, n_steps):
        self.corpora_file_path = './datasets/Shakespeare/Shakespeare_cn.txt'
        self.raw_corpus = self.process_raw_data()
        self.batch_size = batch_size
        self.n_steps = n_steps

        self.vocab_size, self.vocab_int_map, self.data = self.prepare_data()
        self.n_chunks = self.data.shape[0] // batch_size
        self.batch_index = 0

    def prepare_data(self):
        corpus_cut = np.array([jieba.lcut(s) for s in self.raw_corpus])
        vocabs = []
        for l in corpus_cut:
            for i in l:
                vocabs.append(i)
        # vocabs = reduce(lambda x, y: x+y, corpus_cut)
        # count every vocab frequency
        # but currently we don't think about the 'most' frequent one, just let it go
        counter = collections.Counter(vocabs)
        counter = counter.most_common()
        vocabs_set, _ = zip(*counter)
        vocab_int_map = {vocab: index for index, vocab in enumerate(vocabs_set)}

        data_flatten = np.array([vocab_int_map[v] for v in vocabs])
        #step=3
        data = np.array([data_flatten[i: i+self.n_steps+1] for i in range(0,data_flatten.shape[0]-self.n_steps -1,3)])
        # let's shuffle data to see anything happens
        np.random.shuffle(data)
        return len(vocabs_set), vocab_int_map, data

    def next_batch(self):
        next_batch = self.data[self.batch_index*self.batch_size: self.batch_index*self.batch_size + self.batch_size]
        self.batch_index += 1
        #if self.batch_index == self.n_chunks:
         #   self.batch_index = 0

        batch_x = next_batch[:, 0: self.n_steps]
        batch_y = next_batch[:, -1]
        return batch_x, batch_y

    def process_raw_data(self):
        cleaner = CorpusCleanerCN()
        raw = cleaner.clean(self.corpora_file_path, clean_level='normal', min_drop=5, is_save=True)
        return raw

