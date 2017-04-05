# -*- coding: utf-8 -*-
# file: corpus_cleaner_cn.py
# author: JinTian
# time: 08/03/2017 8:02 PM
# Copyright 2017 JinTian. All Rights Reserved.
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
# ------------------------------------------------------------------------
"""
this script using for clean Chinese corpus.
you can set level for clean, i.e.:
level='all', will clean all character that not Chinese, include punctuations
level='normal', this will generate corpus like normal use, reserve alphabets and numbers
level='clean', this will remove all except Chinese and Chinese punctuations

besides, if you want remove complex Chinese characters, just set this to be true:
simple_only=True
"""
import numpy as np
import os
import string
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


class CorpusCleanerCN(object):

    def __init__(self):
        self.cn_punctuation = ['，', '。', '！', '？', '"', '"', '、']
        self.en_punctuation = [',', '.', '?', '!', '"', '"']

    def clean(self, file_name, clean_level, min_drop=1, is_save=True, summary=True):
        if os.path.dirname(file_name):
            base_dir = os.path.dirname(file_name)
        else:
            logging.error('not set dir. please check')

        save_file = os.path.join(base_dir, os.path.basename(file_name).split('.')[0] + '_cleaned.txt')
        with open(file_name, 'r+') as f:
            clean_content = []
            for l in f.readlines():
                l = l.strip()
                if l == '':
                    pass
                elif len(l) < min_drop:
                    pass
                else:
                    l = list(l)
                    should_remove_words = []
                    for w in l:
                        if not self.should_reserve(w, clean_level):
                            should_remove_words.append(w)
                    clean_line = [c for c in l if c not in should_remove_words]
                    clean_line = ''.join(clean_line)
                    if clean_line != '':
                        clean_content.append(clean_line)
        if is_save:
            with open(save_file, 'w+') as f:
                for l in clean_content:
                    f.write(l + '\n')
            logging.info('cleaned file have been saved to %s.' % save_file)
        if summary:
            logging.info('corpora all lines: %s' % len(clean_content))
            logging.info('max length in lines: %s' % np.max([len(i) for i in clean_content]))
            logging.info('min length in lines: %s' % np.min([len(i) for i in clean_content]))
            logging.info('average length in lines: %s' % np.average([len(i) for i in clean_content]))
        return clean_content

    def should_reserve(self, w, clean_level):
        if w == ' ':
            return True
        else:
            if clean_level == 'all':
                # only reserve Chinese characters
                if w in self.cn_punctuation or w in string.punctuation or self.is_alphabet(w):
                    return False
                else:
                    return self.is_chinese(w)
            elif clean_level == 'normal':
                # reserve Chinese characters, English alphabet, number
                if self.is_chinese(w) or self.is_alphabet(w) or self.is_number(w):
                    return True
                elif w in self.cn_punctuation or w in self.en_punctuation:
                    return True
                else:
                    return False
            elif clean_level == 'clean':
                if self.is_chinese(w):
                    return True
                elif w in self.cn_punctuation:
                    return True
                else:
                    return False
            else:
                raise "clean_level not support %s, please set for all, normal, clean" % clean_level

    @staticmethod
    def is_chinese(uchar):
        """is chinese"""
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    @staticmethod
    def is_number(uchar):
        """is number"""
        if u'\u0030' <= uchar <= u'\u0039':
            return True
        else:
            return False

    @staticmethod
    def is_alphabet(uchar):
        """is alphabet"""
        if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
            return True
        else:
            return False

    @staticmethod
    def semi_angle_to_sbc(uchar):
        """半角转全角"""
        inside_code = ord(uchar)
        if inside_code < 0x0020 or inside_code > 0x7e:
            return uchar
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            inside_code += 0xfee0
        return chr(inside_code)

    @staticmethod
    def sbc_to_semi_angle(uchar):
        """全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            return uchar
        return chr(inside_code)








