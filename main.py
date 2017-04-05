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
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Shakespeare Drama Composer.')

    help_ = "set -t to compose type, support 's' for shakespeare, 'f' for fiction "
    parser.add_argument('-t', '--type', default='s', choices=['s', 'f'], help=help_)

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_

if __name__ == '__main__':
    args = parse_args()
    if args.type == 's':
        from inference import shakespeare
        if args.train:
            shakespeare.main(True)
        else:
            shakespeare.main(False)
    elif args.type == 'f':
        from inference import fiction
        if args.train:
            fiction.main(True)
        else:
            fiction.main(False)
    else:
        print('[INFO] write option can only be poem or lyric right now.')