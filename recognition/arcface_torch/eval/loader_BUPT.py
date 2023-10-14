import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd


class Loader_BUPT:
    def __init__(self):
        pass


    def load_protocol(self, protocol_file):
        with open(protocol_file, 'r') as file:
            all_lines = [line.strip() for line in file.readlines()]
            protocol = len(all_lines) * [None]
            # print('all_lines:', all_lines)
            for i, line in enumerate(all_lines):
                # print('line:', line)
                sample0, sample1 = line.split(';')
                # print('sample0:', sample0)
                # print('sample1:', sample1)

                sample0_race, sample0_subj, _ = sample0.split('/')
                sample1_race, sample1_subj, _ = sample1.split('/')

                pair_label = 0
                if sample0_race.lower() == sample1_race.lower() and \
                   sample0_subj.lower() == sample1_subj.lower():
                    pair_label = 1

                pair = {}
                pair['sample0'] = sample0
                pair['sample0_race'] = sample0_race
                pair['sample0_subj'] = sample0_subj
                pair['sample1'] = sample1
                pair['sample1_race'] = sample1_race
                pair['sample1_subj'] = sample1_subj
                pair['pair_label'] = pair_label

                protocol[i] = pair
            return protocol


    def update_paths(self, protocol, data_dir, replace_ext='.png'):
        for i, pair in enumerate(protocol):
            sample0 = pair['sample0']
            sample1 = pair['sample1']

            if replace_ext != '':
                curr_ext = sample0.split('.')[-1]
                sample0 = sample0.replace('.'+curr_ext, replace_ext)
                sample1 = sample1.replace('.'+curr_ext, replace_ext)
                # print('sample0:', sample0)
                # print('sample1:', sample1)

            path_sample0 = os.path.join(data_dir, sample0)
            path_sample1 = os.path.join(data_dir, sample1)
            pair['sample0'] = path_sample0
            pair['sample1'] = path_sample1
            # print('path_sample0:', path_sample0)
            # print('path_sample1:', path_sample1)
            # print('pair_label:', pair_label)
            # print('--------------')
        return protocol


    def load_dataset(self, protocol_file, data_dir, image_size, replace_ext='.png'):
        pairs = self.load_protocol(protocol_file)
        pairs = self.update_paths(pairs, data_dir, replace_ext)

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(pairs)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        issame_list = np.array([bool(pairs[i]['pair_label']) for i in range(len(pairs))])
        races_list = np.array([sorted((pairs[i]['sample0_race'], pairs[i]['sample1_race'])) for i in range(len(pairs))])
        subj_list = np.array([sorted((pairs[i]['sample0_subj'], pairs[i]['sample1_subj'])) for i in range(len(pairs))])
        # for i, (label, races, subjs) in enumerate(zip(issame_list, races_list, subj_list)):
            # print(f'pair:{i} - label: {label} - races: {races} - subjs: {subjs}')
            # if races[0] == races[1]:
            #     print(f'pair:{i} - label: {label} - races: {races} - subjs: {subjs}')
        # print('len(issame_list):', len(issame_list))
        # print('len(races_list):', len(races_list))
        # print('len(subj_list):', len(subj_list))
        # sys.exit(0)

        for idx in range(len(pairs) * 2):
            # _bin = bins[idx]
            # img = mx.image.imdecode(_bin)
            idx_pair = int(idx/2)
            if idx % 2 == 0:
                img = cv2.imread(pairs[idx_pair]['sample0'])
            else:
                img = cv2.imread(pairs[idx_pair]['sample1'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mx.nd.array(img)

            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 1000 == 0:
                print('loading pair', idx)
        print(data_list[0].shape)
        return data_list, issame_list, races_list, subj_list


