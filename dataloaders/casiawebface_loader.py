import os, sys
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

try:
    from . import utils_dataloaders as ud
except ImportError as e:
    import utils_dataloaders as ud


class CASIAWebFace_loader(Dataset):
    def __init__(self, root_dir, transform=None, other_dataset=None):
        super(CASIAWebFace_loader, self).__init__()
        # self.transform = transform
        # self.root_dir = root_dir
        # self.local_rank = local_rank
        # path_imgrec = os.path.join(root_dir, 'train.rec')
        # path_imgidx = os.path.join(root_dir, 'train.idx')
        # self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # s = self.imgrec.read_idx(0)
        # header, _ = mx.recordio.unpack(s)
        # if header.flag > 0:
        #     self.header0 = (int(header.label[0]), int(header.label[1]))
        #     self.imgidx = np.array(range(1, int(header.label[0])))
        # else:
        #     self.imgidx = np.array(list(self.imgrec.keys))

        if not os.path.exists(root_dir):
            raise Exception(f'Dataset path does not exists: \'{root_dir}\'')

        self.root_dir = root_dir
        self.file_ext = '.png'
        self.path_files = ud.find_files(self.root_dir, self.file_ext)
        self.path_files = self.append_dataset_name(self.path_files, dataset_name='casia')
        self.subjs_list, self.subjs_dict, self.races_dict, self.genders_dict = self.get_subj_race_gender_dicts(self.path_files)

        self.samples_list = self.make_samples_list_with_labels(self.path_files, self.subjs_list, self.subjs_dict, self.races_dict, self.genders_dict)
        assert len(self.path_files) == len(self.samples_list), 'Error, len(self.path_files) must be equals to len(self.samples_list)'
        # print('self.samples_list', self.samples_list)
        # print('len(self.samples_list)', len(self.samples_list))

        if not other_dataset is None:
            self.path_files += other_dataset.path_files
            self.subjs_list += other_dataset.subjs_list
            _, max_subj_idx = ud.get_min_max_value_dict(self.subjs_dict)
            self.subjs_dict = ud.merge_dicts(self.subjs_dict, other_dataset.subjs_dict, stride=max_subj_idx+1)
            self.races_dict = ud.merge_dicts(self.races_dict, other_dataset.races_dict)
            self.genders_dict = ud.merge_dicts(self.genders_dict, other_dataset.genders_dict)
            self.samples_list += other_dataset.samples_list

        self.final_samples_list = self.replace_strings_labels_by_int_labels(self.samples_list, self.subjs_dict, self.races_dict, self.genders_dict)
        random.shuffle(self.final_samples_list)
        # print('self.final_samples_list', self.final_samples_list)
        # print('len(self.final_samples_list)', len(self.final_samples_list))


    def append_dataset_name(self, path_files, dataset_name):
        for i in range(len(path_files)):
            path_files[i] = (path_files[i], dataset_name)
        return path_files


    def get_subj_race_gender_dicts(self, path_files):
        subjs_list   = [None] * len(path_files)
        for i, (path_file, dataset_name) in enumerate(path_files):        # '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112/0/1.png'
            subjs_list[i] = dataset_name + '_' + path_file.split('/')[-2] # 'casia_0'
        subjs_list = sorted(list(set(subjs_list)))

        subjs_dict = {subj:i for i,subj in enumerate(subjs_list)}
        races_dict = {None: -1}
        genders_dict = {None: -1}
        return subjs_list, subjs_dict, races_dict, genders_dict


    def make_samples_list_with_labels(self, path_files, subjs_list, subjs_dict, races_dict, genders_dict):
        samples_list = [None] * len(path_files)
        subjs_dict_num_samples = {subj:0 for subj in list(subjs_dict.keys())}
        for i, (path_file, dataset_name) in enumerate(path_files):        # '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112/0/1.png'
            subj = dataset_name + '_' + path_file.split('/')[-2]          # 'casia_0'

            subjs_dict_num_samples[subj] += 1
            samples_list[i] = (dataset_name, path_file, subj, None, None)

        return samples_list


    def replace_strings_labels_by_int_labels(self, samples_list, subjs_dict, races_dict, genders_dict):
        final_samples_list = [None] * len(samples_list)
        for i in range(len(final_samples_list)):
            # print(f'samples_list[{i}]: {samples_list[i]}')
            dataset_name, path_file, subj, race, gender = samples_list[i]
            subj_idx = subjs_dict[subj] if not subjs_dict is None else -1
            race_idx = races_dict[race] if not races_dict is None else -1
            gender_idx = genders_dict[gender] if not genders_dict is None else -1
            final_samples_list[i] = ( dataset_name, path_file, subj_idx, race_idx, gender_idx)
            # print(f'final_samples_list[{i}]: {final_samples_list[i]}')
        return final_samples_list


    def normalize_img(self, img):
        img = np.transpose(img, (2, 0, 1))  # from (224,224,3) to (3,224,224)
        img = ((img/255.)-0.5)/0.5
        # print('img:', img)
        # sys.exit(0)
        return img


    def load_img(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32)


    def __getitem__(self, index):
        # idx = self.imgidx[index]
        # s = self.imgrec.read_idx(idx)
        # header, img = mx.recordio.unpack(s)
        # label = header.label
        # if not isinstance(label, numbers.Number):
        #     label = label[0]
        # label = torch.tensor(label, dtype=torch.long)
        # sample = mx.image.imdecode(img).asnumpy()
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # return sample, label

        # Bernardo
        dataset_name, img_path, subj_idx, race_idx, gender_idx = self.final_samples_list[index]

        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            rgb_data = self.load_img(img_path)
            rgb_data = self.normalize_img(rgb_data)

        # return (rgb_data, subj_idx)
        # return (rgb_data, race_idx)
        return (rgb_data, subj_idx, race_idx, gender_idx)


    def __len__(self):
        # return len(self.imgidx)            # original
        return len(self.final_samples_list)  # Bernardo


    def get_cls_num_list(self):
        cls_num_list = []
        for key in list(self.subjs_dict_num_samples.keys()):
            cls_num_list.append(self.subjs_dict_num_samples[key])
        return cls_num_list



# if __name__ == '__main__':
#     # root_dir = '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112'
#     # root_dir = '/nobackup/unico/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112'
#     root_dir = '/home/bjgbiesseck/datasets/real/1_CASIA-WebFace/imgs_crops_112x112'
#     print('Loading casia paths...')
#     transform=None
#     train_set = CASIAWebFace_loader(root_dir, transform, None)
#
#     min_subj_idx, max_subj_idx = 0, 0
#     for i, sample in enumerate(train_set.final_samples_list):
#         if sample[2] < min_subj_idx: min_subj_idx = sample[2]
#         if sample[2] > max_subj_idx: max_subj_idx = sample[2]
#         print(f'{i} - {sample} - min_subj_idx: {min_subj_idx} - max_subj_idx: {max_subj_idx}')
#
#     # cls_num_list = train_set.get_cls_num_list()
#     # print('cls_num_list:', cls_num_list)
#     # print('len(cls_num_list):', len(cls_num_list))


if __name__ == '__main__':
    import dcface_loader
    root_dir = '/home/bjgbiesseck/datasets/synthetic/DCFace/dcface_wacv/organized'
    print('Loading dcface paths...')
    transform=None
    train_set = dcface_loader.DCFace_loader(root_dir, transform, None)

    # root_dir = '/nobackup/unico/frcsyn_wacv2024/datasets/synthetic/DCFace/dcface_wacv/organized'
    root_dir = '/home/bjgbiesseck/datasets/real/1_CASIA-WebFace/imgs_crops_112x112'
    print('Loading casia paths...')
    transform=None
    train_set = CASIAWebFace_loader(root_dir, transform, train_set)

    min_subj_idx, max_subj_idx = 0, 0
    for i, sample in enumerate(train_set.final_samples_list):
        if sample[2] < min_subj_idx: min_subj_idx = sample[2]
        if sample[2] > max_subj_idx: max_subj_idx = sample[2]
        print(f'{i} - {sample} - min_subj_idx: {min_subj_idx} - max_subj_idx: {max_subj_idx}')

