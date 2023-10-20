import os, sys
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from . import utils_dataloaders as ud
except ImportError as e:
    import utils_dataloaders as ud


class GANDiffFace_loader(Dataset):
    def __init__(self, root_dir, transform=None):
        super(GANDiffFace_loader, self).__init__()
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
        self.samples_list, self.subjs_list, self.races_list, self.genders_list = self.make_samples_list_with_labels(self.path_files)
        # print('len(self.samples_list):', len(self.samples_list))
        # print('len(self.subjs_list):', len(self.subjs_list))
        # print('len(self.races_list):', len(self.races_list))
        # print('len(self.genders_list):', len(self.genders_list))
        # print('self.races_list:', self.races_list)
        # print('self.genders_list:', self.genders_list)
        # sys.exit(0)

        assert len(self.path_files) == len(self.samples_list), 'Error, len(self.path_files) must be equals to len(self.samples_list)'


    def get_subj_race_gender_dicts(self, path_files):
        subjs_list   = [None] * len(path_files)
        races_list   = [None] * len(path_files)
        genders_list = [None] * len(path_files)
        for i, path_file in enumerate(path_files):                    # '/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace/images_crops_112x112/White_Male/7381461/49.png'
            subjs_list[i] = path_file.split('/')[-2]                  # '7381461'
            races_list[i] = path_file.split('/')[-3].split('_')[0]    # 'White'
            genders_list[i] = path_file.split('/')[-3].split('_')[1]  # 'Male'
        subjs_list = sorted(list(set(subjs_list)))
        races_list = sorted(list(set(races_list)))
        genders_list = sorted(list(set(genders_list)))

        subjs_dict = {subj:i for i,subj in enumerate(subjs_list)}
        races_dict = {race:i for i,race in enumerate(races_list)}
        genders_dict = {gender:i for i,gender in enumerate(genders_list)}

        return subjs_dict, races_dict, genders_dict


    def make_samples_list_with_labels(self, path_files):
        subjs_dict, races_dict, genders_dict = self.get_subj_race_gender_dicts(path_files)
        samples_list = [None] * len(path_files)
        for i, path_file in enumerate(path_files):                    # '/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace/images_crops_112x112/White_Male/7381461/49.png'
            subj = path_file.split('/')[-2]                           # '7381461'
            race = path_file.split('/')[-3].split('_')[0]             # 'White'
            gender = path_file.split('/')[-3].split('_')[1]           # 'Male'

            subj_idx = subjs_dict[subj]
            race_idx = races_dict[race]
            gender_idx = genders_dict[gender]
            # samples_list[i] = (path_file, subj, race, gender)
            samples_list[i] = (path_file, subj_idx, race_idx, gender_idx)

        return samples_list, subjs_dict, races_dict, genders_dict


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
        img_path, subj_idx, race_idx, gender_idx = self.samples_list[index]

        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            rgb_data = self.load_img(img_path)
            rgb_data = self.normalize_img(rgb_data)

        # return (rgb_data, subj_idx)
        # return (rgb_data, race_idx)
        return (rgb_data, subj_idx, race_idx, gender_idx)

    def __len__(self):
        # return len(self.imgidx)       # original
        return len(self.samples_list)   # Bernardo
    


if __name__ == '__main__':
    root_dir = '/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace/images_crops_112x112'
    transform=None
    train_set = GANDiffFace_loader(root_dir, transform)
