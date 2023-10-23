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

        self.root_dir = root_dir
        self.file_ext = '.png'
        self.path_files = ud.find_files(self.root_dir, self.file_ext)
        self.samples_list, self.subjs_list, self.races_list, self.genders_list = self.make_samples_list_with_labels(self.path_files)
        # print('self.samples_list:', self.samples_list)
        # print('len(self.samples_list):', len(self.samples_list))
        # print('len(self.subjs_list):', len(self.subjs_list))
        # sys.exit(0)
        assert len(self.path_files) == len(self.samples_list), 'Error, len(self.path_files) must be equals to len(self.samples_list)'

        if not other_dataset is None:
            self.path_files += other_dataset.path_files
            self.samples_list += other_dataset.samples_list
            self.subjs_list = ud.merge_dicts(self.subjs_list, other_dataset.subjs_list)
            self.races_list = ud.merge_dicts(self.races_list, other_dataset.races_list)
            self.genders_list = ud.merge_dicts(self.genders_list, other_dataset.genders_list)


    def get_subj_race_gender_dicts(self, path_files):
        subjs_list   = [None] * len(path_files)
        for i, path_file in enumerate(path_files):                  # '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112/0/1.png'
            subjs_list[i] = 'casia_' + path_file.split('/')[-2]     # 'casia_0'
        subjs_list = sorted(list(set(subjs_list)))

        subjs_dict = {subj:i for i,subj in enumerate(subjs_list)}
        genders_dict = None
        races_dict = None

        return subjs_dict, races_dict, genders_dict


    def make_samples_list_with_labels(self, path_files):
        subjs_dict, races_dict, genders_dict = self.get_subj_race_gender_dicts(path_files)
        samples_list = [None] * len(path_files)
        for i, path_file in enumerate(path_files):        # '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112/0/1.png'
            subj = 'casia_' + path_file.split('/')[-2]    # 'casia_0'

            subj_idx = subjs_dict[subj]
            gender_idx = -1
            race_idx = -1
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
    root_dir = '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112'
    transform=None
    train_set = CASIAWebFace_loader(root_dir, transform, None)
