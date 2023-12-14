# coding: utf-8
# Source: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval_ijbc.py

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path

import sys
import warnings

# import tensorflow as tf
import yaml
# from model import get_embd


sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')

# # RESNET 50
# parser.add_argument('--model-prefix', default='/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r50_fp16/backbone.pth', help='path to load model.')
# parser.add_argument('--network', default='r50', type=str, help='')

# RESNET 100
parser.add_argument('--model-prefix', default='/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r100_fp16/backbone.pth', help='path to load model.')
parser.add_argument('--network', default='r100', type=str, help='')

# parser.add_argument('--image-path', default='/datasets1/bjgbiesseck/IJB-C/rec_data_ijbc/', type=str, help='')
# parser.add_argument('--image-path', default='/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/crops/', type=str, help='')
parser.add_argument('--image-path', default='/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/crops_align/', type=str, help='')
parser.add_argument('--result-dir', default='results_ijbc_single_img', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
args = parser.parse_args()

target = args.target
# config_path = args.config_path
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size

        # original
        # weight = torch.load(prefix)
        # resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        # resnet.load_state_dict(weight)
        # model = torch.nn.DataParallel(resnet)
        # self.model = model
        # self.model.eval()

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]

        # original (commented by Bernardo because images are already aligned)
        # img = cv2.warpAffine(rimg,
        #                      M, (self.image_size[1], self.image_size[0]),
        #                      borderValue=0.0)
        img = rimg   # Bernardo

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        # input_blob = np.zeros((2, self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data, model):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(int)
    medias = ijb_meta[:, 2].astype(int)
    return templates, medias


# BERNARDO
def read_template_original_ijbc(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=',', header=0).values
    template_id = ijb_meta[:, 0].astype(int)
    subject_id = ijb_meta[:, 1]
    filename = ijb_meta[:, 2]
    return template_id, subject_id, filename


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(int))
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    label = pairs[:, 2].astype(int)
    return t1, t2, label


# BERNARDO
def read_template_pair_list_original_ijbc(path):
    pairs = pd.read_csv(path, sep=',', header=0).values
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    # label = pairs[:, 2].astype(int)
    # return t1, t2, label
    return t1, t2


# BERNARDO
def make_labels_from_template_pairs_original_ijbc(enroll_template_id, enroll_subject_id, verif_template_id, verif_subject_id, p1, p2):
    assert len(p1) == len(p2)

    enroll_template_dict = {}
    for i in range(len(enroll_template_id)):
        enroll_template_dict[enroll_template_id[i]] = enroll_subject_id[i]

    verif_template_dict = {}
    for i in range(len(verif_template_id)):
        verif_template_dict[verif_template_id[i]] = verif_subject_id[i]

    label = np.zeros((len(p1)), dtype=int)
    for i, (t1, t2) in enumerate(zip(p1, p2)):
        label[i] = int(enroll_template_dict[t1] == verif_template_dict[t2])

    return label


# BERNARDO
def adjust_file_names(enroll_subject_id, enroll_filenames, verif_subject_id, verif_filenames):
    enroll_img_paths = [None] * len(enroll_filenames)
    verif_img_paths = [None] * len(verif_filenames)

    for i in range(len(enroll_img_paths)):
        enroll_data = enroll_filenames[i].split('/')
        enroll_img_paths[i] = os.path.join(enroll_data[0], str(enroll_subject_id[i])+'_'+enroll_data[1].replace('.png', '.jpg')) + str(' -1' * 10) + ' 1.0'

    for i in range(len(verif_filenames)):
        verif_data = verif_filenames[i].split('/')
        verif_img_paths[i] = os.path.join(verif_data[0], str(verif_subject_id[i])+'_'+verif_data[1].replace('.png', '.jpg')) + str(' -1' * 10) + ' 1.0'

    files_list = enroll_img_paths
    return files_list


# BERNARDO
def resize_img(image, target_size=(112, 112)):
    original_height, original_width, _ = image.shape
    scaling_factor = max((target_size[0]+1) / original_height, (target_size[1]+1) / original_width)

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    resized_image = cv2.resize(image, (new_width, new_height))

    center_x, center_y = new_width // 2, new_height // 2
    half_target_width, half_target_height = target_size[0] // 2, target_size[1] // 2

    left = center_x - half_target_width
    right = left + target_size[0]
    top = center_y - half_target_height
    bottom = top + target_size[1]

    if new_width != target_size[0] or new_height != target_size[1]:
        cropped_image = resized_image[top:bottom, left:right]
    else:
        cropped_image = resized_image

    return cropped_image


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)
    # data_shape = (112, 112, 3)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)
    image_size = (112, 112)
    # self.image_size = image_size

    # LOAD TRAINED MODEL
    print(f'Loading trained model \'{model_path}\'...')
    weight = torch.load(model_path)
    resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    resnet.load_state_dict(weight)
    model = torch.nn.DataParallel(resnet)
    # self.model = model
    model.eval()

    with torch.no_grad():
        batch_data = np.empty((2 * batch_size, 3, 112, 112))
        # batch_data = np.empty((2 * batch_size, 112, 112, 3))
        embedding = Embedding(model_path, data_shape, batch_size)
        num_batches = int(np.ceil(len(files) / batch_size))
        print('Computing embeddings...')
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            # img = resize_img(img, (112, 112))   # Bernardo
            input_blob = embedding.get(img, lmk)

            batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
            batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
            if (img_index + 1) % batch_size == 0:
                # print('batch', batch)
                print(f'batch {batch}/{num_batches-1}', end='\r')
                img_feats[batch * batch_size:batch * batch_size +
                                            batch_size][:] = embedding.forward_db(batch_data, model)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])

        batch_data = np.empty((2 * rare_size, 3, 112, 112))
        # batch_data = np.empty((2 * rare_size, 112, 112, 3))
        embedding = Embedding(model_path, data_shape, rare_size)
        for img_index, each_line in enumerate(files[len(files) - rare_size:]):
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            # img = resize_img(img, (112, 112))   # Bernardo
            input_blob = embedding.get(img, lmk)
            batch_data[2 * img_index][:] = input_blob[0]
            batch_data[2 * img_index + 1][:] = input_blob[1]
            if (img_index + 1) % rare_size == 0:
                # print('batch', batch)
                print(f'batch {batch}/{num_batches-1}', end='\r')
                img_feats[len(files) -
                        rare_size:][:] = embedding.forward_db(batch_data, model)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
        # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
        print('')
        return img_feats, faceness_scores


# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            # print('Finish Calculating {} template features.'.format(count_template))
            print(f'Finish Calculating template features: {count_template}/{img_feats.shape[0]}', end='\r')
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    print('')
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            # print('Finish {}/{} pairs.'.format(c, total_sublists))
            print('Finish {}/{} pairs.'.format(c, total_sublists), end='\r')
    print('')
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats



exper_id = model_path.split('/')[-2]            # Bernardo
save_path = os.path.join(result_dir, exper_id)  # Bernardo
score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
label_save_file = os.path.join(save_path, "label.npy")
img_feats_save_file = os.path.join(save_path, "img_feats.npy")
faceness_scores_save_file = os.path.join(save_path, "faceness_scores.npy")

# Verification protocol files
path_enroll_templates = '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/enroll_templates.csv'
path_verif_templates =  '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/verif_templates.csv'
path_match_pairs =      '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/match.csv'
# path_match_pairs =    '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/match_TEST_BERNARDO.csv'  # Toy example for sanity check



# # Step1: Load Meta Data
# In[ ]:
# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
assert target == 'IJBC' or target == 'IJBB'
start = timeit.default_timer()
# templates, medias = read_template_media_list(os.path.join('%s/meta' % image_path, '%s_face_tid_mid.txt' % target.lower()))
print(f'Loading enroll templates \'{path_enroll_templates}\'...')
enroll_template_id, enroll_subject_id, enroll_filenames = read_template_original_ijbc(path_enroll_templates)  # one image protocol
print(f'num enroll templates: {len(enroll_template_id)}')
print(f'Loading verification templates \'{path_verif_templates}\'...')
verif_template_id, verif_subject_id, verif_filenames = read_template_original_ijbc(path_verif_templates)      # one image protocol
print(f'num verif templates: {len(verif_template_id)}')
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')

# In[ ]:
# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
# p1, p2, label = read_template_pair_list(os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))
print(f'Loading match pairs indexes \'{path_match_pairs}\'...')
p1, p2 = read_template_pair_list_original_ijbc(path_match_pairs)
label = make_labels_from_template_pairs_original_ijbc(enroll_template_id, enroll_subject_id, verif_template_id, verif_subject_id, p1, p2)
num_genuine_pairs = np.sum(label == 1)
num_impostor_pairs = np.sum(label == 0)
assert num_genuine_pairs+num_impostor_pairs == len(label)
print(f'num pairs: {len(label)}    (genuine: {num_genuine_pairs}    impostor: {num_impostor_pairs})')
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')





# # Step 2: Get Image Features
if not os.path.exists(img_feats_save_file):   # Bernardo
    # In[ ]:
    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()
    img_path = image_path     # Bernardo
    files_list = adjust_file_names(enroll_subject_id, enroll_filenames, verif_subject_id, verif_filenames)   # Bernardo

    img_feats, faceness_scores = get_image_feature(img_path, files_list, model_path, 0, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
    
    # Bernardo
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving img_feats:', img_feats_save_file)
    np.save(img_feats_save_file, img_feats)
    print('Saving faceness_scores:', img_feats_save_file)
    np.save(faceness_scores_save_file, faceness_scores)

else:
    print('Loading img_feats:', img_feats_save_file)
    img_feats = np.load(img_feats_save_file)
    print('Loading faceness_scores:', img_feats_save_file)
    faceness_scores = np.load(faceness_scores_save_file)

    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
print('----------------------')





# # Step3: Get Template Features
# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                    2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

print('Computing template embeddings...')
# template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)                     # original
template_norm_feats, unique_templates = image2template_feature(img_input_feats, enroll_template_id, verif_template_id)   # Bernardo
print('Template Features Shape: ({} , {}) .'.format(template_norm_feats.shape[0], template_norm_feats.shape[1]))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')





# # Step 4: Get Template Similarity Scores
# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
print('Computing pairs similarity scores...')
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

if not os.path.exists(save_path):
    os.makedirs(save_path)
# score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
print('Saving scores:', score_save_file)
np.save(score_save_file, score)
print('Saving labels:', label_save_file)
np.save(label_save_file, label)
print('----------------------')




# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
roc_auc = 0.0
print('Evaluating model...')
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
# fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
path_save_fig = os.path.join(save_path, '%s.pdf' % target.lower())
print(f'Saving ROC curve \'{path_save_fig}\'')
fig.savefig(path_save_fig)

path_save_tpr_fpr_table = os.path.join(save_path, '%s_tpr_fpr_table.txt' % target.lower())
with open(path_save_tpr_fpr_table, 'w') as f:
    print(f'Saving tpr_fpr_table \'{path_save_tpr_fpr_table}\'')
    f.write('tpr_fpr_table\n')
    f.write(tpr_fpr_table.get_string())
    f.write('\nROC-AUC = %0.4f %%\n' % (roc_auc * 100))

print(tpr_fpr_table)
print('ROC-AUC = %0.4f %%' % (roc_auc * 100))   # Bernardo
