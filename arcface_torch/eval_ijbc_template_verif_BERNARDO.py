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
from itertools import product


sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')

# RESNET 50
# parser.add_argument('--model-prefix', default='/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r50_fp16/backbone.pth', help='path to load model.')
# parser.add_argument('--network', default='r50', type=str, help='')

# RESNET 100
parser.add_argument('--model-prefix', default='/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r100_fp16/backbone.pth', help='path to load model.')
parser.add_argument('--network', default='r100', type=str, help='')

parser.add_argument('--image-path', default='/datasets1/bjgbiesseck/IJB-C/rec_data_ijbc/', type=str, help='')
parser.add_argument('--result-dir', default='results_ijbc_template', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--majority-voting', action='store_true')
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

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img, lmk)
            batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
            batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
            if (img_index + 1) % batch_size == 0:
                print('batch', batch)
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
            input_blob = embedding.get(img, lmk)
            batch_data[2 * img_index][:] = input_blob[0]
            batch_data[2 * img_index + 1][:] = input_blob[1]
            if (img_index + 1) % rare_size == 0:
                print('batch', batch)
                img_feats[len(files) -
                        rare_size:][:] = embedding.forward_db(batch_data, model)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
        # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
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
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:

# Bernardo
def count_number_sample_matchings(unique_templates, p1, p2):
    num_total_sample_matchings = 0
    num_samples_templ = {}

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        num_samples_templ[uqt] = len(ind_t)

    for c, (t1, t2) in enumerate(zip(p1, p2)):
        num_total_sample_matchings += num_samples_templ[t1] * num_samples_templ[t2]

    return num_samples_templ, num_total_sample_matchings


# Bernardo
def compute_score_between_template_samples(img_input_feats, templates, medias, p1, p2, device='cuda'):
    img_input_feats_norm = sklearn.preprocessing.normalize(img_input_feats)
    templ_sampl_score = np.zeros((num_total_sample_matchings,))

    if device == 'cuda':
        img_input_feats_norm = torch.from_numpy(img_input_feats_norm).float().to(device)
        templ_sampl_score = torch.from_numpy(templ_sampl_score).float().to(device)

    beg_idx_sampl_score = 0
    end_idx_sampl_score = 0
    start = timeit.default_timer()
    print('Computing scores between template samples...')
    for c, (t1, t2) in enumerate(zip(p1, p2)):
        (ind_t1,) = np.where(templates == t1)
        (ind_t2,) = np.where(templates == t2)
        # print('ind_t1:', ind_t1, '    num_samples_templ[t1]:', num_samples_templ[t1])
        # print('ind_t2:', ind_t2, '    num_samples_templ[t2]:', num_samples_templ[t2])

        num_combs = num_samples_templ[t1]*num_samples_templ[t2]
        similarity_score = np.zeros((num_combs,))

        s1_comb = [None] * num_combs
        s2_comb = [None] * num_combs
        for i, (s1, s2) in enumerate(product(ind_t1, ind_t2)):
            s1_comb[i], s2_comb[i] = s1, s2
        # print('s1_comb:', s1_comb)
        # print('s2_comb:', s2_comb)
        feat1 = img_input_feats_norm[s1_comb]
        feat2 = img_input_feats_norm[s2_comb]

        if device == 'cuda':
            similarity_score = torch.sum(feat1 * feat2, -1)
        else:
            similarity_score = np.sum(feat1 * feat2, -1)
        # print('similarity_score:', similarity_score)

        end_idx_sampl_score = beg_idx_sampl_score + num_combs
        # print('beg_idx_sampl_score:', beg_idx_sampl_score)
        # print('end_idx_sampl_score:', end_idx_sampl_score)
        templ_sampl_score[beg_idx_sampl_score:end_idx_sampl_score] = similarity_score
        beg_idx_sampl_score = end_idx_sampl_score

        if c > 0 and c % 10000 == 0:
            print(f'pair: {c}/{len(p1)}')
            # print('t1:', t1)
            # print('t2:', t2)
            stop = timeit.default_timer()
            print('Time: %.2f s.' % (stop - start))
            start = timeit.default_timer()
            # input('PAUSE')
            print('----------------')

    return templ_sampl_score


# Bernardo
def calculate_accuracy(threshold, predict_issame, actual_issame):
    # predict_issame = np.less(scores, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    # acc = float(tp + tn) / scores.size
    acc = float(tp + tn) / predict_issame.size
    return tpr, fpr, acc


# Bernardo
def calculate_tar_far(threshold, predict_issame, actual_issame):
    # predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    tar = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return tar, far


# Bernardo
def predict_templ_label_majority_voting(threshold, scores):
    predict_issame = np.greater(scores, threshold)
    # print('predict_issame:', predict_issame)
    if np.sum(np.where(predict_issame)) >= len(predict_issame):
        return 1
    return 0


# Bernardo
def evaluate_model_majority_voting_template(templates, p1, p2, templ_true_label, templ_sampl_score):
    # thresholds = np.arange(0, 4, 0.001)
    thresholds = np.arange(0, 4, 0.01)
    nrof_thresholds = len(thresholds)
    tprs = np.zeros((nrof_thresholds,))
    fprs = np.zeros((nrof_thresholds,))
    accs = np.zeros((nrof_thresholds,))
    tars = np.zeros((nrof_thresholds,))
    fars = np.zeros((nrof_thresholds,))

    for threshold_idx, threshold in enumerate(thresholds):
        # print(f'{threshold_idx}/{nrof_thresholds} - threshold: {threshold}')

        templ_pred_label = np.zeros((len(templ_true_label)), dtype=int)
        beg_idx_sampl_score = 0
        end_idx_sampl_score = 0
        start = timeit.default_timer()
        for c, (t1, t2) in enumerate(zip(p1, p2)):
            (ind_t1,) = np.where(templates == t1)
            (ind_t2,) = np.where(templates == t2)
            # print('ind_t1:', ind_t1, '    num_samples_templ[t1]:', num_samples_templ[t1])
            # print('ind_t2:', ind_t2, '    num_samples_templ[t2]:', num_samples_templ[t2])

            num_combs = num_samples_templ[t1]*num_samples_templ[t2]
            s1_comb = [None] * num_combs
            s2_comb = [None] * num_combs
            for i, (s1, s2) in enumerate(product(ind_t1, ind_t2)):
                s1_comb[i], s2_comb[i] = s1, s2
            # print('c:', c)
            # print('s1_comb:', s1_comb)
            # print('s2_comb:', s2_comb)

            end_idx_sampl_score = beg_idx_sampl_score + num_combs
            scores = templ_sampl_score[beg_idx_sampl_score:end_idx_sampl_score]
            templ_pred_label[c] = predict_templ_label_majority_voting(threshold, scores)
            beg_idx_sampl_score = end_idx_sampl_score

            if c > 0 and c % 10000 == 0:
                print(f'{threshold_idx}/{nrof_thresholds} - threshold: {threshold}    pair: {c}/{len(p1)}')
                # print(f'template: {c}/{len(p1)}')
                # print('t1:', t1)
                # print('t2:', t2)
                stop = timeit.default_timer()
                print('Time: %.2f s.' % (stop - start))
                start = timeit.default_timer()
                # input('PAUSE')
                print('----------------')

            # input('PAUSED')
            # print('--------------')

        tprs[threshold_idx], fprs[threshold_idx], accs[threshold_idx] = calculate_accuracy(threshold, templ_pred_label, templ_true_label)
        tars[threshold_idx], fars[threshold_idx] = calculate_tar_far(threshold, templ_pred_label, templ_true_label)
        # print(f'tprs: {tprs[threshold_idx]}    fprs: {fprs[threshold_idx]}    accs: {accs[threshold_idx]}')
        # print(f'tars: {tars[threshold_idx]}    fars: {fars[threshold_idx]}')
        # print('--------------')

    roc_auc = auc(fprs, tprs)
    print('ROC-AUC = %0.4f %%' % (roc_auc * 100))
    

            


        



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
            print('Finish {}/{} pairs.'.format(c, total_sublists))
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
templ_sampl_score_save_file = os.path.join(save_path, "%s_templ_sampl_score.npy" % target.lower())




# # Step1: Load Meta Data
# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))
# p1, p2, label = read_template_pair_list(os.path.join('%s/meta' % image_path, 'ijbc_template_pair_label_TESTE-BERNARDO.txt'))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))



# Bernardo
if not os.path.exists(img_feats_save_file):
    # # Step 2: Get Image Features

    # In[ ]:

    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()
    # img_path = '%s/loose_crop' % image_path
    img_path = '%s/refined_img' % image_path
    img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
    img_list = open(img_list_path)
    files = img_list.readlines()
    # files_list = divideIntoNstrand(files, rank_size)
    files_list = files

    # img_feats
    # for i in range(rank_size):
    img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                                model_path, 0, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                            img_feats.shape[1]))
    
    # Bernardo
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
    print('Saving img_feats:', img_feats_save_file)
    np.save(img_feats_save_file, img_feats)
    print('Saving faceness_scores:', img_feats_save_file)
    np.save(faceness_scores_save_file, faceness_scores)

else:
    print('Loading img_feats:', img_feats_save_file)
    img_feats = np.load(img_feats_save_file)
    print('Loading faceness_scores:', faceness_scores_save_file)
    faceness_scores = np.load(faceness_scores_save_file)





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
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats



if not args.majority_voting:   # TEMPLATE VERIFICATION

    template_norm_feats, unique_templates = image2template_feature(
        img_input_feats, templates, medias)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))





    # # Step 4: Get Template Similarity Scores
    # In[ ]:

    # =============================================================
    # compute verification scores between template pairs.
    # =============================================================
    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))


    # In[ ]:
    # exper_id = model_path.split('/')[-2]            # Bernardo
    # save_path = os.path.join(result_dir, exper_id)  # Bernardo
    # save_path = os.path.join(result_dir, args.job)
    # save_path = result_dir + '/%s_result' % target

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
    print('Saving scores:', score_save_file)
    np.save(score_save_file, score)
    print('Saving labels:', label_save_file)
    np.save(label_save_file, label)




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
    fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
    print(tpr_fpr_table)

    # Bernardo
    print('AUC = %0.4f %%' % (roc_auc * 100))



elif args.majority_voting:   # MAJORITY VOTING TEMPLATE VERIFICATION

    # count total number of sample matchings
    start = timeit.default_timer()
    print('\nCounting total number of sample matchings...')
    unique_templates = np.unique(templates)
    num_samples_templ, num_total_sample_matchings = count_number_sample_matchings(unique_templates, p1, p2)
    print('num_total_sample_matchings:', num_total_sample_matchings)
    stop = timeit.default_timer()
    print('Time: %.2f s. \n' % (stop - start))

    if not os.path.exists(templ_sampl_score_save_file):
        device = 'cpu'
        # device = 'cuda'

        start = timeit.default_timer()
        templ_sampl_score = compute_score_between_template_samples(img_input_feats, templates, medias, p1, p2, device)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
        print('Saving templ_sampl_score_save_file:', templ_sampl_score_save_file)
        np.save(templ_sampl_score_save_file, templ_sampl_score)
    else:
        print('\nLoading templ_sampl_score_save_file:', templ_sampl_score_save_file)
        templ_sampl_score = np.load(templ_sampl_score_save_file)
    


    print('Evaluating model (majority voting)...')
    metrics = evaluate_model_majority_voting_template(templates, p1, p2, label, templ_sampl_score)

    
    