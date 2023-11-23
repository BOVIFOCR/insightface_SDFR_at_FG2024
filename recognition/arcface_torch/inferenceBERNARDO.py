import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


def cosine_similarity(embedd1, embedd2):
    embedd1[0] /= np.linalg.norm(embedd1[0])
    embedd2[0] /= np.linalg.norm(embedd2[0])
    sim = float(np.maximum(np.dot(embedd1[0],embedd2[0])/(np.linalg.norm(embedd1[0])*np.linalg.norm(embedd2[0])), 0.0))
    return sim


@torch.no_grad()
def get_face_embedd(model, img):
    embedd = model(img).numpy()
    return embedd


def load_trained_model(network, path_weights):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(path_weights))
    net.eval()
    return net


def load_normalize_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--img1', type=str, default='Aaron_Peirsol_0001.png')
    parser.add_argument('--img2', type=str, default='Aaron_Peirsol_0002.png')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()

    print(f'Loading trained model ({args.network}): {args.weight}')
    model = load_trained_model(args.network, args.weight)

    print(f'Loading and normalizing images {args.img1}, {args.img2}')
    norm_img1 = load_normalize_img(args.img1)
    norm_img2 = load_normalize_img(args.img2)

    print(f'Computing face embeddings')
    face_embedd1 = get_face_embedd(model, norm_img1)
    face_embedd2 = get_face_embedd(model, norm_img2)

    print(f'Computing cosine similarity (0: lowest, 1: highest)')
    sim = cosine_similarity(face_embedd1, face_embedd2)
    print(f'Cosine similarity: {sim}')

    if sim >= args.thresh:
        print('    SAME PERSON')
    else:
        print('    DIFFERENT PERSON')
