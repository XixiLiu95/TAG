
from utils import ImageFilelist
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import mmcv
from os.path import dirname
import pickle
import torchvision
from torchvision import datasets, transforms
import os, json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'ViT-L/14' #   ViT-B/32 # 'ViT-L/14' 'RN-50' ,'RN-101# 
model, preprocess = clip.load(model_name, device)

# CIFAR-100
out_file = f"outputs/{model_name}_clip_cifar100_image_feature.pkl"
cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
id_image_features = get_image_features(cifar100_test) # numpy features
mmcv.mkdir_or_exist(dirname(out_file))
with open(out_file, 'wb') as f:
        pickle.dump(id_image_features , f)

# ImageNet-100
out_file = f"outputs/{model_name}_clip_imagenet_100_val_image_feature.pkl"
data_root = "/home/xixi/Desktop/CLIP_based_OOD/datasets/ImageNet100/val"
imagenet_100_test = datasets.ImageFolder(data_root, transform=preprocess)
id_image_features = get_image_features(imagenet_100_test) # numpy features
mmcv.mkdir_or_exist(dirname(out_file))
with open(out_file, 'wb') as f:
        pickle.dump(id_image_features , f)

# ImageNet-1k 
out_file = f"outputs/{model_name}_clip_imagenet_1k_val_image_feature.pkl"
data_root = "/home/xixi/CLIP_based_OOD/dataset/ImageNet1k/val"
img_list = "dataset_preparation/new_generated_list.txt" # the correct list !!!
imagenet_1k_test = ImageFilelist(data_root, img_list, preprocess)
id_image_features = get_image_features(imagenet_1k_test) # numpy features
mmcv.mkdir_or_exist(dirname(out_file))
with open(out_file, 'wb') as f:
        pickle.dump(id_image_features , f)



# get all ood datasets for CIFAR100
out_datasets = ['SVHN','iSUN', 'places365', 'dtd', 'LSUN']

for dataset in out_datasets:
    ood_dataset = set_ood_dataset_cifar_100(dataset, preprocess)
    print(len(ood_dataset))
    ood_image_features = get_image_features(ood_dataset)
    ood_out_file = f"outputs/{model_name}_clip_{dataset}_image_feature.pkl"
    mmcv.mkdir_or_exist(dirname(ood_out_file))
    with open(ood_out_file, 'wb') as f:
        pickle.dump(ood_image_features , f)


out_datasets = ['openimage_o', 'texture', 'inaturalist', 'imagenet_o']

# extract OOD features
for dataset in out_datasets:
    ood_dataset = set_ood_dataset_imagenet_1k(dataset, preprocess)
    ood_image_features = get_image_features(ood_dataset)
    ood_out_file = f"outputs/{model_name}_clip_{dataset}_image_feature.pkl"
    mmcv.mkdir_or_exist(dirname(ood_out_file))
    with open(ood_out_file, 'wb') as f:
        pickle.dump(ood_image_features , f)
