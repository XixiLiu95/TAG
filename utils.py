import random
import numpy as np
import sklearn.metrics as sk
from sklearn import metrics
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import json
from torch.utils.data import Dataset
import pandas as pd

def default_loader(path):
	return Image.open(path).convert('RGB')


def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			data = line.strip().rsplit(maxsplit=1)
			if len(data) == 2:
				impath, imlabel = data
			else:
				impath, imlabel = data[0], 0
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)


def get_img(path):
    img = Image.open(path).convert('RGB')
    return img

def seed_torch(seed=3407): # Torch.manual_seed(3407) is all you need, David Picard.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_image_features(dataset,device):
    all_features = []
    data_loader = DataLoader(dataset, batch_size=100)
    with torch.no_grad():
         for images, labels in tqdm(data_loader):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    return torch.cat(all_features) 


def set_ood_dataset_cifar_100(out_dataset, transform):

    root = "~/datasets/small_OOD_dataset"

    if out_dataset == 'SVHN':
        testsetout = SVHN(root=os.path.join(root, 'svhn'), split='test', transform=transform,download=True)
                                
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'), transform=transform)
        # should rule out four classes when ImageNet-1k is ID data, img_list is provided by vim

    else:
        testsetout = torchvision.datasets.ImageFolder(root = os.path.join(root, out_dataset), transform=transform)
    return testsetout


# get all ood datasets for ImagNet-100 and ImageNet-1k
def set_ood_dataset_imagenet_1k(out_dataset, transform):
    
    if out_dataset == 'openimage_o':
        data_root = "/home/xixi/Downloads/vim-master/data/openimage_o"
        img_list = "/home/xixi/Downloads/vim-master/datalists/openimage_o.txt"
        testsetout =  ImageFilelist(data_root, img_list, transform)

    elif out_dataset == 'texture':
        data_root = "/home/xixi/Downloads/vim-master/data/texture/dtd/images"
        img_list = "/home/xixi/Downloads/vim-master/datalists/texture.txt"
        testsetout =  ImageFilelist(data_root, img_list, transform)
                           
    elif out_dataset == 'inaturalist':
        data_root = "/home/xixi/Downloads/vim-master/data/inaturalist"
        testsetout = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
             
    elif out_dataset == 'imagenet_o':
        data_root = "/home/xixi/Downloads/vim-master/data/imagenet_o"
        testsetout = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
                                     
    return testsetout

def get_text_features(shuffled_text_inputs, M=1):
    all_features = []
    
    with torch.no_grad():
            for m in range(M):
                features = model.encode_text(shuffled_text_inputs[m].to(device))
                all_features.append(features)
    return  all_features

# # calculate ID score 
def calculate_score(img_feature, all_shuffled_text_feature, M, score_name="MSP"):

    img_feature /= img_feature.norm(dim=-1, keepdim=True)
    all_sorted_probability =[]
    all_sorted_logit =[]

    for m in range(M):
        text_features = all_shuffled_text_feature[m]
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logit =   img_feature @ text_features.T/temp
        probability = logit.softmax(dim=-1)
         
        sorted_logit= torch.sort(logit, descending=True).values
        sorted_probability = torch.sort(probability, descending=True).values
        all_sorted_probability.append(sorted_probability)
        all_sorted_logit.append(sorted_logit)
    # average probability
    total_score = 0
    if score_name =="MSP":

        for m in range(M):
            score = all_sorted_probability[m][:,0]
            total_score += score
        total_score = total_score/M
      
    elif score_name =="Max-Logit":
        
        for m in range(M):
            score = all_sorted_logit[m][:,0]
            total_score += score
        total_score = total_score/M
         
    elif score_name =="Energy":

        for m in range(M):
            score  =  torch.logsumexp(all_sorted_logit[m],1)
            total_score += score
        total_score = total_score/M
    
    elif score_name =="GEN":
        gamma = 0.1
        top_M = 100
        
        for m in range(M):
            probs = all_sorted_probability[m][:,:top_M]
            score = torch.sum(probs**gamma * (1 - probs)**(gamma), axis=1)
            total_score += -score
        total_score = total_score/M
    else: 
        
        assert("score name is incorrect")
    
    return total_score.cpu().numpy()



def shuffle_prompt(prompt, class_name):
    # Tokenize the prompt into words (assuming space-separated words)
    revised_prompt = prompt.replace(class_name, "classname")
    words =  revised_prompt.split()

    # Shuffle the words randomly
    np.random.shuffle(words)
    # Reconstruct the shuffled prompt
    shuffled_prompt = ' '.join(words)
    shuffled_prompt = shuffled_prompt.replace("classname", class_name)
    return shuffled_prompt

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out
 


def obtain_ImageNet100_classes():
    loc=os.path.join('data', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set


 
 