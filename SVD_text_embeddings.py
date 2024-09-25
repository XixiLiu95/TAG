import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import sys
import clip
import os
from torchvision.datasets import CIFAR100,SVHN, CIFAR10
from utils import obtain_ImageNet100_classes
from imagenet_classes import imagenet_classes
import mmcv
from os.path import dirname

def get_text_features(tokenized_text, model=None, device="cuda"):
    all_features = []
    
    with torch.no_grad():
                features = model.encode_text(tokenized_text.to(device))
                all_features.append(features)
    return  all_features

def SVD_text_embeddings(dataset, model_name, device):
    model, preprocess = clip.load(model_name, device)
    text_inputs =[]
    if dataset == "CIFAR-100":
        
        cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
        for c in cifar100_test.classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
         
    elif dataset =="ImageNet-100":
        imagenet_100_classes = obtain_ImageNet100_classes()
        for c in imagenet_100_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))

    elif dataset =="ImageNet-1k":
        for c in imagenet_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
    else:

        AssertionError(" dataset not is not defined correctly")
    tokenized_text = torch.cat(text_inputs).to(device)
    id_text_features = get_text_features(tokenized_text, model,device) # numpy features
    cov_weight =  torch.cat(id_text_features)  
    A = cov_weight.detach().cpu().numpy().astype("float32")
    # Perform SVD decomposition
    U, S, VT = np.linalg.svd(A)
    
    plt.figure(figsize=(10, 3))
    # Visualize the original matrix A
    plt.subplot(131)
    plt.imshow(A, cmap='viridis')
    plt.title('Matrix Text Embeddings')
    plt.colorbar(aspect=5,pad = 0.2)
    
    # Visualize the left singular matrix U
    plt.subplot(132)
    plt.imshow(U, cmap='viridis')
    plt.title('Matrix U')
    plt.colorbar(aspect=5, pad = 0.2)
    
    # Visualize the singular values as a diagonal matrix Σ
    plt.subplot(133)
    plt.imshow(np.diag(S), cmap='viridis')
    plt.title('Singular Values (Σ)')
    plt.colorbar( aspect=5, pad = 0.2)
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Show the plots
    file_name = f'SVD_visulization/{dataset}_{model_name}_SVD_WTW_visual.png'
    mmcv.mkdir_or_exist(dirname(file_name))
    plt.suptitle(f'{dataset}_{model_name}', fontsize=16)
    plt.savefig(file_name)
    plt.show()
   
 

def SVD_text_embeddings_singular(dataset, model_name, device):
    model, preprocess = clip.load(model_name, device)
    text_inputs =[]
    if dataset == "CIFAR-100":
        
        cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
        for c in cifar100_test.classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
         
    elif dataset =="ImageNet-100":
        imagenet_100_classes = obtain_ImageNet100_classes()
        #print(imagenet_100_classes)
        for c in imagenet_100_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))

    elif dataset =="ImageNet-1k":
        for c in imagenet_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
    else:

        AssertionError(" dataset not is not defined correctly")
    tokenized_text = torch.cat(text_inputs).to(device)
    id_text_features = get_text_features(tokenized_text, model,device) # numpy features
    cov_weight =  torch.cat(id_text_features)  
    A = cov_weight.detach().cpu().numpy().astype("float32")
    # Perform SVD decomposition
    U, S, VT = np.linalg.svd(A)

    return S


def SVD_text_embeddings_rank(dataset, model_name, device):
    model, preprocess = clip.load(model_name, device)
    text_inputs =[]
    if dataset == "CIFAR-100":
        
        cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
        for c in cifar100_test.classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
         
    elif dataset =="ImageNet-100":
        imagenet_100_classes = obtain_ImageNet100_classes()
        
        for c in imagenet_100_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))

    elif dataset =="ImageNet-1k":
        for c in imagenet_classes:
            text = f"a photo of a {c}"
            text_inputs.append(clip.tokenize(text))
    else:

        AssertionError(" The dataset name is not defined correctly")

    tokenized_text = torch.cat(text_inputs).to(device)
    id_text_features = get_text_features(tokenized_text, model,device) # numpy features
    cov_weight =  torch.cat(id_text_features)  
    A = cov_weight.detach().cpu().numpy().astype("float32")
    # Perform SVD decomposition
    rank = np.linalg.matrix_rank(A)
    return rank


models = ["RN50","ViT-B/16","ViT-B/32", "ViT-L/14","RN101"]
datasets = [ "ImageNet-1k"]#"CIFAR-100" ,"ImageNet-100", "ImageNet-1k"]  
device = "cuda:0"
Singular_Visualization =False
Rank_Visualization = True
fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
for datset in datasets:
    for model_name in  models:
       for dataset in datasets:
           if Rank_Visualization: 
               rank= SVD_text_embeddings_rank(dataset, model_name, device)
               print(f'{dataset}_{model_name} rank is {rank}')
           
           elif Singular_Visualization:
                S = SVD_text_embeddings_singular(dataset, model_name, device)
                
                ax.plot(S,lw=2.5, markersize=10,label =f"{model_name}")
    
    # Visualize the singular values as a diagonal matrix Σ
     
     
   # ax.legend(fontsize=12)
   # ax.set_title(f'{dataset} Singular Values',fontsize=14)
   # ax.set_xlabel('Component',fontsize=14 )
   # ax.set_ylabel('Singular Value',fontsize=14 )
   # ax.set_yscale('log') 
   # #ax.set_ylim(1000,0)
   #  
   # file_name = f'{dataset}_Singular_visual.png'
   # fig.savefig(file_name)
   # plt.show() 
#      