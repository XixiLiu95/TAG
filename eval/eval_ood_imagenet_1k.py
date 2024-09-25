import os
import clip
import torch
import torchvision
import sklearn.metrics as sk
from torchvision.transforms import transforms
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from svhn_loader import SVHN
recall_level_default = 0.95

import numpy as np
import sklearn.metrics as sk
from sklearn import metrics
import pandas as pd
from list_dataset import ImageFilelist
from imagenet_classes import imagenet_classes
import mmcv
from os.path import dirname
from utils import seed_torch, get_text_features, calculate_score, shuffle_prompt, auc, num_fp_at_recall, fpr_recall
import argparse 


 
def parse_args():
    parser = argparse.ArgumentParser(description='OOD detection on ImageNet-1k ')
    parser.add_argument('--model_name', default = 'RN101', type=str, choices = ['ViT-B/16', 'ViT-B/32','ViT-L/14','RN50','RN101'],
                        help='name of pre-trained CLIP')
    parser.add_argument('--shuffle', action='store_true',
                            help='shuffling the text prompt')
    parser.add_argument('--M', default=10, type=int, help='number of text augmentations')
    parser.add_argument('--temp', default=0.01, type=float,
                        help='temperature scaling')
    parser.add_argument('--score_name', default = 'MSP', type=str, choices = ['MSP', 'Max-Logit','Energy','GEN'],
                        help='score functions')
    return parser.parse_args()

# Load the model
def main():
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    model_name = args.model_name  
    temp = args.temp
    M = args.M
    SHUFFLE = args.shuffle
    score_name = args.score_name   
    model, preprocess = clip.load(model_name, device)
    
     
    M_shuffled_text_inputs = [] 
    for times in range(M):
        text_inputs_shuffled = []
        for c in imagenet_classes:
            if SHUFFLE:
                text = f"a photo of a classname" # Ensure that if the class name consists of two words, they remain together after shuffling.
                text = shuffle_prompt(text,c)
            else: 
                text = f"a photo of a {c}"
            text_inputs_shuffled.append(clip.tokenize(text))
        text_inputs = torch.cat(text_inputs_shuffled).to(device)
        M_shuffled_text_inputs.append(text_inputs)
    
    # Calculate ID image/text features
    out_file = f"outputs/{model_name}_clip_imagenet_val_image_feature.pkl"
    id_image_features = mmcv.load(out_file)
    id_text_features = get_text_features(M_shuffled_text_inputs,M) # numpy features
    
    id_score = calculate_score(id_image_features, id_text_features, M, score_name) 
     
    
    out_datasets = ['openimage_o', 'texture', 'inaturalist', 'imagenet_o']
    result = []
    for dataset in out_datasets:
     
        ood_out_file = f"outputs/{model_name}_clip_{dataset}_image_feature.pkl"
        ood_image_features = mmcv.load(ood_out_file)
        ood_score = calculate_score(ood_image_features, id_text_features, M, score_name)
        ## save scores ##
        auroc = auc(id_score, ood_score)[0]
        fpr, _ = fpr_recall(id_score , ood_score, recall_level_default ) 
        result.append(
                dict(method=score_name, oodset=dataset, auroc=round(auroc*100,2) , fpr=round(fpr*100,2) ))
        
    df = pd.DataFrame(result)
    results_output_file = f'results/{model_name}_{score_name}_temperature_{temp}_augmentations_{M}_imagenet_1k_results.csv'
    mmcv.mkdir_or_exist(dirname(results_output_file))
    df.to_csv(results_output_file )
    print ("shuffle is ", SHUFFLE)
    print("model name: ", model_name + "\n" + "score name: ", score_name)
    print("temperature=", temp, "number of augmentations = ", M)
    print(f'&{df.auroc[0]}&{df.fpr[0]}&{df.auroc[1]}&{df.fpr[1]} &{df.auroc[2]}&{df.fpr[2]}&{df.auroc[3]}&{df.fpr[3]}&{df.auroc.mean():.2f}&{df.fpr.mean():.2f}')
    
            
    
    
    
     