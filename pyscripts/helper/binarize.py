# Transform the dataset of n classes, n > 2 to 2 binary datasets
import os
import log
import torch
import random
import argparse
import numpy as np
from trainer import *
from dataset import *
from datasets import DatasetDict


def balance(data):
    pos = sum(data['label'])
    neg = len(data['label']) - pos
    ths = min(pos,neg)
    cnt = [0,0]
    def filter(sample):
        try:
            cnt[sample['label']] +=  1
            return (cnt[sample['label']] < ths)
        except:
            print(sample)
            raise NotImplementedError
    return data.filter(filter)
             


def main():
    parser = argparse.ArgumentParser(description="Command line interface for Prompt-Training.")
    parser.add_argument("--task_type",default="mnli",type=str,help="which glue task is training")
    parser.add_argument("--data_path",default = "",type = str)
    parser.add_argument("--save_to",default = "",type = str)
    parser.add_argument("--random_seed", default = 0, type = int)
    args=parser.parse_args()

    # Randomness Fixing
    seed = args.random_seed
    seed += hash(args.task_type)%(2**32)
    print("Random seed: ",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Data Preperation
    args.task_type = args.task_type.lower()
    datas = load_from_disk(args.data_path)
    n = datasetLabel[args.task_type]

    if(not os.path.exists(args.save_to)):
        os.makedirs(args.save_to)    

    pn_datas = {}
    neu_datas = {}
    if(n == 3):
        pn = [0,2]
        neu = 1
    elif(n == 4):
        pn = [1,3]
        neu = 2
    for _ in datas.keys():
        print(_)
        data = datas[_]
        data_pn = data.filter(lambda sample: (sample['label'] in pn))
        def process(sample):
            sample['label'] = sample['label']//2
            return sample
        data_pn = data_pn.map(process)
        def process_neu(sample):
            sample['label'] = (sample['label'] == neu)
            return sample
        data_neu = data.map(process_neu)
        data_pn = balance(data_pn)
        data_neu = balance(data_neu)
        pn_datas[_] = data_pn
        neu_datas[_] = data_neu
    pn_datas = DatasetDict(pn_datas)
    neu_datas = DatasetDict(neu_datas)
    pn_datas.save_to_disk(args.save_to + "/" + args.task_type+"-pn")
    neu_datas.save_to_disk(args.save_to + "/" + args.task_type+"-neu")





if __name__ == "__main__":
    main()
