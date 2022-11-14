import os

import log
import torch
import random
import argparse
import numpy as np
from trainer import *


def cut(data,len):
    newdata = dict()
    for key in data:
        newdata[key] = data[key][:len]
    return newdata


def main():
    parser = argparse.ArgumentParser(description="Command line interface for Masking.")
    parser.add_argument("--cmp","--cmp_experiment", action = "store_true", help = "To do control experiment or not")
    parser.add_argument("--info_path",default="",type=str,help = "Path to the info file")
    parser.add_argument("--ths_path",default="",type=str,help = "Path to the ths file")
    parser.add_argument("--lowlayer",default = 0, type = int)    
    parser.add_argument("--highlayer",default = 12,type = int)
    parser.add_argument("--type",default = "gaussian",type = str,help = "Choose from gaussian, mean or zero")
    parser.add_argument("--alpha",default = 1e-1,type = float)

    parser.add_argument("--task_type",default = "sst2",type=str,help="Trained Task")
    parser.add_argument("--data_path",default = "",type = str, help = "Path to the data")
    parser.add_argument("--verb", default = "", type = str, help = "Verbalizers, seperating by commas")


    parser.add_argument("--model_type", default = "RobertaPrompt", type = str, help = "type of model")
    parser.add_argument("--prompt_size", default = 16, type=int, help="how many words are used as prompting, when set to 0, degenerate to finetuning")
    parser.add_argument("-p","--from_pretrained", action = "store_true", help = "From Pretrained or Not")
    parser.add_argument("--save_to", type = str,default = "")
    parser.add_argument("--resume_from",type = str, default = "")

    parser.add_argument("--device", default = "cuda:0", type = str)
    parser.add_argument("--random_seed",default = 0,type = int, help = "random seed used")
    parser.add_argument("--bz","--batch",default = 8, type = int,  help = "batch size")


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
    train, valid, test = getDataset(args.task_type, args.data_path)
    args.num_labels = datasetLabel[args.task_type]

    # Preprocessing args
    seed = args.random_seed
    args.lr = 0

    t=trainer(args)
    if(not os.path.exists(args.save_to)):
        os.makedirs(args.save_to)    
    if args.from_pretrained:
        t.load(args.resume_from)

    t.addmask(args.info_path, thspath = args.ths_path ,cmp = False)
    maskperf = t.eval_dev(test)
    t.logger.info(maskperf)
    if(args.cmp):
        t.addmask(args.info_path, thspath = args.ths_path ,cmp = True)
        randperf = t.eval_dev(test)
        torch.save(randperf, args.save_to+"/cmp_perf")
        t.logger.info(randperf)
    torch.save(maskperf, args.save_to+"/mask_perf")



if __name__ == "__main__":
    main()
