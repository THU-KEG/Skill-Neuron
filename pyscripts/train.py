import os
import torch
import random
import argparse
import numpy as np
from trainer import *
from dataset import *

def main():
    # Arg Parsing
    parser = argparse.ArgumentParser(description="Command line interface for Delta Tuning.")
    parser.add_argument("--model_type", default = "RobertaPrompt", type = str, help = "type of model")
    parser.add_argument("--prompt_size", default = 16, type=int, help="how many words are used as prompting, when set to 0, degenerate to finetuning")
    parser.add_argument("-p","--from_pretrained", action = "store_true", help = "From Pretrained or Not")
    parser.add_argument("--load_backbone",default = "",type = str,help = "Use customized backbone instead of hugging face provided")
    parser.add_argument("--save_to", type = str,default = "")
    parser.add_argument("--resume_from",type = str, default = "")
    
    parser.add_argument("--task_type",default = "sst2",type=str,help="Trained Task")
    parser.add_argument("--data_path",default = "",type = str, help = "Path to the data")
    parser.add_argument("--verb", default = "", type = str, help = "Verbalizers, seperating by commas")
    
    parser.add_argument("--device", default = "cuda:0", type = str)
    parser.add_argument("--random_seed",default = 0,type = int, help = "random seed used")
    parser.add_argument("--epoch",default = 4, type = int,help = "how many epochs")
    parser.add_argument("--eval_every_step",default = 10,type = int, help = "Number of iterations between validation")
    parser.add_argument("--lr","--learning_rate",default = 1e-3,type=float,help="learning rate")
    parser.add_argument("--bz","--batch",default = 8, type = int,  help = "batch size")
    parser.add_argument("--early_stop", action = "store_true", help = "whether apply early stopping")

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
    args.verb = args.verb.split(",")
    if(not os.path.exists(args.save_to)):
        os.makedirs(args.save_to)
    
    
    t=trainer(args)
    t.train(train, valid, test, num_train_epochs = args.epoch)

if __name__ == "__main__":
    main()
