import os
from requests import get
import torch
import argparse
import numpy as np
from trainer import *
from dataset import *
def create_dir(name):
    if(not os.path.exists(name)):
        os.makedirs(name)

def get_agg_accs(name):
    accs = []
    for path in name:
        try:
            acc = torch.tensor(torch.load(path))
            print("Loaded acc from "+path)
            accs.append(acc)
        except:
            print("Cant find acc at ")
            continue
    accs = torch.stack(accs)
    print(accs.shape)
    return accs

def finfo(acc,p, type = "mean"):
    if(p == 0):
        return torch.zeros((12,3072)).bool()
    acc = abs(acc - 0.5)
    print(acc.shape)
    if(type == "mean"):
        h = acc.max(axis = 2).values.mean(axis = 0)
    else:
        h = acc.max(axis = 2).values.min(axis = 0).values
    print(h.shape)
    inf = []
    for l in range(12):
        idx = sorted(range(3072),key = lambda i: -h[l][i])
        inf.append(torch.zeros((3072)))
        for j in range(int(p*3072)):
            inf[-1][idx[j]]  = 1
    return torch.stack(inf).bool()

def main():
    parser = argparse.ArgumentParser(description = "Command line interface for Prompt-Training.")
    parser.add_argument("--task_type",default = "sst2",type=str,help="which glue task is training")
    parser.add_argument("--save_to",type = str, default = "", help = "path to save skillneuron")
    parser.add_argument("--type", type = str, default = "mean", help = "aggregation method, choose from mean or min")
    parser.add_argument("-a","--acc_path",default = "",type = str, help = "path of probed acc to generate skill neuron, seperated by comma")
    parser.add_argument("-aa","--acc_aux_path",default = "",type = str, help = "path of probed acc to generate skill neuron for the auxiliary task (for multilabel task),seperated by comma")
    args=parser.parse_args()

    args.acc_path = args.acc_path.split(",")
    args.acc_aux_path = args.acc_aux_path.split(",")
    args.task_type = args.task_type.lower()
    num_labels = datasetLabel[args.task_type]
    multi_class = num_labels > 2
    if(multi_class):
        step_size = 0.005
    else:
        step_size = 0.01

    args.save_to = args.save_to + "/info/" + args.task_type
    create_dir(args.save_to)
     
    agginfo = dict()
    length = 21

    if(not multi_class):
        accs = get_agg_accs(args.acc_path)
        plist = [step_size * _ for _ in range(length)]
        for p in plist:
            info = finfo(accs, p,args.type)
            print("Calculated top ",info.sum())
            torch.save(info, args.save_to+"/"+str(p))
    if(multi_class):
        bothacc = [get_agg_accs(args.acc_path),get_agg_accs(args.acc_aux_path)]
        plist = [step_size * _ for _ in range(length)]
        for p in plist:
            info = finfo(bothacc[0], p, args.type)
            if(p == 0):
                torch.save(info, args.save_to+"/"+str(p))
            else:
                acc = bothacc[1]
                acc = abs(acc - 0.5)
                if(args.type == "mean"):
                    h = acc.max(axis = 2).values.mean(axis = 0)
                else:
                    h = acc.max(axis = 2).values.min(axis = 0).values
                def perf(k):return -h[l][k]
                for l in range(12):
                    order = sorted(range(3072),key=perf)
                    j = 0
                    while(info[l].sum()!=int(3072*2*p)):
                        info[l][order[j]] = 1
                        j += 1 
                torch.save(info, args.save_to+"/"+str(2*p))





if __name__ == "__main__":
    main()
