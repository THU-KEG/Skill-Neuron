# Mainly two types of probing
# One to find all the activation of some specific neuron
# One to find the predictivity of all the neuron
from cgi import test
import os
import log
import torch
import random
import argparse
import numpy as np
from trainer import *
from dataset import *

def create_dir(name):
    if(not os.path.exists(name)):
        os.makedirs(name)

def main():
    parser = argparse.ArgumentParser(description = "Command line interface for Skill-Neuron Probing.")
    parser.add_argument("--model_type", default = "RobertaPrompt", type = str, help = "type of model")
    parser.add_argument("--prompt_size", default = 16, type=int, help="how many words are used as prompting, when set to 0, degenerate to finetuning")
    parser.add_argument("-p","--from_pretrained", action = "store_true", help = "From Pretrained or Not")
    parser.add_argument("--load_backbone",default = "",type = str,help = "Use customized backbone instead of hugging face provided")
    parser.add_argument("--resume_from",type = str, default = "")
    parser.add_argument("--save_to",type = str, default = "")

    parser.add_argument("--task_type",default = "sst2",type=str,help="Trained Task")
    parser.add_argument("--data_path",default = "/home/wxz/wky_workspace/skillneuron/grandmother-neuron/data/raw/sst2",type = str, help = "Path to the data")
    parser.add_argument("--verb", default = "", type = str, help = "Verbalizers, seperating by commas")

    parser.add_argument("--probe_type",default = "acc", choices=["pos","acc","acc_mean","acc_max","prompt_act","speed"], type = str, help = "kind of probing, between pos and acc")
    parser.add_argument("--pos", default = "9,3,1893", type = str, help = "position, used when probing a single neuron, seperated by commas")

    parser.add_argument("--device", default = "cuda:0", type = str)
    parser.add_argument("--random_seed",default = 0,type = int, help = "random seed used")
    parser.add_argument("--bz","--batch",default = 8, type = int,  help = "batch size")

    args=parser.parse_args()


    ### Randomness Fixing
    seed = args.random_seed
    print(seed)
    print(args.task_type)
    print("Random seed: ",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Data Preperation
    args.task_type = args.task_type.lower()
    train, valid, test = getDataset(args.task_type, args.data_path)
    args.num_labels = datasetLabel[args.task_type]
    args.lr = 0



    # Preprocessing args
    args.verb = args.verb.split(",")


    if not args.save_to:
        args.save_to = args.resume_from
    args.save_to = args.save_to + "/" + args.task_type
    create_dir(args.save_to)


    t=trainer(args)
    if args.from_pretrained or args.load_backbone:
        t.load(args.resume_from)

    # Test subnetwork temporary
    if args.add_mask:
        t.addmask(args.add_mask, thspath = args.ths_path ,cmp = False)



    if(args.probe_type == "pos"):
        print(args.pos)
        pos = [int(_) for _ in args.pos.split(",")]
        create_dir(args.save_to + "/pos_dev")
        create_dir(args.save_to + "/pos_test")
        torch.save(t.probe_pos(valid, pos),args.save_to + "/pos_dev/" + args.pos)
        torch.save(t.probe_pos(test, pos),args.save_to + "/pos_test/"+args.pos)


    if(args.probe_type == "acc"):
        torch.save(t.probe_avg(train),args.save_to + "/train_avg")
        torch.save(t.probe_acc(valid, args.save_to + "/train_avg"),args.save_to+"/dev_perf")
        torch.save(t.probe_acc(test, args.save_to + "/train_avg"),args.save_to+"/test_perf")


    if(args.probe_type == "acc_mean"):
        torch.save(t.probe_avg_mean(train),args.save_to + "/train_mean_avg")
        torch.save(t.probe_acc_mean(valid, args.save_to + "/train_mean_avg"),args.save_to+"/dev_mean_perf")
        torch.save(t.probe_acc_mean(test, args.save_to + "/train_mean_avg"),args.save_to+"/test_mean_perf")

    if(args.probe_type == "acc_max"):
        torch.save(t.probe_avg_max(train),args.save_to + "/train_max_avg")
        torch.save(t.probe_acc_max(valid, args.save_to + "/train_max_avg"),args.save_to+"/dev_max_perf")
        torch.save(t.probe_acc_max(test, args.save_to + "/train_max_avg"),args.save_to+"/test_max_perf")


    if(args.probe_type == "prompt_act"):
        torch.save(t.probe_prompt(valid, pos),args.save_to + "/prompt_activation")

    if(args.probe_type == "speed"):
        times = []
        perfs = []
        import time
        for _ in range(3):
            tic = time.time()
            perfs.append(t.eval_dev(test))
            toc = time.time()
            times.append(toc-tic)
            t.logger.info("time: ")
            t.logger.info(toc-tic)
            t.logger.info("perf: ")
            t.logger.info(perfs[-1])
        torch.save(perfs,args.save_to+"/inference_perf")
        torch.save(times,args.save_to+"/inference_time")
        

if __name__ == "__main__":
    main()
