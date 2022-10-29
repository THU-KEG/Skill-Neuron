# Mainly two types of probing
# One to find all the activation of some specific neuron
# One to find the accuracy of all the neuron
import os
import log
import torch
import random
import argparse
import numpy as np
from trainer import *

def cut(data,newsize):
    example = dict()
    size = newsize
    for key in data.keys():
        example[key] = torch.tensor(data[key][:size])
    return example

def main():
    parser = argparse.ArgumentParser(description = "Command line interface for Prompt-Training.")
    parser.add_argument("--info_path",default="",type=str,help = "Path to the info file")
    parser.add_argument("--ths_path",default="",type=str,help = "Path to the ths file")
    parser.add_argument("--save_to", type = str,default = "")
    parser.add_argument("--resume_from",type = str, default = "")
    parser.add_argument("--random_seed",default = 0,type = int, help = "random seed used")
    args=parser.parse_args()
    
    # Randomness Fixing
    seed = args.random_seed
    print("Random seed: ",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if(not os.path.exists(args.save_to)):
        os.makedirs(args.save_to)    



    roberta = AutoModelForMaskedLM.from_pretrained("roberta-base")
    state_dict = roberta.state_dict()
    prompt = torch.load(args.resume_from,map_location="cpu")


    mask = torch.load(args.info_path)
    ths = torch.tensor(torch.load(args.ths_path))
    if(len(ths.shape) == 3):
        print("Aggregating Threshold")
        print(ths.shape)
        ths = ths.mean(axis = 1)
    for k in range(3,12):
        pbias = ths[k]
        pmask = mask[k]
        inputweight =  state_dict['roberta.encoder.layer.'+str(k)+'.intermediate.dense.weight']
        inputbias =  state_dict['roberta.encoder.layer.'+str(k)+'.intermediate.dense.bias']
        outputweight = state_dict['roberta.encoder.layer.'+str(k)+'.output.dense.weight']
        outputbias = state_dict['roberta.encoder.layer.'+str(k)+'.output.dense.bias']
        idx = []
        for i in range(3072):
            if(pmask[i]):
                idx.append(i)
        for i in range(3072):
            if(len(idx) < int(3072*0.02)):
                if(~pmask[i]):
                    idx.append(i)
        inputweight = inputweight[idx,:]
        inputbias = inputbias[idx]
        diffbias = torch.matmul(outputweight, pbias.float())
        outputbias = outputbias+diffbias
        outputweight = outputweight[:,idx]
        state_dict['roberta.encoder.layer.'+str(k)+'.intermediate.dense.weight'] = inputweight
        state_dict['roberta.encoder.layer.'+str(k)+'.intermediate.dense.bias'] = inputbias
        state_dict['roberta.encoder.layer.'+str(k)+'.output.dense.weight'] = outputweight
        state_dict['roberta.encoder.layer.'+str(k)+'.output.dense.bias'] = outputbias 
    for key in state_dict:
        prompt["backbone."+key] = state_dict[key]

    torch.save(prompt,args.save_to+"/best-backbone")

        
        

if __name__ == "__main__":
    main()
