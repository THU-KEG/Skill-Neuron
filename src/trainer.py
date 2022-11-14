# Trainer Class
from unittest.util import _MAX_LENGTH
import torch
import log
import json
import numpy as np
from modelzoo import *
from dataset import *
from typing import Dict
from torch.optim import Adam
from tqdm import tqdm,trange
from datasets import load_metric
from transformers import get_linear_schedule_with_warmup
from transformers.data.metrics import glue_compute_metrics
from torch.utils.data import RandomSampler,DataLoader,SequentialSampler,Dataset

'''trainer class'''
class trainer:
    def __init__(self, args):
        self.args = args
        self.getModel()
        logger = log.get_logger(args.save_to+"/log")
        self.logger = logger
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        optimizer_parameters = self.model.optimize_parameters()
        self.optimizer = Adam(optimizer_parameters, lr=self.args.lr,eps=1e-6,betas=(0.9,0.98))
        self.criterion=torch.nn.CrossEntropyLoss()
        self.maskhandle = False

        logger.info(args)

    def getModel(self):
        # Getting the required model
        args = self.args
        if("Roberta" in args.model_type):
            if("Prompt" in args.model_type):
                if("Prune" in args.model_type):
                    self.model = RobertaPrunePrompt(args)
                else:
                    self.model = RobertaPrompt(args)
            elif("Bias" in args.model_type):
                self.model = RobertaBias(args)
            elif("Adapter" in args.model_type):
                self.model = RobertaAdapter(args)
    
    #training step
    def mlm_train_step(self,labeled_batch,return_length=False) -> torch.Tensor:
        """Perform a MLM training step."""
        sentences = labeled_batch[datasetRow[self.args.task_type][0]]
        if(len(datasetRow[self.args.task_type]) > 2):
            sentences = [sentences[_] + labeled_batch[datasetRow[self.args.task_type][1]][_] for _ in range(len(sentences))]
        sentences = self.model.tokenizer(sentences, padding = True, truncation = True,  return_tensors="pt"\
            , max_length = 512 - self.args.prompt_size)
        sentences = {k: t.to(self.device) for k, t in sentences.items()}
        outputs = self.model(sentences)
        if(return_length):
            length = sentences["attention_mask"].sum(dim = 1) 
        del sentences
        if(return_length):
            return length
        else:
            return outputs


    def train(self,
              train,
              valid,
              test,
              n_gpu: int = 1,
              num_train_epochs: int = 40,
              max_grad_norm: float = 1,
              logging_steps: int = 100):

        train_sampler = RandomSampler(train)
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=self.args.bz)
        t_total = len(train_dataloader) * num_train_epochs

        self.logger.info("num_steps_per_dataset:")
        self.logger.info(len(train_dataloader))
        self.logger.info("total_steps:")
        self.logger.info(t_total)
        self.logger.info("num_train_epochs:")
        self.logger.info(num_train_epochs)


        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=t_total*0.006, num_training_steps=t_total)

        if(self.args.from_pretrained or self.args.load_backbone):
            self.load(self.args.resume_from)
        
        best_dev_scores = 0.0
        best_global_step = 0
        best_loss = 0.0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0


        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        eval_every_step = self.args.eval_every_step

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            stop = False
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                labels = batch[datasetRow[self.args.task_type][-1]].to(self.device)
                loss = self.criterion(self.mlm_train_step(batch),labels.long())
                del labels
                if n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                global_step += 1
                if logging_steps > 0 and global_step % logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    learning_rate_scalar = self.scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    self.logger.info(json.dumps({**logs, **{'step': global_step}}))
                if global_step % eval_every_step == 0:
                    origin_dev_scores = self.eval_dev(valid)
                    # Shortcut
                    dev_scores = origin_dev_scores[list(origin_dev_scores.keys())[-1]]
                    if dev_scores >= best_dev_scores:
                        best_dev_scores = dev_scores
                        best_global_step = global_step
                        best_loss = tr_loss
                        self.logger.info(origin_dev_scores)
                        self.logger.info("Saving trained model at {}...".format(self.args.save_to))
                        self.logger.info("best_dev_perf: %.4f | best_global_step: %d" %(best_dev_scores, best_global_step))
                        self.save(self.args.save_to,global_step)
                    else:
                        self.logger.info(origin_dev_scores)
                        self.logger.info(global_step)
                if self.args.early_stop and (global_step - best_global_step >= 6*eval_every_step):
                    stop = True
                    break
            if stop:
                epoch_iterator.close()
                break
        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)
    
    def eval_dev(self, dev_data):
        self.model.eval()
        results = self.eval(dev_data)
        predictions = np.argmax(results['logits'], axis=1)
        try:
            scores = glue_compute_metrics(self.args.task_type,predictions,results['labels'])
        except:
            scores = {"acc":np.sum(predictions == results['labels'])/predictions.shape[0]}
        return scores

    def eval(self,
             valid) -> Dict:
        eval_sampler = SequentialSampler(valid)
        eval_dataloader = DataLoader(valid, sampler=eval_sampler, batch_size=self.args.bz)

        preds = None
        out_label_ids, question_ids = None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            labels = batch[datasetRow[self.args.task_type][-1]].to(self.device)
            with torch.no_grad():
                logits = self.mlm_train_step(batch)
                eval_loss=self.criterion(logits,labels.long())
                eval_losses.append(eval_loss.item())
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            del labels
        return {
            "eval_loss": np.mean(eval_losses),
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def save(self, path: str, global_step) -> None:
        self.logger.info("Saving models.")
        self.model.save(path + '/best')
        self.model.save(path + '/'+ str(global_step))
        optimize = self.optimizer.state_dict()
        torch.save(optimize,path + "/optimizer")
        schedule = self.scheduler.state_dict()
        torch.save(schedule,path + "/scheduler")
    
    def load(self, path: str):
        self.logger.info("Loading models from")
        self.logger.info(path)
        self.model.load(path+'/best')
        try:
            optimize = torch.load(path + "/optimizer")
            self.optimizer.load_state_dict(optimize)
        except:
            self.logger.info("fail getting optimizer")
        try:
            schedule = torch.load(path + "/scheduler")
            self.scheduler.load_state_dict(schedule)
        except:
            self.logger.info("fail getting scheduler")

    def probe(self,
            eval_data,
            process):
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.bz)
        hidden = [[] for _ in range(self.model.layer_num)]
        def forward_hook(n):
            def fn(_,__,output):
                hidden[n].append(output)
            return fn
        handle = [self.model.intermediate(n).register_forward_hook(forward_hook(n)) for n in range(12)]
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            hidden = [[] for _ in range(self.model.layer_num)]
            # from IPython import embed;embed()
            labels = batch[datasetRow[self.args.task_type][-1]].to(self.device)
            with torch.no_grad():
                length = self.mlm_train_step(batch,True)
            process(hidden,labels,length)
            del labels
        for h in handle:
            h.remove()
                
            
    
    def probe_pos(self,
             eval_data,
             pos) -> Dict:
        activation = []
        def process(hidden,label,length):
            activation.append(hidden[pos[0]][0][:,:,pos[1]][:, : self.args.prompt_size].detach().cpu().numpy())
        self.probe(eval_data, process)
        activation = np.concatenate(activation)
        return activation
    
    def probe_avg(self,
             eval_data) -> Dict:
        ths = np.zeros((self.model.layer_num, self.model.prompt_size, self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            # Bert, Roberta
            def process(hidden,label,length):
                for k in range(self.model.layer_num):
                    hidden[k][0] = hidden[k][0][:,:self.model.prompt_size,:].detach().cpu().numpy()
                    ths[k] += hidden[k][0].sum(axis = 0)
        elif(self.model.prompt_type == "GPT"):
            def process(hidden,label,length):
                for k in range(self.model.layer_num):
                    hidden[k][0] = hidden[k][0][:,-self.model.prompt_size:,:].detach().cpu().numpy()
                    ths[k] += hidden[k][0].sum(axis = 0)
        else:
            raise NotImplementedError
        self.probe(eval_data, process)
        ths = ths/len(eval_data)
        return ths
    
    def probe_acc(self,
             eval_data,
             thspath) -> Dict:
        ths = torch.load(thspath)
        acctable = np.zeros((self.model.layer_num,self.model.prompt_size,self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            def process(hidden,label,length):
                label = label.detach().cpu().unsqueeze(dim = 1).unsqueeze(dim = 1).numpy()
                for k in range(self.model.layer_num):
                    hidden[k][0] = hidden[k][0][:,:self.model.prompt_size,:].detach().cpu().numpy()
                    pred = (hidden[k][0] > ths[k])
                    acctable[k] = acctable[k] + (pred == label).sum(axis = 0)
        elif(self.model.prompt_type == "GPT"):
            def process(hidden,label,length):
                label = label.detach().cpu().unsqueeze(dim = 1).unsqueeze(dim = 1).numpy()
                for k in range(self.model.layer_num):
                    hidden[k][0] = hidden[k][0][:,-self.model.prompt_size:,:].detach().cpu().numpy()
                    pred = (hidden[k][0] > ths[k])
                    acctable[k] = acctable[k] + (pred == label).sum(axis = 0)
        else:
            raise NotImplementedError
        self.probe(eval_data, process)
        return (acctable/len(eval_data))


    def probe_avg_mean(self,
             eval_data) -> Dict:
        ths = np.zeros((self.model.layer_num,self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            offset = self.model.prompt_size
        elif(self.model.prompt_type == "GPT"):
            offset = 0
        else:
            offset = 0
        def process(hidden,label,length):
            for k in range(self.model.layer_num):
                hidden[k][0] = hidden[k][0].detach().cpu().numpy()
                for snt in range(len(length)):
                    ths[k] += hidden[k][0][snt,offset:offset + length[snt],:].mean(axis = 0)
        self.probe(eval_data,process)
        datalen = len(eval_data)
        ths = ths/datalen
        return ths

    def probe_avg_max(self,
             eval_data) -> Dict:
        ths = np.zeros((self.model.layer_num,self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            offset = self.model.prompt_size
        elif(self.model.prompt_type == "GPT"):
            offset = 0
        else:
            offset = 0
        def process(hidden,label,length):
            for k in range(self.model.layer_num):
                hidden[k][0] = hidden[k][0].detach().cpu().numpy()
                for snt in range(len(length)):
                    ths[k] += hidden[k][0][snt,offset:offset + length[snt],:].max(axis = 0)
        self.probe(eval_data,process)
        datalen = len(eval_data)
        ths = ths/datalen
        return ths

    def probe_acc_mean(self,
             eval_data,
             thspath) -> Dict:
        ths = torch.load(thspath)
        acctable = torch.zeros((self.model.layer_num,self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            offset = self.model.prompt_size
        elif(self.model.prompt_type == "GPT"):
            offset = 0
        else:
            raise NotImplementedError
        def process(hidden,label,length):
            for k in range(self.model.layer_num):
                hidden[k][0] = hidden[k][0].detach().cpu().numpy()
                for snt in range(length.shape[0]):
                    pred = hidden[k][0][snt,offset :offset + length[snt],:].mean(axis = 0)
                    pred = (pred > ths[k])
                    acctable[k] += (pred == bool(label[snt])) 
        self.probe(eval_data,process)     
        return (acctable/len(eval_data))


    def probe_acc_max(self,
             eval_data,
             thspath) -> Dict:
        ths = torch.load(thspath)
        acctable = torch.zeros((self.model.layer_num,self.model.layer_width))
        if(self.model.prompt_type == "MLM"):
            offset = self.model.prompt_size
        elif(self.model.prompt_type == "GPT"):
            offset = 0
        else:
            raise NotImplementedError
        def process(hidden,label,length):
            for k in range(self.model.layer_num):
                hidden[k][0] = hidden[k][0].detach().cpu().numpy()
                for snt in range(length.shape[0]):
                    pred = hidden[k][0][snt,offset:offset + length[snt],:].max(axis = 0)
                    pred = (pred > ths[k])
                    acctable[k] += (pred == bool(label[snt]))    

        self.probe(eval_data,process)
        return (acctable/len(eval_data))


    def probe_prompt(self,
             eval_data) -> Dict:
        activation = []
        def process(hidden,label,length):
            activation.append(hidden[:,0,:].detach().numpy().cpu())
        self.probe(eval_data,process)
        activation = np.concatenate(activation)
        return activation

    def addmask(self, maskpath, thspath = None, cmp = False):
        # mask all informative neuron 
        self.model.pmask = torch.load(maskpath)
        if cmp:
            for k in range(self.model.layer_num):
                N = (self.model.pmask[k]).sum()
                idx = torch.randperm(self.model.layer_width)
                self.model.pmask[k] = 0
                for x in range(N):
                    self.model.pmask[k,idx[x]] = 1
        self.model.pmask = self.model.pmask.to(self.device)
        if(not self.maskhandle):
            self.maskhandle = True
            lowlayer = self.args.lowlayer
            highlayer = self.args.highlayer
            type = self.args.type
            self.model.addmask(thspath, lowlayer, highlayer, type)