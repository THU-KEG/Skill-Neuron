# Containing all the models
# Three types of them 
# RobertaPrompt
# RobertaBias
# RobertaAdapter


from re import L
from turtle import forward
import torch
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer,AdapterConfig,AutoModelWithLMHead, GPT2LMHeadModel, GPT2Tokenizer,AutoModelForPreTraining


modelpath = "/data/MODELS/"
gptprepath = "/data/wxz/gpt-pretrained/"
bertprepath = "/data/wxz/bert-pretrained/"

### Basemodel
class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.getmodel()
        self.args = args
        self.prompt_type = None
    def getmodel(self):
        pass 
    def forward(self, sentences):
        pass
    def intermediate(self,n):
        pass
    def embed(self, input_ids):
        pass
    def getverablizer(self):
        args = self.args
        print(args.verb)
        if(len(args.verb) == 0 or args.verb[0] == ''):
            positive = self.tokenizer.encode("positive")[1]
            negative = self.tokenizer.encode("negative")[1]
            neutral = self.tokenizer.encode("neutral")[1]
            conflict = self.tokenizer.encode("conflict")[1]
            if(self.args.num_labels == 2): 
                self.pos = [negative,positive]
            if(self.args.num_labels == 3):
                self.pos = [negative,neutral,positive]
            if(self.args.num_labels == 4):
                self.pos = [conflict,negative,neutral,positive]
        elif(len(args.verb) == 1):
            self.pos = random.sample(list(range(50265)),self.num_labels)
        else:
            self.pos = [self.tokenizer.encode(word)[1] for word in args.verb]
        print(self.pos)
        print(len(self.pos))
    def processoutput(self, output):
        pass
    def save(self, path):
        pass
    def load(self, path):
        pass
    def optimize_parameters(self):
        pass
    def addmask(self, thspath, lowlayer = 0, highlayer = 12, type = "mean"):
        if(type == "mean"):
            self.bias = torch.tensor(torch.load(thspath)).cuda().mean(axis = 1)
            # from IPython import embed;embed()
        elif(type == "zero"):
            self.bias = [0]*12
        if(type != "gaussian"):
            def save_std_outputs1_hook(k):
                def fn(_,__,output):
                    cmask = self.pmask[k]
                    bias = self.bias[k]
                    bias = bias*cmask
                    # from IPython import embed;embed()
                    output = output*(~cmask)
                    output += bias
                    return output
                return fn
            for k in range(lowlayer,highlayer):
                self.intermediate(k).register_forward_hook(save_std_outputs1_hook(k))
        else:
            def save_std_outputs1_hook(k):
                def fn(_,__,output):
                    cmask = self.pmask[k]
                    bias = torch.randn([output.shape[0],3072]).cuda()*self.args.alpha
                    bias = bias*cmask
                    output += bias.unsqueeze(dim = 1)
                    return output
                return fn
            for k in range(lowlayer,highlayer):
                self.intermediate(k).register_forward_hook(save_std_outputs1_hook(k))


class PromptBaseModel(BaseModel):
    def __init__(self,args):
        super().__init__(args)
    def save(self, path):
        parameter = {}
        state = self.state_dict()
        parameter["prompt"] = state["prompt"]
        parameter["pos"] = self.pos
        torch.save(parameter,path + "-backbone")
    def load(self, path):
        # if(self.args.load_backbone):
        #     print("loading backbone from "+self.args.load_backbone)
        #     self.backbone.load_state_dict(torch.load(self.args.load_backbone))
        if(self.args.from_pretrained):
            parameter = torch.load(path + "-backbone")
            state = self.state_dict()
            state["prompt"] = parameter["prompt"]
            self.pos = parameter["pos"]
            self.load_state_dict(state)
    def optimize_parameters(self):
        return [{'params': [p for n,p in self.named_parameters() if "prompt" in n],'weight_decay': 0.0}]
        


class MLMPromptBaseModel(PromptBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.prompt_size = args.prompt_size
        mask_embedding = self.embed(torch.tensor(self.tokenizer.mask_token_id)) 
        self.prompt = torch.nn.Parameter(torch.randn(args.prompt_size-1, mask_embedding.shape[0]).unsqueeze(dim = 0)/30)
        self.mask = torch.nn.Parameter(((torch.ones(1,1) * mask_embedding).unsqueeze(dim = 0)))
        self.pmask = torch.zeros(self.layer_num,self.layer_width)
        self.getverablizer()
        self.prompt_type = "MLM"
    def forward(self, sentences):
        input_ids = sentences['input_ids']
        inputs_embeds = self.embed(input_ids)
        attention_mask = sentences["attention_mask"]
        final_embeds=torch.cat((self.mask.repeat(inputs_embeds.shape[0],1,1),torch.cat((self.prompt.repeat(inputs_embeds.shape[0],1,1),inputs_embeds),dim=1)),dim = 1)
        if(self.args.device != "cpu"):
            prompt_attention_mask=torch.ones((inputs_embeds.shape[0],self.prompt_size)).cuda()
        else:
            prompt_attention_mask=torch.ones((inputs_embeds.shape[0],self.prompt_size))
        final_mask=torch.cat((prompt_attention_mask,attention_mask),dim=1)
        outputs = self.backbone(
            inputs_embeds=final_embeds,
            attention_mask=final_mask)
        return self.processoutput(outputs)


class GPTPromptBaseModel(PromptBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.prompt_size = args.prompt_size
        self.prompt = torch.nn.Parameter(torch.randn(args.prompt_size,768).unsqueeze(dim = 0)/30)
        self.pmask = torch.zeros(self.layer_num,self.layer_width)
        self.getverablizer()
        self.prompt_type = "GPT"
    def forward(self, sentences):
        input_ids = sentences['input_ids']
        inputs_embeds = self.embed(input_ids)
        attention_mask = sentences["attention_mask"]
        final_embeds= torch.cat((inputs_embeds,self.prompt.repeat(inputs_embeds.shape[0],1,1)),dim=1)
        prompt_attention_mask=torch.ones((inputs_embeds.shape[0],self.prompt_size)).cuda()
        final_mask=torch.cat((attention_mask,prompt_attention_mask),dim=1)
        outputs = self.backbone(inputs_embeds=final_embeds,attention_mask = final_mask)
        return self.processoutput(outputs)





class MLMBaseModel(BaseModel):
    def __init__(self,args):
        super().__init__(args)
        self.pmask = torch.zeros(12,3072)
        self.getverablizer()
    def forward(self,sentences):
        input_ids = sentences['input_ids']
        inputs_embeds = self.embed(input_ids)
        attention_mask = sentences["attention_mask"]
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask)
        return self.processoutput(outputs)





class AdapterBaseModel(BaseModel):
    def __init__(self,args):
        super().__init__(args)
        self.pmask = torch.zeros(12,3072)
        self.getverablizer()
        config = AdapterConfig.load("pfeiffer")
        self.backbone.add_adapter("dummy", config=config)
        self.backbone.set_active_adapters("dummy")

    def forward(self,sentences):
        input_ids = sentences['input_ids']
        inputs_embeds = self.embed(input_ids)
        attention_mask = sentences["attention_mask"]
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask)
        return self.processoutput(outputs)
    def optimize_parameters(self):
        return [{'params': [p for n,p in self.named_parameters() if "dummy" in n],'weight_decay': 0.0}]










### RobertaPrompt
class RobertaPrompt(MLMPromptBaseModel):
    def __init__(self, args):
        self.tokenizer= AutoTokenizer.from_pretrained("roberta-base")
        self.layer_num = 12
        self.layer_width = 3072
        super().__init__(args)
    def getmodel(self):
        self.backbone= AutoModelForMaskedLM.from_pretrained("roberta-base")
    def processoutput(self, outputs):
        return outputs.logits[:,0,self.pos].squeeze(dim = 1)
    def intermediate(self,n):
        return self.backbone.roberta.encoder.layer[n].intermediate
    def embed(self, input_ids):
        return self.backbone.roberta.embeddings.word_embeddings(input_ids).detach()



### RobertaPruneRoberta

class RobertaPrunePrompt(MLMPromptBaseModel):
    def __init__(self, args):
        self.tokenizer= AutoTokenizer.from_pretrained("roberta-base")
        self.layer_num = 12
        self.layer_width = 3072
        super().__init__(args)
    def getmodel(self):
        self.backbone = torch.load("prune_structure/PruneRoberta")
    def processoutput(self, outputs):
        return outputs.logits[:,0,self.pos].squeeze(dim = 1)
    def intermediate(self,n):
        return self.backbone.roberta.encoder.layer[n].intermediate
    def embed(self, input_ids):
        return self.backbone.roberta.embeddings.word_embeddings(input_ids).detach()
    def load(self, path):
        parameter = torch.load(path + "-backbone")
        self.pos = parameter["pos"]
        self.load_state_dict(parameter, strict = False)
    def save(self, path):
        parameter = self.state_dict()
        parameter["pos"] = self.pos
        torch.save(parameter,path + "-backbone")






### RobertaBias
class RobertaBias(MLMBaseModel):
    def __init__(self, args):
        self.tokenizer= AutoTokenizer.from_pretrained(modelpath+"roberta-base")
        self.layer_num = 12
        self.layer_width = 3072
        super().__init__(args)
    def getmodel(self):
        self.backbone = AutoModelForMaskedLM.from_pretrained(modelpath+"roberta-base")
    def processoutput(self, outputs):
        return outputs.logits[:,0,self.pos].squeeze(dim = 1)
    def intermediate(self,n):
        return self.backbone.roberta.encoder.layer[n].intermediate
    def embed(self, input_ids):
        return self.roberta.roberta.embeddings.word_embeddings(input_ids).detach()
    def optimize_parameters(self):
        return [{'params': [p for n,p in self.named_parameters() if "bias" in n],'weight_decay': 0.0}]



### RobertaAdapter
class RobertaAdapter(AdapterBaseModel):
    def __init__(self, args):
        self.tokenizer= AutoTokenizer.from_pretrained(modelpath+"roberta-base")
        self.layer_num = 12
        self.layer_width = 3072
        super().__init__(args)
    def getmodel(self):
        self.backbone = AutoModelForMaskedLM.from_pretrained(modelpath+"roberta-base")
    def processoutput(self, outputs):
        return outputs.logits[:,0,self.pos].squeeze(dim = 1)
    def intermediate(self,n):
        return self.backbone.roberta.encoder.layer[n].intermediate
    def embed(self, input_ids):
        return self.roberta.roberta.embeddings.word_embeddings(input_ids).detach()


