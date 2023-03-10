### This code is used to generate dataset.
### The dataset will be split into train, validation and test.
from cgitb import text
import csv
import json
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_from_disk
from torch.utils.data import random_split as split


class RawTextDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path 
        self.text = self.read(path)
    def getsnt(self,text):
        pass
    def getlabel(self,text):
        pass
    def read(self,path):
        pass
    def __getitem__(self, index):
        text = self.text[index]
        snt = self.getsnt(text)
        label = self.getlabel(text)
        return {'sentence':snt,'label':label}
    def __len__(self):
        return len(self.text)



class CsvDataset(RawTextDataset):
    def __init__(self, path):
        super().__init__(path)
    def getsnt(self,text):
        pass
    def getlabel(self,text):
        pass
    def read(self,path):
        with open(path) as f:
            reader = csv.reader(f)
            line = []
            sent = False
            for row in tqdm(reader):
                if(sent):
                    line.append(row)
                sent = True
            random.shuffle(line)
        return line


class JsonDataset(RawTextDataset):
    def __init__(self, path):
        super().__init__(path)
    def getsnt(self,text):
        pass
    def getlabel(self,text):
        pass
    def read(self,path):
        with open(path) as f:
            line = json.load(path)
            random.shuffle(line)
        return line
 


class DbpediaDataset(CsvDataset):
    def __init__(self, path):
        super().__init__(path)
    def getsnt(self,text):
        return text[2]
    def getlabel(self,text):
        return int(text[0]) 

class DbpediaSDataset(CsvDataset):
    def __init__(self, path):
        super().__init__(path)
    def read(self,path):
        with open(path) as f:
            reader = csv.reader(f)
            line = []
            sent = False
            for row in tqdm(reader):
                if(sent and (self.getlabel(row) in [0,1])):
                    line.append(row)
                sent = True
            # from IPython import embed;embed()
            random.shuffle(line)
        return line
    def getlabel(self,text):
        origin = int(text[0]) 
        if(origin in [0,4]):
            return [0,4].index(origin)
        else:
            return -1
    def getsnt(self,text):
        return text[2]

class EmptyDataset(RawTextDataset):
    def __init__(self,path):
        super().__init__(path)
    def getsnt(self, text):
        return ""
    def getlabel(self, text):
        return 0
    def read(self, path):
        return [0,0,0]




datasetColumn = {
    "ag_news":["train","test"],
    "ag_news_s":["train","test"],
    "cola":["train","validation","test"],
    "deontology":["train","validation","test"],
    "ethos":["train"],
    "hate":["train"],
    "hate-pn":["train"],
    "hate-neu":["train"],
    "imdb":["train","test"],
    "justice":["train","validation","test"],
    "tweet_eval":["train","validation","test"],
    "tweet_eval-pn":["train","validation","test"],
    "tweet_eval-neu":["train","validation","test"],
    "mnli":["train","validation_matched"],
    "mnli-pn":["train","validation_matched"],
    "mnli-neu":["train","validation_matched"],
    "qnli":["train","validation"],
    "qqp":["train","validation"],
    "sst2":["train","validation"],
    "tweets_hate":["train"],
    "virtue":["train","validation","test"],
}

datasetRow = {
    "ag_news":["text","label"],
    "ag_news_s":["text","label"],
    "cola":["sentence","label"],
    "dbpedia":["sentence","label"],
    "dbpedia_s":["sentence","label"],
    "deontology":["text","label"],
    "ethos":["text","label"],
    "empty":["sentence","label"],
    "hate":["text","label"],
    "hate-pn":["text","label"],
    "hate-neu":["text","label"],
    "imdb":["text","label"],
    "justice":["text","label"],
    "mnli":["premise","hypothesis","label"],
    "mnli-pn":["premise","hypothesis","label"],
    "mnli-neu":["premise","hypothesis","label"],
    "tweet_eval":["text","label"],
    "tweet_eval-pn":["text","label"],
    "tweet_eval-neu":["text","label"],
    "qnli":["question","sentence","label"],
    "qqp":["question1","question2","label"],
    "sst2":["sentence","label"],
    "tweets_hate":["tweet","label"],
    "virtue":["sentence1","sentence2","label"]
}

datasetLabel = {
    "ag_news":4,
    "ag_news_s":2,
    "cola":2,
    "dbpedia":14,
    "dbpedia_s":2,
    "deontology":2,
    "ethos":2,
    "empty":2,
    "hate":4,
    "hate-pn":2,
    "hate-neu":2,
    "imdb":2,
    "justice":2,
    "mnli":3,
    "mnli-pn":2,
    "mnli-neu":2,
    "tweet_eval":3,
    "tweet_eval-pn":2,
    "tweet_eval-neu":2,
    "qnli":2,
    "qqp":2,
    "sst2":2,
    "qqp":2,
    "tweets_hate":2,
    "virtue":2
}

datasetType = {
    "dpbedia": DbpediaDataset,
    "dbpedia_s":DbpediaSDataset,
    "empty":EmptyDataset
}

def getDataset(taskName,path):
    print(f'### Task Name = {taskName} ###')
    if(datasetType.get(taskName,"hugging") != "hugging"):
        train = datasetType[taskName](path+"/train.csv")
        train, valid = split(dataset = train, lengths = [int(0.8*len(train)),len(train)-int(0.8*len(train))])
        test = datasetType[taskName](path+"/test.csv")
    else:
        text = load_from_disk(path)
        columns = datasetColumn.get(taskName,[])
        if(len(columns) == 3):
            train, valid, test = text[columns[0]],text[columns[1]],text[columns[2]]
        elif(len(columns) == 2):
            train, test = text[columns[0]],text[columns[1]]
            train, valid = split(dataset = train, lengths = [int(0.8*len(train)),len(train)-int(0.8*len(train))])
        elif(len(columns) == 1):
            train, valid, test = split(dataset = text, lengths = [int(0.8*len(text)),(len(text) - int(0.8*len(text)))//2,
                (len(text) - int(0.8*len(text)))//2])
        else:
            print(taskName+" not implemented!")
            raise NotImplementedError
        print("Finish Processing "+taskName)
    return train, valid, test







