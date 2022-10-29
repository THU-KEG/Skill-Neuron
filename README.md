# README of Skill Neuron V1.0

This is the codebase for paper *Finding Skill Neurons in Pre-trained Transformers via Prompt Tuning*. Using this repo, you can find skill neuron, analyze their causal effect on the model performance and prune model using the skill neuron you found. Let's motivate by an example.


## Data Preparation

The data can be downloaded using the link https://cloud.tsinghua.edu.cn/d/fd753ed7b9f94a099cef/

One should generate the prune structure by first run the following scripts

> mkdir prune_structure

> python3 pyscripts/helper/getprunestructure.py

For the following, suppose the data is stored at path *data/raw*. One can check the list of supported dataset under the directory.

To add another customized dataset, one need to follow the steps.

* If the dataset is downloaded via huggingface

    *  Modify the scripts in *pyscripts/dataset.py* and add the name of the dataset to the following dicts *datasetColumn, datasetRow, datasetLabel*.

* If the dataset is stored in csv or json forms

    * Modify the scripts in *pyscripts/dataset.py* and define a new dataset class based on *CsvDataset* or *JsonDataset*. In the code, *DbpediaDataset* is provided as an example, one only need to define the customized function *getsnt* and *getlabel*. One should then modify the following dicts *datasetType, datasetRow, datasetLabel*.

We also provide a script *pyscripts/binarize.py* to transfer multiclass classification into binary subtask. Currently we only support label ['negative','neutral','positive'] or ['contradict','negative','neutral','positive']. A sample usage would be like

> python3 pyscripts/binarize.py --task_type mnli --data_path data/raw/mnli --data_path data/raw

The dataset will be split into two subdataset.

- mnli-pn: a balanced dataset consisting only of positive and negative sample of mnli

- mnli-neu: a balanced dataset with binary label marking whether the original sentence is marked neutral.

## Model Training

After the data is prepared, one should proceed to train the model using the script *pyscripts/train.py*. A sample usage would be like

> python3 pyscripts/train.py --task_type sst2 --data_path saber/data/raw/sst2 --save_to example --bz 128 --eval_every_step 2000 --model_type RobertaPrompt

After training, a directory named *example* will be created and it will contain trained prompt named as *best-backbone* and also optimizer and scheduler named *optimzer* and *scheduler*. In case the training is interrupted, one can proceed from the previous best checkpoint using the following args *--resume_from example -p*.

*pyscripts/train.py* contains the following args.

- model_type, we currently support
    - RobertaPrompt: prompt model with frozen roberta-base backbone
    - RobertaBias: Bias Tuning model with roberta-base backbone
    - RobertaAdapter: Bias Tuning model with roberta-base backbone
    - RobertaPrunePrompt: prompt model with pruned roberta-base backbone using skill neuron
    - One can modify *pyscripts/modelzoo.py* and *getmodel() in pyscripts/trainer.py* to add a new model
- prompt_size, the length of prompt, prompt is inserted at beginning for MLM model (begin with a frozen [MASK]) and at the end for GPT model.
- p, whether to load a pretrained model, store true
- save_to, directory to save the model
- resume_from, directory to load the pretrained model
- task_type, choose from the dataset in dataset.py
- data_path, directory to the dataset
- verb, verbalizer for the task, when left brank, will use default
- device, choose from "cuda:0" and "cpu"
- random_seed, random seed
- epoch, eval_every_step, batch, early_stop, common training parameters


## Probing

Our work is based on probing the accuracy of neurons in feed forward layer of the transformer backbone. One can use the *pyscipts/probe.py* to perform probing. As an example, after we trained the sst2 task in previous section. We can use the following command to probe the skill neuron based on the prompt we train.

>  python3 pyscripts/probe.py --resume_from example --data_path data/raw/sst2 --save_to example --probe_type acc --bz 16 -p --task_type sst2 --model_type RobertaPrompt

After probing, a directory named *example/sst2* will be created and three files will be saved.
* train_avg: average activation of the feed forward neuron, with shape [#layers, prompt size, FFN width]
* dev_perf: precision of neuron using train_avg as threshold, with shape [#layers, prompt size, FFN width]
* test_perf: precision of neuron using train_avg as threshold, with shape [#layers, prompt size, FFN width]

*pyscripts/probe.py* contains the following args.

- model_type, prompt_size, p, resume_from, save_to, task_tyoe, data_path, verb, device, random_seed, batch_size, as previous section about training
- probe_type, we can choose from

    - pos: to probe the activation of a specific neuron
        - using the -pos args to input (layers, index)
    - acc: to probe the prediction precision of all the neurons, will generate the three files as explained (only support prompt based model)
    - acc_mean: to probe the prediction precision of all the neurons using average fusing
        - will create three files, train_mean_avg, dev_mean_perf, test_mean_perf
        - each with shape [#layers, FFN width]
    - acc_max: to probe the prediction precision of all the neurons using max fusing
        - will create three files, train_max_avg, dev_max_perf, test_max_perf
        - each with shape [#layers, FFN width]
    - prompt_act: to probe the activation of the prompt on the first token [[MASK] for MLM prompt based model], will contain shape [#sample, #layer, FFN width], should be paired with dataset *empty*
    - speed: probe the speed and test performance of model on data, will generate two files inference_perf and inference_time, storing the performance and time

## Generate Skill Neuron

After the probing in the previous section, one can use the accuracy found to generate skill neuron via *pyscripts/skillneuron.py*. A sample code would be like 

> python3 pyscripts/skillneuron.py --save_to example -a saber/example/sst2/dev_perf

*pyscripts/skillneuron.py* has the following args.

- task_type, save_to: same as previous section
- type, choose from mean or min, meaning to aggregate the accuracy of different model using average fusing or minimum fusing
- a/acc_path, path of probed acc seperated by comma
- aa/acc_aux_path, path of probed acc for the auxiliary task for multilabel task, see the paper for detailed.

After the command, a directory named as *example/info/sst2* will be created, consisting file *0.0* to *0.19*, each is a boolean tensor with shape *[#layer, layer width]* with 1 indicating high accuracy and hence better skill.

## Masking using the skill neuron

One may examine the causal effect of the skill neuron by masking the neuron during inference using scripts *pyscripts/mask.py*. A sample command would be like 

> python3 pyscripts/mask.py --info_path example/info/sst2/0.1 --resume_from example --data_path data/raw/sst2 --save_to example --bz 16 -p  --cmp

*pyscripts/mask.py* has the following args

- cmp, wheter do control experiment and mask the same number of random neurons
- info_path, path to the info files, which should be a boolean tensor with shape *[#layer, layer width]* with 1 indicating skill neuron
- type, method of masking, we now support the following
    - gaussian, add gaussian noise with standard error of args.alpha (recommended)
    - zero, replace the activation by zero
    - mean, replace the activation by some customized threshold
- ths_path, when choosing type mean, used to index the custom threshold file, should have shape *[#layer, layer width]*
- low_layer, high layer, will only mask layer in [lower_layer, high_layer)
- alpha, standard error for the gaussian noise
- task_type, data_path, verb, model_type, prompt_size, p, save_to, resume_from, device, random_seed, batch, as previous session


## Pruning using skill neuron

One may also prune the model using the skill neuron by scripts modelzoo.py/skillneuron.py. A sample command would be like

> python3 pyscripts/prune.py --info_path example/info/sst2/0.02 --ths_path example/sst2/train_avg --save_to example/prune --resume_from example/best-backbone 

This will create a prune model at example/prune/best-backbone, one can further evaluate the inference speed of this model by
> python3 pyscripts/probe.py -p --resume_from example/prune --model_type RobertaPrunePrompt --save_to example/prune --probe_type speed --data_path data/raw/sst2 

or further train the model by
> python3 pyscripts/train.py -p --resume_from example/prune --model_type RobertaPrunePrompt --task_type sst2 --data_path saber/data/raw/sst2 --save_to example/prune --bz 128 --eval_every_step 2000

*pyscripts/prune.py* now support only the pruning of RobertaPrompt model to RobertaPrunePrompt model. It contains the following args.

- save_to, random_seed, as previous session
- info_path, take a path to the skill neuron file as previous mentioned, must contains int(0.02*3072) skill neuron on every layer
- resume_from, take as input a path to a trained model
- ths_path, take as input as a file used as fixed activation of non-skill neuron, can have shape [#layers, FFN width] or [#layers, prompt size, FFN width] 


