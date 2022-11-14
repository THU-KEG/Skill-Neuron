Source code and dataset for EMNLP 2022 paper "Finding Skill Neurons in Pre-trained Transformer-based Language Models".

## Data Preparation

The datasets used in our experiments can be downloaded [here](https://cloud.tsinghua.edu.cn/d/fd753ed7b9f94a099cef). Please unzip the dara to the path `data/raw`.

### Add New Datasets 
To add another customized dataset, one need to follow the steps.

- If the dataset is downloaded from huggingface's datasets
    - Modify the codes in `src/dataset.py` and add the name of the dataset to the dicts `datasetColumn`, `datasetRow` and `datasetLabel`.

- If the dataset is stored in `.csv` or `.json` formats
    - Modify the codes in `src/dataset.py` and define a new dataset class based on `CsvDataset` or `JsonDataset`. `DbpediaDataset` is provided as an example, one only need to define the customized function `getsnt` and `getlabel`. One should then modify the following dicts `datasetType`, `datasetRow` and `datasetLabel`.

### Convert Multi-class Tasks

We provide a script `src/binarize.py` to transfer multi-class classification tasks into binary subtasks. Currently we support labelsets {`negative`,`neutral`,`positive`} or {`contradict`, `negative`, `neutral`, `positive`}. A sample command to convert the MNLI task is:

```bash
python src/binarize.py --task_type mnli --data_path data/raw/mnli --data_path data/raw
```

The dataset will be split into two subdatasets.

- `mnli-pn`: a balanced dataset consisting only of `positive` and `negative` samples of MNLI.

- `mnli-neu`: a balanced dataset with binary labels marking whether the original sentence is `neutral`.

## Prompt Tuning

After data preparation, to locate the skill neurons, one should conduct prompt tuning using the code `src/train.py`. In the following instructions, we take the usage on the SST-2 task as the running sample. Its command is like:

```bash
python src/train.py --task_type sst2 --data_path data/raw/sst2 --save_to example --bz 128 --eval_every_step 2000 --model_type RobertaPrompt
```

After training, a directory named `example` will be created and it will contain trained prompt named as `best-backbone` and also its optimizer and scheduler states named as `optimzer` and `scheduler`, respectively. In case the training is interrupted, one can proceed from the previous best checkpoint using the following argument `--resume_from example -p`.

`src/train.py` contains the following arguments to support various other functions:

- `model_type`: we support
    - `RobertaPrompt`: prompt tuning with frozen `roberta-base` backbone
    - `RobertaBias`: BitFit with frozen `roberta-base` backbone,
    - `RobertaAdapter`: Adapter-based tuning with frozen `roberta-base` backbone
    - `RobertaPrunePrompt`: prompt tuning with `roberta-base` backbone pruned using skill neurons
- `prompt_size`: the length of prompt
- `p`: whether to load a pretrained model, store true
- `save_to`: path to save the model
- `resume_from`: path to load the trained models
- `task_type`: choose from the datasets defined in `dataset.py`
- `data_path`: path to the dataset
- `verb`: verbalizer for the task, left brank for default setups
- `device`: choose from "cuda:0" and "cpu"
- `random_seed`: the random seed
- `epoch`, `eval_every_step`, `batch`, `early_stop`: common training parameters


## Obtaining Predictivity

As described in Section 3 of the paper, we need to calculate the predictivities of every neurons to find the skill neurons. One can use the `src/probe.py` to obtain predictivities. A sample usage command is:

```bash
python src/probe.py --resume_from example --data_path data/raw/sst2 --save_to example --probe_type acc --bz 16 -p --task_type sst2 --model_type RobertaPrompt
```

After running, a directory named `example/sst2` will be created and three files will be saved:
- `train_avg`: average activation for every neuron, with shape [`#layers`, `prompt length`, `FFN width`], corresponding to $a_{bsl}$ in the paper
- `dev_perf`: predictivity of every neuron using `train_avg` as the threshold, with shape [`#layers`, `prompt length`, `FFN width`]
- `test_perf`: predictivity of every neuron using `train_avg` as the threshold, with shape [`#layers`, `prompt length`, `FFN width`]

`src/probe.py` contains the following arguments to support various other functions:

- `probe_type`, to choose other analytical indexs, we support:
    - `acc`: to obtain the predictivities of all the neurons, will generate the three files as explained (only support prompt-based model)
    - `acc_mean` and `acc_max`: to obtain the prediction accuracies of neurons with mean pooling and max pooling, used for experiments in Section 4.1 of the paper
        - the outputs with shape [`#layers`, `FFN width`]
    - `pos`: to obtain the activations of a specific neuron, needing to specify (`layers`, `index`)
        - will create three files, train_mean_avg, dev_mean_perf, test_mean_perf
        - each with shape [`#layers`, `FFN width`]
    - `prompt_act`: to obtain the activations on the first special `[MASK]` token, paired with dataset `empty`, with output shape [`#sample`, `#layer`, `FFN width`], used for experiments in Section 6.2 of the paper
    - `speed`: to obtain the test performance and running time, will generate two files `inference_perf` and `inference_time`
- `model_type`, `prompt_size`, `p`, `resume_from`, `save_to`, `task_type`, `data_path`, `verb`, `device`, `random_seed`, `batch_size`: as introduced in the last section


## Find Skill Neurons

With the predictivities, one can find skill neurons with `src/skillneuron.py`. A sample command is:

```bash
python src/skillneuron.py --save_to example -a example/sst2/dev_perf
```

After the command, a directory named as `example/info/sst2` will be created, consisting files `0.0` to `0.19`, each is a boolean tensor with shape [`#layer`, `FFN width`] indicating the locations of skill neurons. For example, in the tensor of file `0.05`, the `1` elements indicating the neurons here are top 5% skill neurons.

`src/skillneuron.py` contains the following arguments.

- `type`: choose from `mean` or `min`, meaning to aggregate the accuracy of different prompt tokens with average pooling or min pooling
- `a`/`acc_path`: path to dumped predictivity files, seperated by comma
- `aa/acc_aux_path`: path to dumped predictivity files of more decomposed subtasks for multi-class tasks
- `task_type`, `save_to`: the same as introduced above

## Perturbing Skill Neurons

To do the neuron perturbation experiments for skill neurons, one may use the code `src/mask.py`. A sample command is like:

```
python src/mask.py --info_path example/info/sst2/0.1 --resume_from example --data_path data/raw/sst2 --save_to example --bz 16 -p  --cmp
```

`src/mask.py` has the following arguments:

- `cmp`: wheter do control experiment perturbing the same number of random neurons, store true
- `info_path`: path to the skill neuron location files generated in the last step
- `type`: choose perturbation method, we support:
    - `gaussian`, add gaussian noises with standard error of `alpha`(recommended)
    - `zero`, replace the activations by zero
    - `mean`, replace the activations by customized thresholds
- `alpha`: standard error for the gaussian noise if set `type` as `gaussian`
- `ths_path`: to specify the customized threshold file if set `type` as `mean`. The threshold should be a tensor of shape [`#layer`, `FFN width`]
- `low_layer` and `high layer`: to only perturb layers in [`lower_layer`, `high_layer`]
- `task_type`, `data_path`, `verb`, `model_type`, `prompt_size`, `p`, `save_to`, `resume_from`, `device`, `random_seed`, `batch`: as introduced above


## Pruning Transformers with Skill Neurons

As shown in Section 6.1 of the paper, we can use the skill neurons to guide network pruning on Transformers.

To do this, one should first generate the prune model structure by running the following commands:

```bash
mkdir prune_structure
python src/helper/getprunestructure.py
```

To generate a pruned Transformer model with the skill neurons, a sample command is:

```bash
python src/prune.py --info_path example/info/sst2/0.02 --ths_path example/sst2/train_avg --save_to example/prune --resume_from example/best-backbone 
```

This will create a pruned model at `example/prune/best-backbone`. One can further evaluate the inference speed of this model by:

```bash
python src/probe.py -p --resume_from example/prune --model_type RobertaPrunePrompt --save_to example/prune --probe_type speed --data_path data/raw/sst2 
```

One can also further train the pruned model by:

```bash
python src/train.py -p --resume_from example/prune --model_type RobertaPrunePrompt --task_type sst2 --data_path  data/raw/sst2 --save_to example/prune --bz 128 --eval_every_step 2000
```

`src/prune.py` support only pruning `RobertaPrompt` model to `RobertaPrunePrompt` model. It needs the following arguments:

- `info_path`: path to the skill neuron location files generated in the last step
- `resume_from`: a path to a trained model as the input of pruning
- `ths_path`: path to a file for the fixed activations of non-skill neurons. It shall be a tensor of shape [`#layers`, `FFN width`] or [`#layers`, `prompt length`, `FFN width`]
- `save_to`, `random_seed`: as introduced above

## Citation
TBD