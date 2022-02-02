#  Improving and Evaluating Contrastive Learning in Abstractive Summarization

  
  

##  Overview

This notebook is the runner of our EECS 595 Natural Language Processing (NLP) project - Improving and Evaluating Contrastive Learning in Abstractive Summarization with a Better Baseline and More Variable Datasets

Our work is built highly based on SimCLS ([paper](https://arxiv.org/pdf/2106.01890.pdf) and [official implementation](https://github.com/yixinL7/SimCLS)) and SimCSE ([paper](https://arxiv.org/pdf/2104.08821.pdf) and [official implementation](https://github.com/princeton-nlp/SimCSE))

As shown below, SimCLS framework consists of for two stages: Candidate Generation and Reference-free evaluation, where Doc, S, Ref} represent the document, generated summary and reference respectively.

  



  
  
  
  

##  1. How to Install

  

###  Requirements

-  `python3.8.7`

-  `virtualenv venv && source venv/bin/activate`

-  `pip3 install -r requirements.txt`

- Download [compare-mt](https://github.com/neulab/compare-mt) to `./`

- `cd compare_mt/ && python setup.py install`

  

###  Description of Codes

-  `main.py` -> training and evaluation procedure of original SimCLS

-  `main_SimCSE.py` -> training and evaluation procedure of our works

-  `model.py` -> models of original SimCLS

-  `model_SimCSE.py` -> models of our works

-  `data_utils.py` -> dataloader

-  `utils.py` -> utility functions

-  `preprocess.py` -> data preprocessing

-  `get_data.py` -> get subset of the dataset with required amount

-  `load_dataset.py` -> generate candidate summaries

  

###  Workspace

Following directories should be created for our experiments.

-  `./cache` -> storing model checkpoints

-  `./result` -> storing evaluation results

-  `./output` -> storing outputs of the model and the references

  

##  2. Preprocessing

We use the following datasets for our experiments.

  

- CNN/DailyMail -> https://github.com/abisee/cnn-dailymail

- XSum -> https://github.com/EdinburghNLP/XSum

- Webis-TLDR-17 Corpus (Reddit) -> https://www.tensorflow.org/datasets/catalog/reddit

- Gigaword -> https://www.tensorflow.org/datasets/catalog/gigaword

For acquiring a small subset of dataset, please run:
```
python get_data.py
```
For generating candidates, please run (make sure you have the `.obj` file and have created the folder of the dataset name)
```
python load_dataset.py --split test --data [path of pkl files] --max_length 50 --min_length 5
```
And you would have the following files in your `data` path(using test split as an example):
-  `test.source`

-  `test.source.tokenized`

-  `test.target`

-  `test.target.tokenized`

-  `test.out`

-  `test.out.tokenized`

Make sure you have the above files before you do preprocessing.

For data preprocessing, please run

```

python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries]

```


Each line of these files should contain a sample. In particular, you should put the candidate summaries for one data sample at neighboring lines in `test.out` and `test.out.tokenized`.

  

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

  

We have provided an example file in `./example`.

  

##  3. How to Run

  

###  Preprocessed Data

You can download the preprocessed data for our experiments on [CNNDM](https://drive.google.com/file/d/1WRvDBWfmC5W_32wNRrNa6lEP75Vx5cut/view?usp=sharing) and [XSum](https://drive.google.com/file/d/1nKx6RT4zNxO4hFy8y3dPbYV-GBu1Si-u/view?usp=sharing) (provided by the SimCLS).

  

After donwloading, you should unzip the zip files to `./`.

  

###  Hyper-parameter Setting

You may specify the hyper-parameters in `main.py` and `main_SimCSE.py`.

  

To reproduce our results, you could use the original configuration in the file, except that you should make sure that on CNNDM, Gigaword, and Reddit

`args.max_len=120`, and on XSum `args.max_len = 80`.

  
  
substitute `main_SimCSE.py` with `main.py` if you want to review SimCLS's works

###  Train

```

python main_SimCSE.py --cuda --gpuid [list of gpuid] -l

```

###  Fine-tune

```

python main_SimCSE.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]

```

model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

###  Evaluate

```

python main_SimCSE.py --cuda --gpuid [single gpu] -e --model_pt [model path]

```

model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`). If you do not specify the model, it would evaluate the untrained version of our model.

  

##  4. Results

Our model outputs on these datasets can be found in `./output`.

We have also provided the finetuned checkpoints on [CNNDM](https://drive.google.com/file/d/1CSFeZUUVFF4ComY6LgYwBpQJtqMgGllI/view?usp=sharing) and [XSum](https://drive.google.com/file/d/1yx9KhDY0CY8bLdYnQ9XhvfMwxoJ4Fz6N/view?usp=sharing) (by the original well trained SimCLS).

### Citations

SimCLS
```bibtex
@inproceedings{liu-liu-2021-simcls,
    title = "{S}im{CLS}: A Simple Framework for Contrastive Learning of Abstractive Summarization",
    author = "Liu, Yixin  and
      Liu, Pengfei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.135",
    doi = "10.18653/v1/2021.acl-short.135",
    pages = "1065--1072",
}
```

SimCSE
```
@inproceedings{gao-etal-2021-simcse,
    title = "{S}im{CSE}: Simple Contrastive Learning of Sentence Embeddings",
    author = "Gao, Tianyu  and
      Yao, Xingcheng  and
      Chen, Danqi",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.552",
    pages = "6894--6910",
    abstract = "This paper presents SimCSE, a simple contrastive learning framework that greatly advances the state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation and removing it leads to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language inference datasets into our contrastive learning framework, by using {``}entailment{''} pairs as positives and {``}contradiction{''} pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT base achieve an average of 76.3{\%} and 81.6{\%} Spearman{'}s correlation respectively, a 4.2{\%} and 2.2{\%} improvement compared to previous best results. We also show{---}both theoretically and empirically{---}that contrastive learning objective regularizes pre-trained embeddings{'} anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.",
}
```
