# Finding Paths for Explainable MOOC Recommendation: A Learner Perspective<!-- omit from toc -->

This repository contains code for the paper [Finding Paths for Explainable MOOC Recommendation: A Learner Perspective.](https://arxiv.org/abs/2312.10082)


## Table of Contents<!-- omit from toc -->

- [Datasets](#datasets)
- [Requirements](#requirements)
- [Install required packages](#install-required-packages)
- [How to run the code on Xuetang](#how-to-run-the-code-on-xuetang)
- [How to run the code on COCO](#how-to-run-the-code-on-coco)

## Datasets

### Xuetang

Download Xuetang from [http://moocdata.cn/data/MOOCCube](http://moocdata.cn/data/MOOCCube), extract the file and place the MOOCCube folder in data/mooc/

You should get two folders:

- data/mooc/MOOCCube/entities/
- data/mooc/MOOCCube/relations/

### COCO

Get the coco dataset by contacting the authors of [COCO: Semantic-Enriched Collection of Online Courses at Scale with Experimental Use Cases](https://link.springer.com/chapter/10.1007/978-3-319-77712-2_133) by email. Extract the file and place it in data/coco/

You sould get one folder:

- data/coco/coco/

Note: Because you might get a more recent version of the dataset, some of the characteristics (number of learners, courses, etc... ) might be different.

## Requirements

Python 3.10

If you intent to run the skill extractor on the coco datset, you will need to download en_core_web_lg:

```bash
python -m spacy download en_core_web_lg
```

## Install required packages

```bash
pip install -r requirements.txt
```

## How to run the code on Xuetang

### Process Xuetang's original files

```bash
python preprocess_mooc.py
```

After this process, all the files from MOOCCUbe have been standardized into the format needed by PGPR. The files are saved in the folder data/mooc/MOOCCube/processed_files.

We used the same file format as in the original PGPR repoisitory: [https://github.com/orcax/PGPR](https://github.com/orcax/PGPR).

### Xuetang's Dataset and Knowledge Graph creation

```bash
python make_dataset.py --config config/mooc.json
```

After this process, the files containing the train, validation and test sets and the Knowledge Graph have been created in tmp/mooc.

### Train the Xuetang's Knowledge Graph Embeddings

```bash
python train_transe_model.py --config config/mooc.json
```

The KG embeddings are saved in tmp/mooc.

### Train the RL agent on Xuetang

```bash
python python train_agent.py --config config/mooc.json
```

The agent is saved in tmp/mooc.

### Evaluation on Xuetang

```bash
python test_agent.py --config config/mooc.json 
```

The results are saved in tmp/mooc.

## How to run the code on COCO

### Extract the skills from COCO's course descriptions

```bash
python extract_skills.py
```

After this process

### Process coco's original files

```bash
python preprocess_coco.py
```

After this process, all the files from coco have been standardized into the format needed by PGPR. The files are saved in the folder data/mooc/MOOCCube/processed_files.

We used the same file format as in the original PGPR repoisitory: [https://github.com/orcax/PGPR](https://github.com/orcax/PGPR).

### COCO's Dataset and Knowledge Graph creation

```bash
python make_dataset.py --config config/coco.json
```

After this process, the files containing the train, validation and test sets and the Knowledge Graph have been created in tmp/mooc.

### Train the COCO's Knowledge Graph Embeddings

```bash
python train_transe_model.py --config config/coco.json
```

The KG embeddings are saved in tmp/coco.

### Train the RL agent on COCO

```bash
python python train_agent.py --config config/coco.json
```

The agent is saved in tmp/coco.

### Evaluation on COCO

```bash
python test_agent.py --config config/coco.json 
```

The results are saved in tmp/coco.
