# Finding Paths for Explainable MOOC Recommendation: A Learner Perspective<!-- omit from toc -->

This repository contains code for the paper [Finding Paths for Explainable MOOC Recommendation: A Learner Perspective.](https://arxiv.org/abs/2312.10082)

![Alt text](pipeline.jpg)

## Table of Contents<!-- omit from toc -->

- [Datasets](#datasets)
- [Installation](#installation)
- [How to run UPGPR on Xuetang](#how-to-run-upgpr-on-xuetang)
- [How to run UPGPR on COCO](#how-to-run-upgpr-on-coco)
- [Additional information about UPGPR config files](#additional-information-about-upgpr-config-files)
- [How to run the baselines](#how-to-run-the-baselines)
- [Using a custom dataset](#using-a-custom-dataset)
- [Citation](#citation)

## Datasets

<details>

<summary>Datasets</summary>

### Xuetang

Download Xuetang from [http://moocdata.cn/data/MOOCCube](http://moocdata.cn/data/MOOCCube), extract the file and place the MOOCCube folder in data/mooc/

We assume that you will have at least the following two folders:

- data/mooc/MOOCCube/entities/
- data/mooc/MOOCCube/relations/

### COCO

Get the coco dataset by contacting the authors of [COCO: Semantic-Enriched Collection of Online Courses at Scale with Experimental Use Cases](https://link.springer.com/chapter/10.1007/978-3-319-77712-2_133) by email. Extract the file and place it in data/coco/

You sould get one folder:

- data/coco/coco/

Note: Because you might get a more recent version of the dataset, some of the characteristics (number of learners, courses, etc... ) might be different.

</details>

## Installation

<details>

<summary>Installation</summary>

### Requirements

Python 3.10 is required.

We recommend using a conda environment, but feel free to use wahthever you are the most confortable with:

```bash
conda create -n upgpr python=3.10
conda activate upgpr
```

### Install required packages

```bash
pip install -r requirements.txt
```

If you intent to run the skill extractor on the coco datset, you will need to download en_core_web_lg:

```bash
python -m spacy download en_core_web_lg
```

</details>

## How to run UPGPR on Xuetang

<details>

<summary>UPGPR on Xuetang</summary>

### Process Xuetang's original files

```bash
python src/UPGPR/preprocess_mooc.py
```

After this process, all the files from MOOCCUbe have been standardized into the format needed by PGPR. The files are saved in the folder data/mooc/MOOCCube/processed_files.

We used the same file format as in the original PGPR repoisitory: [https://github.com/orcax/PGPR](https://github.com/orcax/PGPR).

### Xuetang's Dataset and Knowledge Graph creation

```bash
python src/UPGPR/make_dataset.py --config config/UPGPR/mooc.json
```

After this process, the files containing the train, validation and test sets and the Knowledge Graph have been created in tmp/mooc.

### Train the Xuetang's Knowledge Graph Embeddings

```bash
python src/UPGPR/train_transe_model.py --config config/UPGPR/mooc.json
```

The KG embeddings are saved in tmp/mooc.

### Train the RL agent on Xuetang

```bash
python src/UPGPR/train_agent.py --config config/UPGPR/mooc.json
```

The agent is saved in tmp/mooc.

### Evaluation on Xuetang

```bash
python src/UPGPR/test_agent.py --config config/UPGPR/mooc.json 
```

The results are saved in tmp/mooc.

</details>

## How to run UPGPR on COCO

<details>

<summary>UPGPR on COCO</summary>

### Extract the skills from COCO's course descriptions

```bash
python src/UPGPR/extract_skills.py
```

After this process, the files course_skill.csv and learner_skill.csv have been created in data/coco/coco

### Process coco's original files

```bash
python src/UPGPR/preprocess_coco.py 
```

After this process, all the files from coco have been standardized into the format needed by PGPR. The files are saved in the folder data/mooc/MOOCCube/processed_files.

We used the same file format as in the original PGPR repoisitory: [https://github.com/orcax/PGPR](https://github.com/orcax/PGPR).

### COCO's Dataset and Knowledge Graph creation

```bash
python src/UPGPR/make_dataset.py --config config/UPGPR/coco.json
```

After this process, the files containing the train, validation and test sets and the Knowledge Graph have been created in tmp/mooc.

### Train the COCO's Knowledge Graph Embeddings

```bash
python src/UPGPR/train_transe_model.py --config config/UPGPR/coco.json
```

The KG embeddings are saved in tmp/coco.

### Train the RL agent on COCO

```bash
python src/UPGPR/train_agent.py --config config/UPGPR/coco.json
```

The agent is saved in tmp/coco.

### Evaluation on COCO

```bash
python src/UPGPR/test_agent.py --config config/UPGPR/coco.json 
```

The results are saved in tmp/coco.

</details>

## Additional information about UPGPR config files

<details>

<summary>Config files</summary>

### Run original PGPR

To run the original PGPR, change the config files in config/UPGPR as follows:

- Set the "reward" attribute in "TRAIN_AGENT" and "TEST_AGENT" to "cosine".
- Set the "use_pattern" attribute in "TRAIN_AGENT" and "TEST_AGENT" to "true".
- Set the "max_path_len" attribute in "TRAIN_AGENT" and "TEST_AGENT" to 3.

To run UPGPR, change the config files in config/UPGPR as follows:

- Set the "reward" attribute in "TRAIN_AGENT" and "TEST_AGENT" to "binary_train".
- Set the "use_pattern" attribute in "TRAIN_AGENT" and "TEST_AGENT" to "false".
- Set the "max_path_len" attribute in "TRAIN_AGENT" and "TEST_AGENT" to an integer > 2
- If "max_path_len" has a value different than 3, change the value of the "topk" attribute in "TEST_AGENT" to list of the same length as "max_path_len".

</details>

## How to run the baselines

<details>

<summary>Baselines</summary>

### Process the files for Recbole

Process the Xuetang files for RecBole (requires data/mooc/MOOCCube/processed_files)

```bash
python src/baselines/format_moocube.py
```

After this process, all the files from coco have been standardized into the format needed by RecBole. The files are saved in the folder data/mooc/recbolemoocube.

We follow the same process for coco:

```bash
python src/baselines/format_coco.py
```

The files are saved in the folder data/coco/recbolecoco.

### Run the baselines

To run the baselines, choose a config file in config/baselines and run the following:

```bash
python src/baselines/baseline.py --config config/baselines/coco_Pop.yaml
```

This example runs the Pop baseline on the coco dataset.

You can ignore the warning "command line args [--config config/baselines/coco_Pop.yaml] will not be used in RecBole". The argument is used properly.

</details>

## Using a custom dataset

<details>

<summary>Custom</summary>

### Files structure

In the folder [example](data/example/), we have provided a minimalistic example of a synthetic dataset to help understadning the format of the files required by UPGPR. This dataset is to understand the format of the files only and is too small to be used to test the code.

Below, you will find a detailed description of the files:

- **Enrolments file**. You must have a file named "enrolments.txt" containing the enrollments of each student. The structure is the following: each line contain one enrolment with the student id and the course id separated by a space. id must be integers. An example is provided here: [enrolments.txt](data/example/enrolments.txt)
- **Entities files.** For each entity in your knowledge graph (student, course, teacher, school, etc...) you must have a file named "entity_name.txt" and each line contains the name of the entity associated to the id line number - 1. For example in the file [courses.txt](data/example/courses.txt), on line 1 we have the course "Math", meaning that it's id is 0.
- **Relations files.** For each relation in your knowledge grahp (course_teacher, course_school, eacher_school, etc...) you must have a file named "sourceentity_targetentity.txt" where each line corresponds to the source entity id and contains all the tagets entities id that are related to the source entity. For example in the file [course_teachers.txt](data/example/course_teachers.txt), on line 3 we have the course "2 3", meaning that Charlie and Dave are teaching History.
  
### Config file

You also need to modify the config file to be suited to your custom dataset. You will need to modify the content of "KG_ARGS" in the config file to specify the entities,relations, and files names that contain these realtions. You can have a look at the file [example.json](config/UPGPR/example.json) to have an example of the content of "KG_ARGS" for our example dataset.

</details>

## Citation

```tex
@article{frej2023finding,
  title={Finding Paths for Explainable MOOC Recommendation: A Learner Perspective},
  author={Frej, Jibril and Shah, Neel and Kne{\v{z}}evi{\'c}, Marta and Nazaretsky, Tanya and K{\"a}ser, Tanja},
  journal={arXiv preprint arXiv:2312.10082},
  year={2023}
}
```
