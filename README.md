# PMLDL-Assignment-2

## Student information
* Name: Gleb
* Surname: Kirillov
* Innopolis email: g.kirillov@innopolis.university
* Group: BS21-RO

## Installation
### Clone repository:
```
git clone https://github.com/glebuben/PMLDL-Assignment-2.git
```
### Create and activate virtual environment
```console
cd PMLDL-Assignment-2
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```console
pip install torch pandas
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

## How to use

Run the recommendation script with the required arguments:
```console
python ./benchmark/evaluate.py [output_type] [data_dir_path] [model_dir_path] --K --user_id
```
**output_type** - type for the output (evaluate or recommend). Evaluate - prints Precision and Recall.
Recommend - prints recommended movies ids

**data_dir_path** - path to benchmark data directory

**model_dir_path** - path to models directory

**--K** - number of movies to recommend

**--user_id** - specify when use `single` type



For example
```console
python ./PMLDL-Assignment-2/benchmark/evaluate.py evaluate PMLDL-Assignment-2/benchmark/data PMLDL-Assignment-2/models --K 20
```

## Assignment description
This assignment is to create a recommender system of movies for users:
* Your system should suggest some movies to the user based on user's gemographic information(age, gender, occupation, zip code) and favorite movies (list of movie ids).
* Solve this task using a machine learning model. You may consider only one model: it will be enough.
* Create a benchmark that would evaluate the quality of recommendations of your model. Look for commonly used metrics to evaluate a recommender system and use at least one metric.
* Make a single report describing data exploration, solution implementation, training process, and evaluation on the benchmark.
* Explicitly state the benchmark scores of your systems.

## Repository structure

### [📂 data](./data)
* #### [📂 external](./data/external) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; # Data from third party sources
* #### [📂 interim](./data/interim) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # Intermediate data that has been transformed.
* #### [📂 raw](./data/raw) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; # The original, immutable data

### [📂 models](./models) &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; # Trained and serialized models, final checkpoints

### [📂 notebooks](./notebooks) &emsp; &emsp; &emsp; &emsp; &nbsp;  # Jupyter notebooks. Naming convention is a number (for ordering)

### [📂 references](./references) &emsp; &emsp; &emsp; &emsp; &nbsp; # Data dictionaries, manuals, and all other explanatory materials.

### [📂 reports](./reports)
* #### [📂 figures](./reports/figures) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; # Generated graphics and figures to be used in reporting
* #### [📜 final_report.pdf](./reports/final_report.pdf) &emsp; &emsp; &emsp; &emsp; &nbsp;  # Report containing data exploration, solution exploration, training process, and evaluation

### [📂 benchmark](./benchmark)
* #### [📂 data](./benchmark/data) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; # dataset used for evaluation 
* #### [📜 evaluate.py](./benchmark/evaluate.py) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # script that performs evaluation of the given model


