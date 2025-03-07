# Ontology-based box embeddings and knowledge graphs for predicting phenotypic traits in *Saccharomyces cerevisiae*

Code and material for the paper Ontology-based box embeddings and knowledge graphs for predicting phenotypic traits in *Saccharomyces cerevisiae*

Unprocessed data is found in `data`, KGs and class hierarchies found in `graphs`, datasets used to train models found in `datasets`, explanations for predictions in `explanations`, data and processing script related to the biological experiment in `bio_experiment`, and the code is in `code`. `code/data_prep` contains code to download, and process data, generate the graphs and the datasets etc. `code/embeddings` contains code for the class embeddings, `code/prediction_models` contains code to train, evaluate, and find explanations for predictions. `code/explanations` contain code to clean and analyse explanations.

To run the code install the packages (with Python 3.10.15) in the requirements files, `requirements.txt` and `requirements_torch.txt`. The second file contain torch requirements and can be installed after `requirements.txt`.

Some large files, not stored on this git are needed as well. Begin by downloading the content in this Zenodo: https://zenodo.org/records/14989969?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUyNmM3YmQzLTA1YzMtNDI0MC05ZGExLTQwYzRiMjcwYWNhMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjOTk5YjY2YWE5Nzk4N2I2MjZiNDQ2ZTQzYmJjMDBmMSJ9.Dj28lYFKv_lCblA9B45RO_UkW2elBD94QCig00OJbMBy-oXJF9DwZrKja3I-9FDIWu2rXZt7u2cFDHGAoi1lKA

Then unzip the file and put the directory, called `large_files` in this directory, and run `move_files.sh`