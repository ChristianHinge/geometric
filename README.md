Geometric Project
==============================

The final project for Machine Learning Operations June course at DTU.

# Description
The overall goal of the project is to apply MLOps techniques to deep learning model workflow. The main idea behind that is to automate processes responsible for retrieving data, training and validation of the model, returning predictions for given input, and do visualisations of the results, model parameters, and scores. Another aim of the project is to incorporate the best practices during code development, such as _unit testing_ (maybe even _Test-Driven Development_), _Continuous Integration_ pipeline where the test will be run and all things connected to automated code formatting (linting, code smell, optimising imports, etc.). The project will be extended with a new things that the exercises will introduce throughout the following weeks.

For the deep learning ecosystem the PyTorch Geometric was selected, thus graphs will be used as an input. The category of the problem is going to be graph classification and the domain is biochemistry. However, we are still quite unsure about the exact dataset we are going to use. Initially, we decided to use a set from the TU Dortmund collection called `MUTAG` which can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset). According to the [instruction](https://chrsmrrs.github.io/datasets/docs/datasets/) embedded into the dataset, it have two classes of 188 chemical compounds in total with 17.93	nodes and 19.79 edges on average.

Based on that, the classification problem for that domain typically touches the molecule property prediction – given a certain molecule graph, one tries to find out which properties the substance will have. This might be applied to e.g drug discovery, which might be helpful solving issues with pandemics by quick medicine design. For the chosen dataset, the model is going to predict classes of effects occurring whilst substance exposition to bacteria. According to that, the PyTorch Geometric library will be used, since it facilitates model creation and training processes when data has a graph format. Hence, a simple Graph Neural Network will be implemented with the [Graph Convolutional Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) to solve the given problem. The design of the network will be similar to the official ones provided by PyTorch Geometric documentation [here](https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=V2Q37tbHyQ6A).

To start with, using the `cookiecutter` project structure and ready Python environment, we will going to create a basic working net with the download data, training, and validation subroutines. The tests will cover the correctness of the pipeline. Afterwards, the GitHub Actions will be incorporated to cover the need of running automated tests and reformatting the code. After that, another functionalities of MLOps pipeline will be added to the project.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
