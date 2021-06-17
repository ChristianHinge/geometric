FROM python:3.8
RUN echo "Hello from custom container!"
RUN ls

RUN conda create -p cool_env python=3.8
RUN pip install --upgrade pip

RUN pip install torch
RUN pip install azureml-core
RUN pip install azureml-widgets
RUN pip install azureml-defaults
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install torchvision
RUN pip install wandb
RUN pip install joblib
RUN pip install python-dotenv
