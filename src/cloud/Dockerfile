FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04
RUN echo "Hello from custom container!"

RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install torch
RUN /opt/venv/bin/pip install azureml-core
RUN /opt/venv/bin/pip install azureml-widgets
RUN /opt/venv/bin/pip install azureml-defaults
RUN /opt/venv/bin/pip install matplotlib
RUN /opt/venv/bin/pip install seaborn
RUN /opt/venv/bin/pip install torchvision
RUN /opt/venv/bin/pip install wandb
RUN /opt/venv/bin/pip install joblib
RUN /opt/venv/bin/pip install python-dotenv
RUN /opt/venv/bin/pip install torch-geometric
RUN /opt/venv/bin/pip install torch-sparse
RUN /opt/venv/bin/pip install torch-scatter
RUN /opt/venv/bin/pip install pytorch_lightning