#FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04
FROM ubuntu:20.04
RUN apt-get update -y
#RUN sudo apt-get update -y
RUN apt-get install -y python3.8
RUN apt-get install -y python3-venv
RUN echo "Hello from custom container!"

RUN python3.8 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install wheel
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

#TO GET torch-sparse working
RUN apt-get install -y aptitude
RUN apt install -y build-essential
RUN apt-get install -y python3-dev
#RUN apt-get install gcc-5 g++-5
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
RUN /opt/venv/bin/pip install torch-sparse

RUN /opt/venv/bin/pip install torch-scatter 
RUN /opt/venv/bin/pip install pytorch_lightning

#COPY . /app
#RUN /opt/venv/bin/pip install -e /app


#CMD ["/opt/venv/bin/python","app/src/main.py","--train"]