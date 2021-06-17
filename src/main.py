import argparse
import os
from datetime import datetime
import configparser
from src import settings

from src.models.train_model import train

class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Script for either training or evaluating',
            usage='python main.py <command>'
        )
        parser.add_argument(
            '--train', '-t', dest='train', action='store_true', help='Train model')
        parser.add_argument(
            '--test', '-v', dest='test', help='Test model from given path')
        parser.add_argument(
            '--name','-n', dest='name', help='Name for model', type=str, default='no_name'
        )
        parser.add_argument(
            '-c', '--config_section', action="store",type=str, help="Name of the config section for overwriting default values"
        )
        
        self.args = parser.parse_args()

        self.config = configparser.ConfigParser()
        print(settings.MODULE_PATH)
        if self.args.train:
            self.config.read(os.path.join(settings.MODULE_PATH,'src','config', 'train_config.ini'))
        elif self.args.test:
            self.config.read(os.path.join(settings.MODULE_PATH,'src','config','eval_config.ini'))

        if self.args.config_section is not None:
            if self.args.config_section in self.config:
                self.settings = self.config[self.args.config_section]
            else:
                raise KeyError(f'Config {self.args.config_section} not found in configuration file')
        else: 
            self.settings = self.config["DEFAULT"]
        

    def run_train(self):
        #uses self.settings
        kwargs = {'lr':float(self.settings["LearningRate"]),
        'epochs':int(self.settings["Epochs"]),
        'batch_size':int(self.settings["BatchSize"]),
        'p':float(self.settings["DropOutRate"]),
        'layers':[int(layer_size) for layer_size in self.settings["Layers"].split(" ")],
        'GPU':bool(self.settings["GPU"] == "True"),
        'name':str(self.args.name)}

        train(**kwargs)

    def run_eval(self):
        #uses self.settings
        pass

def main():
    argument_parser = ArgumentParser()
    if argument_parser.args.train:
        argument_parser.run_train()

if __name__ == '__main__':
    main()
