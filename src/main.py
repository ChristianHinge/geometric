from datetime import datetime
from datetime import datetime
from src import settings
from src.models import train_model, test_model
from src.models.wandb_hyp_sweep import hyp_opt_iter
from src.models.hyp_opt_config import sweep_config
import argparse
import configparser
import os
import wandb

class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Script for either training or evaluating',
            usage='python main.py <command>'
        )
        parser.add_argument(
            '--train', '-t', dest='train', action='store_true', help='Train model')
        parser.add_argument(
            '--test', '-v', dest='test', action='store_true', help='Test model by giving name')
        parser.add_argument(
            '--hyp_opt', '-ho', dest='hyp_opt', type=int, help='Do hyperparameter optimization with sweep_config file')    
        parser.add_argument(
            '-c', '--config_section', action="store",type=str, help="Name of the config section for overwriting default values"
        )
        parser.add_argument(
            '-mn', '--model_name', action='store', type=str, help='Model of name to test.',
        )
        parser.add_argument(
            '--aml',  action="store_true", help="Denoting whether on AML"
        )
        
        self.args = parser.parse_args()

        self.config = configparser.ConfigParser()

        if self.args.train:
            self.config.read(os.path.join(settings.MODULE_PATH,'src','config', 'train_config.ini'))
        elif self.args.test:
            self.config.read(os.path.join(settings.MODULE_PATH,'src','config','eval_config.ini'))

            if self.args.model_name is None:
                print("Need flag: -mn or --model_name {name of model}")
                exit(1)


        if self.args.config_section is not None:
            if self.args.config_section in self.config:
                self.settings = self.config[self.args.config_section]
            else:
                raise KeyError(f'Config {self.args.config_section} not found in configuration file')
        else:
            self.settings = self.config["DEFAULT"]

        self.time_name = datetime.strftime(datetime.now(), '%d-%H-%M')

        self.settings["AML"] = str(self.args.aml)


    def run_train(self):
        #uses self.settings
        kwargs = {'lr':float(self.settings["LearningRate"]),
        'epochs':int(self.settings["Epochs"]),
        'batch_size':int(self.settings["BatchSize"]),
        'p':float(self.settings["DropOutRate"]),
        'layers':[int(layer_size) for layer_size in self.settings["Layers"].split(" ")],
        'GPU':bool(self.settings["GPU"] == "True"), # match choice of compute instance to gpu if running on azure
        'name':str(self.time_name),
        'azure': self.args.aml
        }

        train_model.train(**kwargs)

    def run_eval(self):

        kwargs = {'project': str(self.settings['Project']),
                  'entity': str(self.settings['Entity']),
                  'filename': self.args.model_name}

        print(self.settings['Project'])
        print(self.settings['Entity'])
        print(self.args.model_name)


        test_model.eval(**kwargs)
    
    def run_hyp_opt(self,counts):
        wandb.login(key=os.getenv("WANDB_KEY"))
        sweep_id = wandb.sweep(sweep_config, project="geometric_hyp_opt", entity="classy_geometric")
        wandb.agent(sweep_id, function=hyp_opt_iter, count=counts,project="geometric_hyp_opt", entity="classy_geometric")
        

def main():
    argument_parser = ArgumentParser()
    if argument_parser.args.train:
        argument_parser.run_train()

    if argument_parser.args.test:
        argument_parser.run_eval()
    
    if argument_parser.args.hyp_opt:
        argument_parser.run_hyp_opt(argument_parser.args.hyp_opt)

if __name__ == '__main__':
    main()
