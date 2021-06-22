from azureml.core.authentication import (
    AzureCliAuthentication,
    InteractiveLoginAuthentication,
)
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
import configparser
import argparse
from src import settings
import os


from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", "-t", dest="train", action="store_true", help="Train model"
)
parser.add_argument("--test", "-v", dest="test", help="Test model from given path")
parser.add_argument(
    "-c",
    "--config_section",
    action="store",
    type=str,
    help="Name of the config section for overwriting default values",
)
parser.add_argument(
    "-aml_c",
    "--aml_config_section",
    action="store",
    type=str,
    help="Name of the config section for overwriting default azure run",
)

args = parser.parse_args()
cfg = configparser.ConfigParser()
cfg.read(os.path.join("src", "config", "aml_config.ini"))

if args.aml_config_section is not None:
    cfg = cfg[args.aml_config_section] 
else:
    cfg = cfg["DEFAULT"]


ws = Workspace.from_config(os.path.join(settings.CLOUD_PATH,"config.json"))

## Uncomment all below if you are not able to access workspace ##

#interactive_auth = InteractiveLoginAuthentication(
#    tenant_id="your tenant id",
#    force=True
#)

#ws = Workspace(
#    subscription_id="",
#    resource_group="Geometric-Group",
#    workspace_name="geometric-ws",
#    auth=interactive_auth,
#)

compute_target = ws.compute_targets[cfg["ComputeTarget"]]

# start compute target
# compute_target.start(wait_for_completion=True,show_output=True)

# DOCKER
env = Environment(name="geo-docker-train")
env.docker.enabled = True
env.docker.base_image = None
env.docker.base_dockerfile = "src/cloud/Dockerfile"
env.python.user_managed_dependencies = True
env.python.interpreter_path = "/opt/venv/bin/python"
env.environment_variables = {"WANDB_KEY": os.getenv("WANDB_KEY")}

# Arguments for main.py
arguments = ["--aml"]

if args.train:
    arguments.append("--train")
if args.test:
    arguments.append("--test")
if args.config_section:
    arguments.append("-c")
    arguments.append(args.config_section)


config = ScriptRunConfig(
    environment=env,  # set the python environment
    source_directory=".",
    script="src/main.py",
    compute_target=compute_target,
    arguments=arguments,
)

exp = Experiment(ws, cfg["Experiment"])
run = exp.submit(config)
print(run.get_portal_url())

# Wait until trial is done to stop compute target
run.wait_for_completion()
compute_target.stop(wait_for_completion=True, show_output=True)
