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
from src.settings import paths

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cpu",
    action="store_true",
    type=bool,
    help="",
)

parser.add_argument(
    "--gpu",
    action="store_true",
    type=bool,
    help="",
)

parser.add_argument(
    "-e",
    "--experiment",
    action="store",
    type=str,
    help="",
    default="exp-3"
)

args = parser.parse_args()

if args.gpu:
    target = "geo-gpu"
elif args.cpu:
    target = "geo-cpu"
else:
    raise Exception("Please choose either gpu or cpu")

ws = Workspace.from_config(os.path.join(paths.CLOUD_PATH,"config.json"))

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

compute_target = ws.compute_targets[target]

#DOCKER
env = Environment(name="geo-docker-train")
env.docker.enabled = True
env.docker.base_image = None
env.docker.base_dockerfile = "src/cloud/Dockerfile.train"
env.python.user_managed_dependencies=True
#env.python.interpreter_path = "/opt/venv/bin/python"
env.environment_variables = {
    "WANDB_KEY":os.getenv("WANDB_KEY")
}

arguments = ["--aml"]

config = ScriptRunConfig(
    environment=env,  # set the python environment
    source_directory='.',
    script='src/main.py',
    compute_target = compute_target,
    arguments = [""]


exp = Experiment(ws, args.experiment)
run = exp.submit(config)
print(run.get_portal_url())

# Wait until trial is done to stop compute target
run.wait_for_completion()
compute_target.stop(wait_for_completion=True, show_output=True)
