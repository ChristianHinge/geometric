import argparse
import configparser
import os
import sys

import hydra
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.authentication import (AzureCliAuthentication,
                                         InteractiveLoginAuthentication)
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig

from src import settings
from src.settings import paths


def main():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    args = sys.argv[1:]
    target = None
    cfg = configparser.ConfigParser()
    cfg.read("src/config/aml_config.ini")

    if "--gpu" in args:
        target = "geo-gpu"
        args.remove("--gpu")
    elif "--cpu" in args:
        target = "geo-cpu"
        args.remove("--cpu")
    else:
        target = cfg["DEFAULT"]["target"]

    if "-e" in args:
        exp = args[args.index("-e") + 1]
        args.remove("-e")
        args.remove(exp)
    else:
        exp = cfg["DEFAULT"]["exp"]

    ws = Workspace.from_config(os.path.join(paths.CLOUD_PATH, "config.json"))
    compute_target = ws.compute_targets[target]

    # DOCKER
    env = Environment(name="geo-docker-train")
    env.docker.enabled = True
    env.docker.base_image = None
    env.docker.base_dockerfile = "src/cloud/Dockerfile.train"
    env.python.user_managed_dependencies = True
    env.environment_variables = {"WANDB_KEY": os.getenv("WANDB_KEY")}

    config = ScriptRunConfig(
        environment=env,  # set the python environment
        source_directory=".",
        script="src/main.py",
        compute_target=compute_target,
        arguments=["azure=True"] + args,
    )

    exp = Experiment(ws, exp)
    run = exp.submit(config)
    print(run.get_portal_url())


if __name__ == "__main__":
    main()
