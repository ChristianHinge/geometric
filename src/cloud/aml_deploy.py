import azureml.core
from azureml.core import Workspace,Webservice
import os
from azureml.core.webservice import AciWebservice, LocalWebservice
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core import Environment
import omegaconf

def deploy(cfgs):
    # Load the workspace from the saved config file
    ws = Workspace.from_config("src/cloud/config.json")
    print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

    model = ws.models[cfgs['filename']]
    print(model.name, 'version', model.version)

    # Set path for scoring script
    script_file = os.path.join("src/cloud/score.py")

    #DOCKER 
    #env = Environment.from_pip_requirements("test_env","src/cloud/req_test.txt")
    env = Environment("test")
    env.docker.enabled = True
    env.docker.base_image = None
    env.docker.base_dockerfile = "src/cloud/Dockerfile.3"
    env.python.user_managed_dependencies=True

    inference_config = InferenceConfig(entry_script=script_file,
                                    environment=env)

    if cfgs["local"]:
        deployment_config = LocalWebservice.deploy_configuration(port=6790)
    else:
        deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
    
    service_name = cfgs['service-name']
    #service  = Webservice(name=service_name, workspace=ws)
    #service.update(inference_config=inference_config)
    service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

if __name__ == "__main__":
    cfgs = omegaconf.OmegaConf.load("src/config/deployment.yaml")
    deploy(cfgs)