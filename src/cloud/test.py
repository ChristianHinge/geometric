import azureml
from azureml.core import Webservice, Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config("src/cloud/config.json")
print("Ready to use Azure ML {} to work with {}".format(azureml.core.VERSION, ws.name))
# Choose the webservice you are interested i

service = Webservice(ws, "endpoint-2")
print(service.get_logs())
