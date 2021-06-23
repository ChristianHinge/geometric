from azureml.core import Workspace

ws = Workspace.from_config("src/cloud/config.json")

print(ws.webservices)

# Choose the webservice you are interested in

from azureml.core import Webservice

service = Webservice(ws, 'geo-service2')
print(service.get_logs())