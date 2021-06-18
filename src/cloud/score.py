import json
import joblib
import numpy as np
#from azureml.core.model import Model
import dotenv
import os
import torch

# ws = Workspace.from_config("src/cloud/config.json")
# print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
# model = ws.models['diabetes_model']

# Called when the service is loaded
def init():

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir,os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)
    #dotenv.load_dotenv(".env")
    global model
    # Get the path to the deployed model file and load it
    #model_path = Model.get_model_path('diabetes_model')
    #model = ws.models['diabetes_model']#
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = torch.from_numpy(np.array(json.loads(raw_data)['data']))#.type(torch.DoubleTensor)
    # Get a prediction from the model
    predictions = model.forward(data.float())
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = [str(x) for x in range(10)]
    json_res = []
    for prediction in predictions.data.numpy().tolist():
        json_res.append(dict(zip(classnames,prediction)))
    # Return the predictions as JSON
    return json.dumps(json_res)


