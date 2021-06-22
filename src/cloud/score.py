import json
import joblib
import numpy as np
from azureml.core.model import Model
import dotenv
import os
import torch
from src.models.model import GCN
from torch_geometric.data import Data, Batch
# ws = Workspace.from_config("src/cloud/config.json")
# print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
# model = ws.models['diabetes_model']

# Called when the service is loaded
def init():

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir,os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)
    global model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.ckpt')

    model = GCN.load_from_checkpoint(model_path)

    model.eval()
    #torch.no_grad()
    #global model
    #model = 2
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    #model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    #return json.dumps([model])
    d = json.loads(raw_data)
    x           = torch.from_numpy(np.array(d["x"])).float()
    edge_index  = torch.from_numpy(np.array(d["edge_index"]))
    edge_attr   = torch.from_numpy(np.array(d["edge_attr"])).float()

    data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr)
    batch = Batch.from_data_list([data])
    predictions = model.predict_step(batch)

    pred_list = predictions.data.numpy().tolist()
    return json.dumps(pred_list)

if __name__ == "__main__":
    init()
    x = [[1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0.]]

    edge_index = [[ 0,  0,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
          8,  8,  8,  9,  9, 10, 10, 10, 11, 12],
        [ 1,  9,  0,  2,  1,  3,  7,  2,  4,  8,  3,  5,  4,  6,  5,  7,  2,  6,
          3,  9, 10,  0,  8,  8, 11, 12, 10, 10]]
    
    edge_attr =[[1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.]]
    data = json.dumps({"x":x,"edge_index":edge_index,"edge_attr":edge_attr})

    a = run(data)
    print(a)
