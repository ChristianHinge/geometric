import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        x = [list(map(int, i)) for i in preprocessed_data["x"]]
        edge_index = [list(map(int, i)) for i in preprocessed_data["edge_index"]]
        batch = list(map(int, preprocessed_data["batch"]))

        x, edge_index, batch = (
            torch.as_tensor(x).type(torch.FloatTensor),
            torch.as_tensor(edge_index),
            torch.as_tensor(batch),
        )

        return x, edge_index, batch

    def inference(self, data, *args, **kwargs):
        x, edge_index, batch = data
        with torch.no_grad():
            model_output = self.model.forward(x, edge_index, batch)
        return model_output

    def postprocess(self, inference_output):
        postprocess_output = inference_output
        postprocess_output = F.softmax(postprocess_output)
        return torch.argmax(postprocess_output, dim=1).tolist()

    def handle(self, data, context):
        data = self.preprocess(data)
        model_output = self.inference(data)
        return self.postprocess(model_output)
