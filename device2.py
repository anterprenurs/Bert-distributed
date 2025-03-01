import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from ray import serve
from fastapi import FastAPI, Request

# Initialize Ray
ray.init()
serve.start()

# Define Second Half of Model (Layers 6-10)
class SecondHalf(nn.Module):
    def __init__(self):
        super(SecondHalf, self).__init__()
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 256)
        self.fc10 = nn.Linear(256, 10)  # Final output

    def forward(self, x):
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        return self.fc10(x)  # Final output

# Instantiate model
model_part2 = SecondHalf()
model_part2.eval()

# FastAPI for HTTP request handling
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class ModelInference:
    async def infer(self, request: Request):
        data = await request.json()
        tensor_data = torch.tensor(data["tensor"])

        with torch.no_grad():
            final_output = model_part2(tensor_data)

        return {"output": final_output.tolist()}

# Deploy with route prefix specified here instead
serve.run(ModelInference.bind(), route_prefix="/infer")
