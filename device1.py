import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define First Half of Model (Layers 1-5)
class FirstHalf(nn.Module):
    def __init__(self):
        super(FirstHalf, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)  # Send this to Device 2

# Instantiate model
model_part1 = FirstHalf()
model_part1.eval()

def run_inference(input_tensor):
    """Runs the first part of inference and sends activations to Device 2."""
    with torch.no_grad():
        intermediate_output = model_part1(input_tensor)

    # Convert tensor to list for transmission
    payload = {"tensor": intermediate_output.tolist()}

    # Send to Device 2 (D2) via HTTP
    response = requests.post("http://DEVICE2_IP:8000/infer", json=payload)

    if response.status_code == 200:
        return response.json()  # Final output from D2
    else:
        raise RuntimeError(f"Device 2 Error: {response.text}")

# Example input tensor
input_tensor = torch.randn(1, 128)
final_output = run_inference(input_tensor)
print("Final output from D2:", final_output)
