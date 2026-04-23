from simulation_parameters import RAY_COUNT
import torch.nn as nn



INPUT_COUNT = RAY_COUNT + 2
OUTPUT_COUNT = 2



class RobocarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(INPUT_COUNT, 200)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(200, 200)
        self.act2 = nn.Tanh()
        self.output = nn.Linear(200, OUTPUT_COUNT)
        self.act_output = nn.Tanh()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
