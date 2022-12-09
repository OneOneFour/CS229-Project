import torch.nn as nn
import torch.nn.functional as nnf
import torch 

OUTPUT_FEATURES = 4

class TCPredict(nn.Module):
    def __init__(self,initial_timesteps=8,initial_features=8,fc_width=50):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=initial_features,out_channels=20,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv1d(in_channels=20,out_channels=20,kernel_size=3,stride=1,padding=0)
        self.fc_1 = nn.Linear(in_features=20*(initial_timesteps-4),out_features=fc_width)
        self.output = nn.Linear(in_features=fc_width,out_features=4)
    
    def forward(self,x):
        x = nnf.relu(self.conv1(x))
        x = nnf.relu(self.conv2(x))
        x = torch.flatten(x,start_dim=0)
        x = nnf.relu(self.fc_1(x))
        return self.output(x)

class TCPredict2(nn.Module):
    def __init__(self,initial_timesteps=8,initial_features=8,fc_width=50):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=initial_features,out_channels=20,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv1d(in_channels=20,out_channels=40,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv1d(in_channels=40,out_channels=20,kernel_size=3,stride=1,padding=0)
        self.fc_1 = nn.Linear(in_features=20*(initial_timesteps-6),out_features=fc_width)
        self.output = nn.Linear(in_features=fc_width,out_features=4)
    
    def forward(self,x):
        x = nnf.relu(self.conv1(x))
        x = nnf.relu(self.conv2(x))
        x = nnf.relu(self.conv3(x))
        x = torch.flatten(x,start_dim=0)
        x = nnf.relu(self.fc_1(x))
        return self.output(x)