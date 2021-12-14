import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, airports):
    super().__init__()
    self.weekday_embeddings = nn.Sequential(
        nn.Linear(7, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    self.airline_embeddings = nn.Sequential(
        nn.Linear(14, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    )
    self.fc = nn.Sequential(
        nn.Linear(13, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


  def forward(self, airlines, weekdays, distance, departure_delay):
    '''
      Forward pass.
    '''
    airlines_embedding = self.airline_embeddings(airlines)
    weekdays_embedding = self.weekday_embeddings(weekdays)
    x = torch.cat([airlines_embedding, weekdays_embedding, distance, departure_delay])
    return self.fc(x)
