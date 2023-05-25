import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(868, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150)
        )

    def forward(self, x):
        x = self.fusion(x)
        return x

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__() 
        
    def forward(self, anchor, positive, negative, margin=0.5):
        # compute euclidean distance between the anchor and the positive
        distance_positive = (anchor - positive).pow(2).sum(1)
        # compute euclidean distance between the anchor and the negative
        distance_negative = (anchor - negative).pow(2).sum(1)
        # use relu to make sure the loss is always positive
        losses = torch.relu(distance_positive - distance_negative + margin)
        # return the mean loss over the current batch
        return losses.mean()
    