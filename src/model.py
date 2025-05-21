import torch.nn as nn

NUMBER_OF_NEURONS = 128  

class StreamingPreferencesDatasetMLP(nn.Module):     
    def __init__(self, input_size=10, dropout_percentage=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, NUMBER_OF_NEURONS),
            nn.BatchNorm1d(NUMBER_OF_NEURONS),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS, NUMBER_OF_NEURONS//2),
            nn.BatchNorm1d(NUMBER_OF_NEURONS//2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS//2, NUMBER_OF_NEURONS//4),
            nn.BatchNorm1d(NUMBER_OF_NEURONS//4),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS//4, NUMBER_OF_NEURONS//8),
            nn.ReLU(),

            nn.Linear(NUMBER_OF_NEURONS//8, 3)  
        )

    def forward(self, X):
        return self.model(X)
    
    