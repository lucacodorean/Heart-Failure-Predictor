import torch.nn as nn

NUMBER_OF_NEURONS = 256                 # This is a hyperparameter for which it should be found the best value.  
                                        # From my little research, it should be a power of two. 64 is a good value to start
                                        # An analisys should be done in order to find the best value.

    

class StreamingPreferencesDatasetMLP(nn.Module):       # Any model inherits from nn.Module which denotes the base class for Pytorch model.
    def __init__(self, input_size=10, dropout_percentage=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, NUMBER_OF_NEURONS),
            nn.ReLU(),
            nn.BatchNorm1d(NUMBER_OF_NEURONS),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS, NUMBER_OF_NEURONS//2),
            nn.ReLU(),
            nn.BatchNorm1d(NUMBER_OF_NEURONS//2),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS//2, NUMBER_OF_NEURONS//4),
            nn.ReLU(),
            nn.BatchNorm1d(NUMBER_OF_NEURONS//4),
            nn.Dropout(dropout_percentage),

            nn.Linear(NUMBER_OF_NEURONS//4, NUMBER_OF_NEURONS//8),
            nn.ReLU(),

            nn.Linear(NUMBER_OF_NEURONS//8, 3)  
        )

    def forward(self, X):
        return self.model(X)
    
    