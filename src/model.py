import  torch.nn as nn

NUMBER_OF_NEURONS = 64                  # This is a hyperparameter for which it should be found the best value.  
                                        # From my little research, it should be a power of two. 64 is a good value to start
                                        # An analisys should be done in order to find the best value.

class HeartFailureMLP(nn.Module):       # Any model inherits from nn.Module which denotes the base class for Pytorch model.
    def __init__(self, input_size=11, dropout_percentage=0.3):
        super().__init__()
        self.model = nn.Sequential(                             # Describes the order of the layers.
            nn.Linear(input_size, NUMBER_OF_NEURONS),           # First layer
            nn.ReLU(),                                          # Apply ReLU on the output of Linear
            nn.Dropout(dropout_percentage),                     # Dropout function will ensure that the given % of the input values are set to 0 in order to prevent overfitting
            
            nn.Linear(NUMBER_OF_NEURONS, int(NUMBER_OF_NEURONS / 2)),  # Second layer in which I wanted to reduce the number of neurons
            nn.ReLU(),                                                 
            nn.Dropout(dropout_percentage),

            nn.Linear(int(NUMBER_OF_NEURONS / 2), 1),                  # Preparing input for the sigmoid function 
            nn.Sigmoid()
        )

    def forward(self, X):                   
        return self.model(X)