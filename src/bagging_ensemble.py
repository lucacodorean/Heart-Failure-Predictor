import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


class BaggingEnsemble:
    def __init__(self, base_model_class, input_dim=10, num_classes=3, n_models=10, lr=0.01, epochs=100, DEVICE="cpu", dropout_percentage = None):
        self.models = []
        self.optimizers = []
        self.epochs = epochs
        self.lr = lr
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.base_model_class = base_model_class
        self.n_models = n_models
        self.dropout_percentage = dropout_percentage

        for _ in range(n_models):
            if self.dropout_percentage is None:
                model = base_model_class(input_dim, num_classes).to(DEVICE)
            else:
                model = base_model_class(input_dim, self.dropout_percentage).to(DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            self.models.append(model)
            self.optimizers.append(optimizer)

    def fit(self, X, y, writer=None):
        X_np, y_np = X.numpy(), y.numpy()
        all_accuracies = {f"model_{i}": [] for i in range(self.n_models)} 

    
        bootstraps = [bootstrap_sample(X_np, y_np) for _ in range(self.n_models)]

        for epoch in range(self.epochs):
            epoch_accuracies = {}

            for i in range(self.n_models):
                model = self.models[i]
                optimizer = self.optimizers[i]
                criterion = nn.CrossEntropyLoss()

                X_boot_np, y_boot_np = bootstraps[i]
                X_boot = torch.tensor(X_boot_np, dtype=torch.float32)
                y_boot = torch.tensor(y_boot_np, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(X_boot)
                loss = criterion(outputs, y_boot)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    probs = torch.softmax(model(X), dim=1)
                    pred = torch.argmax(probs, dim=1)
                    acc = (pred == y).float().mean().item()

                epoch_accuracies[f"model_{i}"] = acc
                all_accuracies[f"model_{i}"].append(acc)
        
            avg_acc = sum(epoch_accuracies.values()) / len(epoch_accuracies)
            writer.add_scalar("Accuracy/Average Across Models", avg_acc, epoch)


    def predict(self, X):
        preds = [torch.softmax(model(X), dim=1).detach().numpy() for model in self.models]
        avg_preds = np.mean(preds, axis=0)
        return np.argmax(avg_preds, axis=1)
