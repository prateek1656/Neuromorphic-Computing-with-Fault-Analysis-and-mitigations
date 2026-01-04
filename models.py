
import torch.nn as nn
import torch

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Adjusted for CIFAR-10 dimensions
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass with safer implementation.
        """
        # Process through CNN layers
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = x
        if hasattr(self, 'memristive_model') and self.memristive_model is not self:
            with torch.set_grad_enabled(self.memristive_model.training):
                output = self.memristive_model(x)
                
        # Update stats if tracking is enabled
        if hasattr(self, 'total_operations'):
            self.total_operations += 1
            
            # Periodically check health if implemented
            if hasattr(self, 'check_health') and callable(self.check_health) and self.total_operations % 10 == 0:
                with torch.no_grad():
                    self.check_health()
        
        return output
