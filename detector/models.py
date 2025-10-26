import torch
import torch.nn as nn

class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  # Regularization

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  # Regularization

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Dropout2d(dropout),  # Regularization

            nn.Conv2d(256, 256, 2), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.rnn = nn.LSTM(256*3, 128, 1, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)  # Regularization
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.view(b, c*h, w).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)  # Regularization
        return self.fc(x).permute(1, 0, 2)
