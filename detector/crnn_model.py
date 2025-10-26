import torch
import torch.nn as nn

class CRNNModel(nn.Module):
    """
    A standard CRNN model for text recognition.
    It consists of a CNN backbone, a recurrent LSTM network, and a final linear layer.
    """
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()

        # --- 1. CNN Backbone ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=(2,1)), nn.BatchNorm2d(512), nn.ReLU(True)
        )

        # --- 2. Recurrent Network (LSTM) ---
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # --- 3. Final Linear Layer ---
        self.fc = nn.Linear(
            in_features=256 * 2,
            out_features=num_classes
        )

    def forward(self, x):
        # Pass through CNN
        features = self.cnn(x)
        
        # Reshape for RNN
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        
        # Pass through LSTM
        rnn_output, _ = self.lstm(features)
        
        # Pass through final classification layer
        output = self.fc(rnn_output)
        
        return output