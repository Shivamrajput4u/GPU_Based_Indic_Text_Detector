import torch
from detector.models import ImprovedCRNN  # Update this import if your model class is elsewhere

# Path to your original .pt checkpoint
checkpoint_path = 'detector/model_weights/best_crnn_mozhi_model.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Instantiate your model with correct params
vocab_size = checkpoint.get('vocab_size', 851)  # Update if needed
model = ImprovedCRNN(num_classes=vocab_size, dropout=0.2)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cpu')
model.eval()

# Example input shape (adjust if needed)
example_input = torch.randn(1, 3, 32, 128)

# Trace the model on CPU
traced_model = torch.jit.trace(model, example_input)
traced_model.save('detector/model_weights/best_crnn_mozhi_model_cpu.pt')

print("✅ Model re-exported and saved to detector/model_weights/best_crnn_mozhi_model_cpu.pt")
