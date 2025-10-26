import torch
import os

print("--- 🚀 Retracing Model for Final Fix ---")
device = torch.device('cpu')

# The source is the last good model we created
source_path = 'detector/model_weights/best_crnn_cpu_final_model.pt'
# The destination is the file your code is now looking for
final_traced_path = 'detector/model_weights/best_crnn_traced_model.pt'

try:
    if not os.path.exists(source_path):
        print(f"❌ ERROR: The source model '{source_path}' is missing!")
    else:
        model = torch.jit.load(source_path, map_location=device)
        model.eval()
        
        # Create a dummy input with the correct shape (batch, channels, height, width)
        dummy_input = torch.randn(1, 3, 32, 128, device=device)
        
        # Re-trace the model. This records its operations and creates a clean graph.
        traced_model = torch.jit.trace(model, dummy_input)
        
        print(f"Saving new TRACED model to: {final_traced_path}")
        traced_model.save(final_traced_path)
        print("\n✅ Retracing successful! This should be the final fix.")

except Exception as e:
    print(f"❌ Error during re-tracing: {e}")