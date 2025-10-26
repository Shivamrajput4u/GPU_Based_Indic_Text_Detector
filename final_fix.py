import torch
import os

print("--- ☢️  Running Final Fix Script ---")
device = torch.device('cpu')

# Source: The last model we cleaned
source_path = 'detector/model_weights/best_crnn_cpu_final_model.pt'
# Destination: The final model your Django app will use
final_traced_path = 'detector/model_weights/best_crnn_traced_model.pt'

try:
    # Delete the old traced model if it exists to ensure a clean slate
    if os.path.exists(final_traced_path):
        print(f"Deleting old traced model at {final_traced_path}...")
        os.remove(final_traced_path)

    if not os.path.exists(source_path):
        print(f"❌ ERROR: The source model '{source_path}' is missing! Cannot proceed.")
    else:
        model = torch.jit.load(source_path, map_location=device)
        model.eval()

        # Create a dummy input tensor with the correct shape for tracing
        dummy_input = torch.randn(1, 3, 32, 128, device=device)

        # Re-trace the model to create a clean computation graph
        traced_model = torch.jit.trace(model, dummy_input)

        print(f"Saving new, clean TRACED model to: {final_traced_path}")
        traced_model.save(final_traced_path)
        print("\n✅ Final fix successful! The new model is ready.")

except Exception as e:
    print(f"❌ An error occurred: {e}")