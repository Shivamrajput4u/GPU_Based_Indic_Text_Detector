import torch
import cv2

def ctc_greedy_decoder(output, alphabet, blank=0):
    """
    Decodes CRNN output probabilities into text using a simple greedy CTC decoder.

    Args:
        output (torch.Tensor): Output from CRNN network, shape [seq_len, batch, num_classes].
        alphabet (str): String of characters representing valid classes (excluding blank).
        blank (int): Index reserved for the CTC blank token.

    Returns:
        str: Decoded text string.
    """
    out_best = output.argmax(2)  # [seq_len, batch]
    out_best = out_best.squeeze(1).detach().cpu().numpy()
    prev = blank
    decoded = []
    for c in out_best:
        if c != prev and c != blank:
            decoded.append(alphabet[c - 1])  # c-1 because blank is 0
        prev = c
    return ''.join(decoded)

def draw_boxes_with_labels(img, detections):
    """
    Draws bounding boxes and recognized text on the image.

    Args:
        img (numpy.ndarray): Input image in BGR format.
        detections (list): List of dicts with 'box' tuple and 'text' string.

    Returns:
        numpy.ndarray: Image with drawn boxes and text.
    """
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['text']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
    return img

def load_torch_model(model_path, model_class=None, **kwargs):
    """
    Loads a torch model from file. Can load state_dict into model_class or entire model.

    Args:
        model_path (str): Path to torch model .pth or .pt file.
        model_class (torch.nn.Module): Model class to instantiate if loading state_dict.
        kwargs: Arguments passed to model_class constructor.

    Returns:
        torch.nn.Module: Loaded model set to eval mode.
    """
    if model_class:
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model
