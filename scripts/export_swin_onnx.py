import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torchvision.transforms as transforms

def export_swin_to_onnx(model_path, output_path="swin_fairface_best.onnx"):
    # Load the model
    num_classes = 7  # Number of classes in FairFace
    model = AutoModelForImageClassification.from_pretrained(
        'microsoft/swinv2-base-patch4-window16-256',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Get processor for correct normalization
    processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256")
    mean = processor.image_mean
    std = processor.image_std

    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported successfully to {output_path}")

if __name__ == "__main__":
    # Replace with your model path
    model_path = "swinv2_fairface_best.pth"
    export_swin_to_onnx(model_path) 