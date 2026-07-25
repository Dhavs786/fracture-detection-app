from ultralytics import YOLO
import sys
from pathlib import Path

def main():
    # Find model weights
    model_path = "best (1).pt"
    model_files = list(Path(".").glob("best*.pt"))
    if len(model_files) > 0:
        model_path = str(max(model_files, key=lambda p: p.stat().st_mtime))
    
    print(f"Loading PyTorch model '{model_path}'...")
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    model = YOLO(model_path)
    print("Exporting model to ONNX format (dynamic input shapes)...")
    try:
        # Export with dynamic=True so we can run at 416, 640, 832, 1024 etc.
        onnx_path = model.export(format="onnx", imgsz=416, dynamic=True)
        print(f"Model successfully exported to: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
