import argparse
import torch
from models.yolo import Model

def main(input_model_path, output_model_path):
    device = torch.device("cpu")
    cfg = "./models/detect/gelan-c.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=3)
    model = model.to(device)
    _ = model.eval()
    ckpt = torch.load(input_model_path, map_location='cpu')
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # Update the weights of the model
    model.load_state_dict(ckpt['model'].state_dict())

    # Save Model
    m_ckpt = {'model': model.half(),
              'optimizer': None,
              'best_fitness': None,
              'ema': None,
              'updates': None,
              'opt': None,
              'git': None,
              'date': None,
              'epoch': -1}
    torch.save(m_ckpt, output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to convert YOLO model")
    parser.add_argument("input_model_path", type=str, help="Path to the input YOLO model")
    parser.add_argument("output_model_path", type=str, help="Path to save the converted YOLO model")
    args = parser.parse_args()
    
    main(args.input_model_path, args.output_model_path)
