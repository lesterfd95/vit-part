import torch
import utility
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Model. Ej: tiny, small", required=True)

    args = parser.parse_args()
    return args

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Cargar pesos
print("Cargando pesos")
checkpoint_path = f"./trained_models/vit_{args.model}_cifar10.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)

partial_state_dict = {}
for k, v in checkpoint["model_state_dict"].items():
    if k.startswith("transformer.layers.0.0.fn."):
        new_key = k.replace("transformer.layers.0.0.fn.", "")
        partial_state_dict[new_key] = v

checkpoint = {"model_state_dict": partial_state_dict}
utility.create_directory("./trained_models")
torch.save(checkpoint, f"./trained_models/vit_{args.model}_part_att.pth")
print(f"Modelo guardado en ./trained_models/vit_{args.model}_part_att.pth")