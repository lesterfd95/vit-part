import torch
import utility
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model", choices=["tiny", "small", "base"], required=True)
    parser.add_argument("--submodel", type=str, help="submodel", choices=["embedding", "attention", "transformer", "classifier"] , required=True)

    args = parser.parse_args()
    return args

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Cargar pesos
print("Cargando pesos")
checkpoint_path = f"./trained_models/vit_{args.model}_cifar10.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)

model = utility.build_submodel(args.model, args.submodel)
print(f"SUBMODEL {args.submodel}")
utility.print_module_list(model)


# Save a state dictionary with only the layers of the submodel
# Convert names of full model layers to names of submodel layers 
partial_state_dict = {}
for k, v in checkpoint["model_state_dict"].items():
    
    if args.submodel == "embedding":
        if k.startswith("to_patch_embedding"):
            new_key = k
            partial_state_dict[new_key] = v
        elif k == "pos_embedding":
            new_key = k
            partial_state_dict[new_key] = v
        elif k == "cls_token":
            new_key = k
            partial_state_dict[new_key] = v

    elif args.submodel == "attention":
        if k.startswith("transformer.layers.0.0.fn."):
            new_key = k.replace("transformer.layers.0.0.fn.", "")
            partial_state_dict[new_key] = v

    elif args.submodel == "transformer":
        if k.startswith("transformer.layers.0.0"):
            new_key = k.replace("transformer.layers.0.0", "attn")
            partial_state_dict[new_key] = v
        elif k.startswith("transformer.layers.0.1"):
            new_key = k.replace("transformer.layers.0.1", "ff")
            partial_state_dict[new_key] = v

    elif args.submodel == "classifier":
        if k.startswith("mlp_head.0"):
            new_key = k.replace("mlp_head.0", "norm")
            partial_state_dict[new_key] = v
        elif k.startswith("mlp_head.1"):
            new_key = k.replace("mlp_head.1", "fc")
            partial_state_dict[new_key] = v

print("Saved:")
print(partial_state_dict.keys())

checkpoint = {"model_state_dict": partial_state_dict}
utility.create_directory("./trained_models")
torch.save(checkpoint, f"./trained_models/vit_{args.model}_{args.submodel}.pth")
print(f"Modelo guardado en ./trained_models/vit_{args.model}_{args.submodel}.pth")