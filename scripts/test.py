import torch
import utility
import argparse

# Este script realiza inferencia de un modelo con CIFAR10 y genera un golden y datasets para modulos internos del modelo 
# recibe nombre del modelo y tamaño de batch, carga los pesos de un .pth y realiza inferencia
# guarda un golden con el tensor de salida del modelo (logits) y datasets (entradas, salidas) de bloques internos seleccionados 

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model to test. Ej: tiny, small", required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    
    args = parser.parse_args()
    return args

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    batch_size = args.batch_size
    print(f"batch_size {batch_size}\n")

    model = utility.build_model(f"{args.model}")
    
    # hooks
    utility.print_module_list(model)

    # cada elemento sera un tensor correspondiente a una imagen (elemento del batch)
    all_attn_inputs = [] 
    all_attn_outputs = []

    # definir hooks
    def attn_hook(module, input, output):
        all_attn_inputs.append(input[0].detach().cpu().clone().contiguous())
        all_attn_outputs.append(output.detach().cpu().clone().contiguous())

    # enganchar hooks
    attn_module = model.transformer.layers[0][0].fn # sacado de la lista al inspeccionar modulos
    handle = attn_module.register_forward_hook(attn_hook)

    # dataloader
    testloader = utility.build_dataloader_validation(batch_size)

    # Cargar pesos
    print("Loading weights")
    checkpoint_path = f"./trained_models/vit_{args.model}_cifar10.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.to(device)

    # Probar
    print("Forward propagation")
    model.eval()
    correct = 0
    total = 0

    all_outputs = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)      
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            all_outputs.append(outputs)

    accuracy = 100.0 * correct / total
    print(f"Accuracy CIFAR-10: {accuracy:.2f}%")

    # eliminar hooks
    handle.remove()

    # lo siguiente hace que pase a ser un arreglo donde cada elemento 
    # es un tensor independiete del tamaño del batch
    # como si se hubiesen recolectado en inferencia de tamaño de batch = 1 
    all_outputs = torch.cat(all_outputs, dim= 0)
    print(f"Golden output shape: {all_outputs.shape}")
    attn_inputs = torch.cat(all_attn_inputs, dim=0)
    print(f"attn_inputs output shape: {attn_inputs.shape}")
    attn_outputs = torch.cat(all_attn_outputs, dim=0)
    print(f"attn_outputs output shape: {attn_outputs.shape}") 

    utility.create_directory("./module_datasets")
    print(f"Saving attention dataset in ./module_datasets/vit_{args.model}_attn_dataset.pt") 
    torch.save({"inputs": attn_inputs,"targets": attn_outputs,},f"./module_datasets/vit_{args.model}_attn_dataset.pt")

    utility.create_directory("./goldens")
    print(f"Saving golden in ./goldens/golden_{args.model}.pt") 
    torch.save(all_outputs,f"./goldens/golden_{args.model}.pt")

if __name__ == '__main__':
    
    args = parse_args()
    main(args)
