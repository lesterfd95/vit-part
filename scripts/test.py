import torch
import utility
import argparse

# Este script realiza inferencia de un modelo con CIFAR10 y genera un golden y datasets para modulos internos del modelo 
# recibe nombre del modelo y tamaño de batch, carga los pesos de un .pth y realiza inferencia
# guarda un golden con el tensor de salida del modelo (logits) y datasets (entradas, salidas) de bloques internos seleccionados 

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model to test", choices=["tiny", "small", "base"], required=True)
    parser.add_argument("--submodel", type=str, help="submodel to capture dataset", choices=["embedding", "attention", "transformer", "classifier", "full"] , required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    
    args = parser.parse_args()
    return args

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    batch_size = args.batch_size
    print(f"batch_size {batch_size}\n")

    model = utility.build_model(f"{args.model}")

    if args.submodel == "full":
        print(f"Creating golden for full model\n")
    else:
        print(f"Creating dataset for {args.submodel} submodel\n")
    
    # lista de modulos del modelo
    #utility.print_module_list(model)
    
    # listas para guardar tensores de entrada y salida
    # cada elemento sera un tensor correspondiente a una imagen (elemento del batch)
    
    all_submodel_inputs = []
    all_submodel_outputs = []

    # definir hooks, seleccionar capas de la lista, enganchar hooks
    handles = []
    # embedding
    if args.submodel == "embedding":
        def emb_in_hook(submodel, input, output):
            all_submodel_inputs.append(input[0].detach().cpu().clone().contiguous())
        def emb_out_hook(submodel, input, output):
            all_submodel_outputs.append(input[0].detach().cpu().clone().contiguous())

        emb_submodel_in = model.to_patch_embedding[0]
        emb_submodel_out = model.dropout

        handles.append(emb_submodel_in.register_forward_hook(emb_in_hook)) 
        handles.append(emb_submodel_out.register_forward_hook(emb_out_hook))

    # attention
    elif args.submodel == "attention":
        def attn_hook(submodel, input, output):
            all_submodel_inputs.append(input[0].detach().cpu().clone().contiguous())
            all_submodel_outputs.append(output.detach().cpu().clone().contiguous())

        attn_submodel = model.transformer.layers[0][0].fn 

        handles.append(attn_submodel.register_forward_hook(attn_hook))

    # transformers
    elif args.submodel == "transformer":
        def tran_in_hook(submodel, input, output):
            all_submodel_inputs.append(input[0].detach().cpu().clone().contiguous())
        def tran_out_hook(submodel, input, output):
            all_submodel_outputs.append(input[0].detach().cpu().clone().contiguous())

        tran_submodel_in = model.transformer.layers[0][0]
        tran_submodel_out = model.transformer.layers[1][0]

        handles.append(tran_submodel_in.register_forward_hook(tran_in_hook))
        handles.append(tran_submodel_out.register_forward_hook(tran_out_hook))

    # classifier
    elif args.submodel == "classifier":
        def cla_hook(submodel, input, output):
            all_submodel_inputs.append(input[0].detach().cpu().clone().contiguous())
            all_submodel_outputs.append(output.detach().cpu().clone().contiguous())

        cla_submodel = model.mlp_head

        handles.append(cla_submodel.register_forward_hook(cla_hook))

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

            if args.submodel == "full":
                all_outputs.append(outputs)

    accuracy = 100.0 * correct / total
    print(f"Accuracy CIFAR-10: {accuracy:.2f}%")

    # eliminar hooks
    for h in handles:
        h.remove()

    # lo siguiente hace que pase a ser un arreglo donde cada elemento 
    # es un tensor independiete del tamaño del batch
    # como si se hubiesen recolectado en inferencia de tamaño de batch = 1 
    
    if args.submodel == "full":
        all_outputs = torch.cat(all_outputs, dim= 0)
        print(f"Golden output shape: {all_outputs.shape}")

        utility.create_directory("./goldens")
        print(f"Saving golden in ./goldens/golden_{args.model}.pt") 
        torch.save(all_outputs,f"./goldens/golden_{args.model}.pt")

    else:
        inputs = torch.cat(all_submodel_inputs, dim=0)
        print(f"{args.submodel} inputs shape: {inputs.shape}")
        outputs = torch.cat(all_submodel_outputs, dim=0)
        print(f"{args.submodel} outputs shape: {outputs.shape}") 

        utility.create_directory("./module_datasets")
        print(f"Saving embedding dataset in ./module_datasets/vit_{args.model}_{args.submodel}_dataset.pt") 
        torch.save({"inputs": inputs,"targets": outputs,},f"./module_datasets/vit_{args.model}_{args.submodel}_dataset.pt")
    

if __name__ == '__main__':
    
    args = parse_args()
    main(args)
