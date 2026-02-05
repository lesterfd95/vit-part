import torch
import argparse

# este script crea un dataset que es un subset del dataset completo
# el subset a crear puede ser basado en un rango especifico de indices de imagenes o
# puede ser basado en confianza, para ello se usan listas pre-elaboradas de indices distribuidos en grupos de confianza
# para basado en confianza se tiene que pasar de argumetno un arreglo de tamaños para saber cuantos grupos de confianza
# hacer y cuantos elementos poner en cada uno


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Model. Ej: tiny, small", required=True)
    parser.add_argument("--mode", type=str, help="Mode of division. 'range' for providing a range. 'conf' for using a confidence file and provide group sizes", required=True)
    parser.add_argument("--range", type=int, nargs=2, help="Range for range mode of division. End value not included in range. Ex --range 0 512")
    parser.add_argument("--group_sizes", type=int, nargs='+' ,help="Group sizes for confidence mode. Ex --group_sizes 64 64 128 256")
    
    args = parser.parse_args()
    return args

args = parse_args()

# --- CONFIGURACIÓN ---
input_file = f"./module_datasets/vit_{args.model}_attn_dataset.pt" # Dataset original


# --- CARGAR DATASET ORIGINAL ---
data = torch.load(input_file)

# Suponemos que el .pt es un diccionario con 'inputs' y 'targets'
inputs = data['inputs']
targets = data['targets']

# --- VERIFICAR DIMENSIONES ---
print(inputs.shape)
print(targets.shape)
if len(inputs) != len(targets):
    raise ValueError("inputs y targets no tienen la misma cantidad de elementos")

# --- CREAR SUBCONJUNTO ---
if args.mode == "range":
    # intervalo
    start_idx, end_idx = args.range # Índice inicial (inclusive) Índice final (exclusive)
    output_file = f"./module_datasets/vit_{args.model}_attn_dataset_{start_idx}_{end_idx}.pt" # Nuevo dataset
    inputs_subset = inputs[start_idx:end_idx]
    targets_subset = targets[start_idx:end_idx]


elif args.mode == "conf":
    # elementos en posiciones especificas
    group_sizes = args.group_sizes
    group_number = len(group_sizes)
    specific_index = torch.load(f"./confidence/{args.model}_cnf_corr_out_cat_{group_number}.pt")
    output_file = f"./module_datasets/vit_{args.model}_attn_dataset_cnf_{group_number}.pt" # Nuevo dataset

    print("tamaños de grupos disponibles")
    for i in range(len(specific_index)):
        print(len(specific_index[i]))

    inputs_list = []
    targets_list = []
    for group_id, group_size in enumerate(group_sizes):
        idxs = specific_index[group_id][:group_size]

        # convertir listas a tensor 
        idxs = torch.as_tensor(idxs, dtype=torch.long)

        inputs_list.append(inputs[idxs])
        targets_list.append(targets[idxs])

    print("tamaños de grupos usados")
    for i in range(len(inputs_list)):
        print(len(inputs_list[i]))

    inputs_subset = torch.cat(inputs_list, dim=0)
    targets_subset = torch.cat(targets_list, dim=0)

    print("Subset size:", inputs_subset.shape[0])
    print("Expected:", sum(group_sizes))

    assert inputs_subset.shape[0] == sum(group_sizes)
    assert targets_subset.shape[0] == sum(group_sizes)


print(inputs_subset.shape)
print(targets_subset.shape)

# --- GUARDAR NUEVO DATASET ---
subset_data = {
    'inputs': inputs_subset.clone(),
    'targets': targets_subset.clone()
}

torch.save(subset_data, output_file)
if args.mode == "range":
    msg = f"Nuevo dataset guardado en '{output_file}' con {len(inputs_subset)} elementos"
    print(msg)
elif args.mode == "conf":
    msg = f"Nuevo dataset guardado en '{output_file}' con {len(inputs_subset)} elementos, tamaños de grupo: {group_sizes}"
    print(msg)
    
with open("./confidence/creation_log.log", "a") as f:
    f.write(msg + "\n")
