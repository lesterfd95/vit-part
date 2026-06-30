import torch
import utility
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model", choices=["tiny", "small", "base"], required=True)
    parser.add_argument("--submodel", type=str, help="submodel", choices=["embedding", "attention", "transformer", "classifier", "full"] , required=True)
    parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
    parser.add_argument("--atol", type=float, help="Absolute tolerance.", default=1e-6)
    parser.add_argument("--rtol", type=float, help="Relative tolerance.", default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inject", action="store_true", help="Inject errors")
    parser.add_argument("--no-inject", dest="inject", action="store_false")
    parser.set_defaults(inject=False)

    args = parser.parse_args()
    print(f"Using submodel {args.submodel} from ViT-{args.model}")
    print(f"Attempting evaluation over the data in {args.dataset} with batch size {args.batch_size}")
    print(f"Errors Injected: {args.inject}")
    print(f"Error detection Absolute tolerance: {args.atol} and Relative tolerance: {args.rtol}")
    return args

args = parse_args()

def compare(y, y_golden, atol=1e-6, rtol=1e-5):

    # máscara de elementos correctos
    mask = torch.isclose(y, y_golden, rtol=rtol, atol=atol)

    # detectar batchs fallidos
    faulty = (~mask).view(y.shape[0], -1).any(dim=1)

    # determinar índices de batch fallidos
    bad_idx = torch.nonzero(faulty, as_tuple=True)[0]

    if bad_idx.numel() == 0:
        return None  # No hay fallos, no devolvemos nada

    # extraer tensores con fallos
    tensors_bad = y[bad_idx].detach()

    return {"index": bad_idx, "tensors": tensors_bad}
    
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

batch_size = args.batch_size

# Crear sub modelo 
if args.submodel == "full":
    model = utility.build_model(model_name=args.model)
else:
    model = utility.build_submodel(model_name=args.model, submodel_name=args.submodel)

# Inspeccionar modulos
utility.print_module_list(model)

# Prepare dataset
print("Cargando datasets")
data = torch.load(args.dataset)

# introducir error (cambiando valores de salida con los que se compara)
if args.inject:
    if args.submodel == "classifier" or args.submodel == "full":
        data["targets"][123][0] = torch.tensor(-4.8128, dtype=torch.float32)
        data["targets"][150][0] = torch.tensor(12, dtype=torch.float32)
        data["targets"][200][1] = torch.tensor(torch.inf)
        data["targets"][500][9] = torch.tensor(torch.nan)
    else:
        data["targets"][150][34][13] = torch.tensor(12, dtype=torch.float32)
        data["targets"][150][34][54] = torch.tensor(12, dtype=torch.float32)
        data["targets"][200][5][5] = torch.tensor(torch.inf)
        data["targets"][201][2][23] = torch.tensor(torch.nan)
        data["targets"][300][34][13] = torch.tensor(12, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(data["inputs"], data["targets"])
testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Cargar pesos
print("Cargando pesos")
checkpoint_path = f"./trained_models/vit_{args.model}_{args.submodel}.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint["model_state_dict"]

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.to(device)

# Probar
print("Haciendo inferencia")
model.eval()
fault_log = {"index": [], "tensors": []}

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        print(f"Batch {batch_idx}")
        outputs = model(inputs)      
        
        # comprobar errores
        faults = compare(outputs,targets,atol=args.atol,rtol=args.rtol)
        if faults is not None:
            faults["index"] = faults["index"] + batch_idx * batch_size
            print(f"error in output tensors {faults['index']}")

            fault_log["index"].append(faults["index"].cpu())
            fault_log["tensors"].append(faults["tensors"].cpu())
    
if fault_log["index"]:
    fault_log["index"] = torch.cat(fault_log["index"], dim=0)
    fault_log["tensors"] = torch.cat(fault_log["tensors"], dim=0)

#print(f"error in output tensors:")
#for i in fault_log["index"]:
#    print(f"{i}")
#torch.save(fault_log, "faulty_outputs.pt")