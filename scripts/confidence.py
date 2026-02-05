import torch
import torchvision
import argparse
import utility

# este script a partir de el dataset CIFAR10 y un golden de una inferencia sobre este, genera listas de confianza
# recibe el tipo de modelo con el que se creo el golden, carga el golden y el CIFAR 10
# guarda listas de indices de imagenes y su confianza de clasificacion 

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model that produced the golden. Ej: tiny, small", required=True)
    
    args = parser.parse_args()
    return args

def main(args):
    
    # load dataset for correct classification values
    dataset_class = torchvision.datasets.CIFAR10
    testset = dataset_class(root='./data', train=False, download=True)
    
    # load logit outputs of trained model
    outputs = torch.load(f"./goldens/golden_{args.model}.pt")

    # convert logits to probabilities
    outputs = torch.softmax(outputs, dim=1) 

    # load all predictions of trained model 
    _, preds = outputs.max(1)

    # storage of image index and confidence value  
    correct_outputs = {"index":[],"conf":[]}
    incorrect_outputs = {"index":[],"conf":[]}

    # run all outputs
    for i in range(len(outputs)):
        
        # find confidence
        largest, _ = torch.topk(outputs[i], 2)
        confidence = largest[0] - largest[1]  
        
        # compare output prediction vs correct classification
        if preds[i] == testset[i][1]: 
        
        # correct outputs
            # save output index
            correct_outputs["index"].append(i)
            # save confidence value
            correct_outputs["conf"].append(confidence)

        # incorrect outputs
        else: 
            # save output index
            incorrect_outputs["index"].append(i)
            # save confidence value
            incorrect_outputs["conf"].append(confidence)

    # saving files

    utility.create_directory("./confidence")
    torch.save(correct_outputs, f"./confidence/{args.model}_cnf_corr_out.pt")
    torch.save(incorrect_outputs, f"./confidence/{args.model}_cnf_inc_out.pt")

    # load correct file
    data = torch.load(f"./confidence/{args.model}_cnf_corr_out.pt")

    # divide by confidence 5 
    divided_outputs = [[],[],[],[],[]]
    for i in range(len(data["index"])):
        if data["conf"][i] <= 0.2:
            divided_outputs[0].append(data["index"][i])
        elif data["conf"][i] <= 0.4:
            divided_outputs[1].append(data["index"][i])
        elif data["conf"][i] <= 0.6:
            divided_outputs[2].append(data["index"][i])
        elif data["conf"][i] <= 0.8:
            divided_outputs[3].append(data["index"][i])
        elif data["conf"][i] <= 1:
            divided_outputs[4].append(data["index"][i])
    
    print("5 groups")
    for i in range(len(divided_outputs)):
        print(len(divided_outputs[i]))

    torch.save(divided_outputs, f"./confidence/{args.model}_cnf_corr_out_cat_5.pt")

    # divide by confidence 4
    divided_outputs = [[],[],[],[]]
    for i in range(len(data["index"])):
        if data["conf"][i] <= 0.25:
            divided_outputs[0].append(data["index"][i])
        elif data["conf"][i] <= 0.50:
            divided_outputs[1].append(data["index"][i])
        elif data["conf"][i] <= 0.75:
            divided_outputs[2].append(data["index"][i])
        elif data["conf"][i] <= 1:
            divided_outputs[3].append(data["index"][i])
    
    print("4 groups")
    for i in range(len(divided_outputs)):
        print(len(divided_outputs[i]))

    torch.save(divided_outputs, f"./confidence/{args.model}_cnf_corr_out_cat_4.pt")

if __name__ == '__main__':
    
    args = parse_args()
    main(args)