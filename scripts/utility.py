import torch
import torchvision
import torchvision.transforms as transforms 
from vit import ViT
from vit import Attention
from vit import PatchEmbedding
from vit import TransformerBlock
from vit import ClassificationHead
import os

def build_model(model_name):

    # Build model 
    #image_size     tamaño de imagenes de entrada: 224x224 (estandar para ViT)
    #patch_size     tamaño de los patches: 16x16 (muy usado en ViT)
    #num_classes    numero de clases: depende del dataset
    #dim            dimension interna de la red (luego de proyectar): dim_head * heads
    #depth          numero de transformer encoders 
    #heads          numero de cabezas en cada mecanismo de atencion de cada transformer encoder 
    #mlp_dim        dimension interna en el MLP de cada transformer encoder: 4 * dim
    #channels       numero de canales en las imagenes: 3 (estandar en imagenes)
    #dim_head       dimension interna de cada cabeza del transformer: 64 es estandar
    #dropout        dropout luego del embedding
    #emb_dropout    dropout de los MLP del transformer encoder

    # ViT-Tiny-16
    if model_name == "tiny":
        print('Creating ViT-Tiny model')
        model = ViT(
            image_size = 224,   
            patch_size = 16,    
            num_classes = 1000,  
            dim = 192,           
            depth = 12,         
            heads = 3,          
            mlp_dim = 768,      
            channels = 3,       
            dim_head = 64,      
            dropout = 0.0,      
            emb_dropout = 0.0   
        )
    # ViT-Small-16
    elif model_name == "small":
        print('Creating ViT-Small model')
        model = ViT(
            image_size = 224,   
            patch_size = 16,    
            num_classes = 1000,  
            dim = 384,           
            depth = 12,         
            heads = 6,          
            mlp_dim = 1536,      
            channels = 3,       
            dim_head = 64,      
            dropout = 0.0,      
            emb_dropout = 0.0   
        )
    # ViT-Base-16
    elif model_name == "base":
        print('Creating ViT-Base model')
        model = ViT(
            image_size = 224,   
            patch_size = 16,    
            num_classes = 1000,  
            dim = 768,           
            depth = 12,         
            heads = 12,          
            mlp_dim = 3072,      
            channels = 3,       
            dim_head = 64,      
            dropout = 0.0,      
            emb_dropout = 0.0   
        )
    else: print(f"ERROR: The required model name '{model_name}' does not match any avaliable model")
    return model

def build_submodel(model_name, submodel_name):
    # ViT-Tiny-16
    if model_name == "tiny":
        if submodel_name == "attention":
            submodel = Attention(
                dim = 192,
                heads = 3, 
                dim_head = 64, 
                dropout = 0.0
            )
        elif submodel_name == "embedding":
            submodel = PatchEmbedding(
                image_size = 224,   
                patch_size = 16,    
                dim = 192,  
                channels = 3
            )
        elif submodel_name == "transformer":
            submodel = TransformerBlock(
                dim = 192,   
                heads = 3,
                dim_head = 64, 
                mlp_dim = 768,   
                dropout = 0.0
            )
        elif submodel_name == "classifier":
            submodel = ClassificationHead(
                dim = 192, 
                num_classes = 1000
            )
        else: print(f"ERROR: The required submodel name '{submodel_name}' does not match any avaliable submodel for the model '{model_name}'")
    # ViT-Small-16
    elif model_name == "small":
        if submodel_name == "attention":
            submodel = Attention(
                dim = 384,
                heads = 6, 
                dim_head = 64, 
                dropout = 0.0
            )
        elif submodel_name == "embedding":
            submodel = PatchEmbedding(
                image_size = 224,   
                patch_size = 16,    
                dim = 384,  
                channels = 3
            )
        elif submodel_name == "transformer":
            submodel = TransformerBlock(
                dim = 384,      
                heads = 6,
                dim_head = 64, 
                mlp_dim = 1536,   
                dropout = 0.0
            )
        elif submodel_name == "classifier":
            submodel = ClassificationHead(
                dim = 384, 
                num_classes = 1000
            )
        else: print(f"ERROR: The required submodel name '{submodel_name}' does not match any avaliable submodel for the model '{model_name}'")
    # ViT-Base-16
    elif model_name == "base":
        if submodel_name == "attention":
            submodel = Attention(
                dim = 768,
                heads = 12, 
                dim_head = 64, 
                dropout = 0.0
            )
        elif submodel_name == "embedding":
            submodel = PatchEmbedding(
                image_size = 224,   
                patch_size = 16,    
                dim = 768,  
                channels = 3
            )
        elif submodel_name == "transformer":
            submodel = TransformerBlock(
                dim = 768,      
                heads = 12,
                dim_head = 64, 
                mlp_dim = 3072,   
                dropout = 0.0
            )
        elif submodel_name == "classifier":
            submodel = ClassificationHead(
                dim = 768, 
                num_classes = 1000
            )
        else: print(f"ERROR: The required submodel name '{submodel_name}' does not match any avaliable submodel for the model '{model_name}'")

    else: print(f"ERROR: The required model name '{model_name}' does not match any avaliable model")
    return submodel

def build_dataloader_train_cifar10(batch_size):

    # normalization cifar10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    dataset_class = torchvision.datasets.CIFAR10

    # train transformations cifar10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])

    # Prepare dataset
    print("Loading train dataset")
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return trainloader

def build_dataloader_validation_cifar10(batch_size):

    # normalization cifar10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    dataset_class = torchvision.datasets.CIFAR10

    # validation transformations cifar10
    transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])

    # Prepare dataset
    print("Loading validation dataset")
    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return testloader

def build_dataloader_train_imagenet1k(batch_size, imagenet_root):
    
    # ImageNet-1K normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("Loading ImageNet-1K train dataset")

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(imagenet_root, 'train'),
        transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    return trainloader

def build_dataloader_validation_imagenet1k(batch_size, imagenet_root):

    # ImageNet-1K normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("Loading ImageNet-1K validation dataset")

    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(imagenet_root, 'val'),
        transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return testloader

def print_module_list(model):
    # Print model's module list (name and class)
    for i, (name, module) in enumerate(model.named_modules()):
        print(f"[{i:03d}] {name:40s} {module.__class__.__name__}")

def create_directory(path):
    # Create directory if doesnt exist
    os.makedirs(path, exist_ok=True)