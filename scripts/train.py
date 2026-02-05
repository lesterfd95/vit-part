import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import utility

# Este script entrena un modelo con CIFAR10 y guarda un .pth incluyendo el model_dict con pesos
# recibe nombre del modelo, numero de batches y epochs 
# genera un archivo .pth 

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model to train. Ej: tiny, small", required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    
    args = parser.parse_args()
    return args

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # hyper params
    batch_size = args.batch_size
    learning_rate = 1e-4
    n_epochs = args.epochs
    print(f"\nbatch_size {batch_size}\n"
        f"learning_rate: {learning_rate}\n"
        f"n_epochs:   {n_epochs}\n")

    # model
    model = utility.build_model(args.model)

    # dataloaders
    trainloader = utility.build_dataloader_train(batch_size)
    testloader = utility.build_dataloader_validation(batch_size)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    model.to(device)

    max_acc = 0

    print(f"Entrenando")
    for epoch in range(n_epochs):
        epoch_start_time = time.perf_counter()
        
        # TRAIN
        model.train()
        train_start_time = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            running_total += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / (batch_idx + 1)
            train_acc = 100. * running_correct / running_total

            print(
                f"[TRAIN] Epoch [{epoch+1}/{n_epochs}] "
                f"Batch [{batch_idx+1}/{len(trainloader)}] "
                f"Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%"
            )

        train_time = time.perf_counter() - train_start_time
        scheduler.step()

        # VALIDATION
        model.eval()
        val_start_time = time.perf_counter()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                current_val_loss = val_loss / (batch_idx + 1)
                current_val_acc = 100. * val_correct / val_total

                print(
                    f"[VAL]   Epoch [{epoch+1}/{n_epochs}] "
                    f"Batch [{batch_idx+1}/{len(testloader)}] "
                    f"Loss: {current_val_loss:.4f} | Acc: {current_val_acc:.2f}%"
                )


        val_time = time.perf_counter() - val_start_time
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"\n Epoch {epoch+1}/{n_epochs} SUMMARY\n"
            f"Train time: {train_time:.2f}s\n"
            f"Val time:   {val_time:.2f}s\n"
            f"Total:      {epoch_time:.2f}s\n")
        print("-" * 80)

        if current_val_acc > max_acc:
            max_acc = current_val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            utility.create_directory("./trained_models")
            torch.save(checkpoint, f"./trained_models/vit_{args.model}_cifar10.pth")
            print(f"Modelo guardado en ./trained_models/vit_{args.model}_cifar10.pth")

if __name__ == '__main__':
    
    args = parse_args()
    main(args)