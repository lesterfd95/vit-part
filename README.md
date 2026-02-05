# ViT Attention Subnetwork Experiment

## Project Overview

This repository contains part of the resources used to prepare an experiment in which inference is performed with a neural network composed only of the layers that form the first attention module/block of a ViT-base model trained on CIFAR10.

In the documentation, this neural network may be referred to in several ways, such as **subnetwork**, **submodel**, **partial network**, or **partial model**. Additionally, all of the following terms refer to the same structure: **attention module**, **attention block**, and **attention mechanism**.

The results of the experiment that used these resources can be found at:  
`--insert link--`

---

## Setup 

### Clone the Repository

Clone the project repository into your workspace:

```bash
git clone https://github.com/lesterfd95/vit-part
cd vit-part
```
### 🐳 Docker Instructions

1.  **Install Docker**

  Consult the official website and documentation for installation steps:  
  
  * [Docker Homepage](https://www.docker.com/)  
  * [Docker Documentation](https://docs.docker.com/)

2.  **Create the Docker Image**

  Execute the following command from the project root:  
  
  ```bash
  docker build -t vit_ti:1.0 .
  ```  
  The resulting image, named `vit_ti:1.0`, will contain all necessary dependencies extracted from `requirements.txt`.  
  *Note: The image does not include the `scripts` folder or its contents.*

3.  **Instantiate a Docker Container**

  Run the setup script:  
  
  ```bash
  ./create_container.sh
  ```  
  This will instantiate a container from the image and open a terminal inside the `/app` directory. This directory is linked to your local `scripts` folder, allowing you to use and edit content in real-time without bloating the image size with datasets or heavy files.

4.  **Enable GPU Resources**
   
  To use GPU resources inside the container, you may need the **NVIDIA Container Toolkit**:  
    
  * [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


> [!CAUTION]  
> **Compatibility Note:** Conflicts can occur between the PyTorch version in the container and the CUDA version supported by your GPU drivers. Use the `nvidia-smi` tool to check your driver and CUDA versions, then verify compatibility at [pytorch.org](https://pytorch.org/).

### 🐍 Local Setup (Without Docker)

Ensure you have **Python** installed along with the dependencies listed in `requirements.txt`. All commands and development should be performed from within the `scripts` directory:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## 🛠 Script Descriptions

| Script | Description |
| :--- | :--- |
| **vit.py** | Contains the `ViT` class and component classes used to build the model. Original source: [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py). |
| **utility.py** | Common functions used across multiple scripts, such as building a ViT model from `vit.py` or creating data loaders for training and evaluation. `build_model` can be extended for more ViT versions, and `build_submodel` can be adapted for other modules or module sets. |
| **train.py** | Trains a ViT model on CIFAR10. Arguments: model name, number of batches, and epochs. Saves a `.pth` file containing the `model_dict` with trained weights. |
| **test.py** | Performs inference using a trained ViT model on CIFAR10. Arguments: model name and batch size. Generates a "golden" file with output logits and creates a dataset for the first attention module by capturing inputs/outputs via hooks. |
| **model_subset.py** | Extracts a subset of weights from a trained `.pth` file corresponding specifically to the layers of the first attention module. Generates a `.pth` file for the partial network. |
| **confidence.py** | Uses CIFAR10 and a golden inference file to group correctly classified image indices into tiers based on classification confidence. Generates a list of lists representing these confidence ranges. |
| **data_subset.py** | Extracts a subset from the sub-network dataset. Can be based on an index range or classification confidence (using pre-made lists). It accepts an array of sizes to define how many elements to include in each confidence group. |
| **attention.py** | Runs inference on the trained sub-network using the prepared dataset to verify expected outputs. Includes functionality to inject errors into consecutive images or batches for testing purposes. |
---

## 📖 Workflow Example

This guide demonstrates how to prepare a sub-network consisting only of the **first attention module** of a **ViT-base** model. To perform inference with this sub-network, we will generate a dataset containing the input and output values of that specific module, captured during a full ViT inference on CIFAR10. Furthermore, we will organize this data into subsets based on classification confidence levels.

### Step 1

Use `train.py` to train the **ViT-base** model:

```bash
python3 train.py --model base --batch_size 100 --epochs 100
```
The execution will download the CIFAR10 training and validation subsets if they do not exist in the working directory and save them in the `data` directory. It will perform training for the specified number of epochs and batch size. Other hyperparameters used are a **learning rate of 1e-4**, **Cross-Entropy loss**, and the **Adam optimizer**; these or other training parameters can be modified by editing the `train.py` code.

For each training epoch, a validation step will be performed, and the weights that achieve the best accuracy will be retained. At the end of the process, a file named `vit_base_cifar10.pth` will be saved in the `trained_models` directory. The feasibility and duration of the training depend on the selected hyperparameters and the hardware used for execution.

> [!TIP]
> If you are using Docker on a remote server, the `launch_detached.sh` script can be used to automatically trigger the Docker container build and start the training process. This script also allows the process to continue running after you disconnect from the server. 
> When used, a `full_train.log` file is generated containing the execution results. You can edit the script directly for further details and customization.

### Step 2

Use `test.py` to obtain a "golden" reference and a dataset for the subnetwork to be used:

```bash
python3 test.py --model base --batch_size 100
```
The execution will use the CIFAR10 validation dataset, build the ViT-base model, and load the trained weights from `/trained_models/vit_base_cifar10.pth`. It will then attach hooks to capture the inputs and outputs of the first attention module and perform inference on the data. This step allows you to verify if the network's accuracy matches the one reported during training.

Upon completion, the script will store the model outputs (logits) for all inputs in the `goldens` directory as `golden_base.pt`. Additionally, the inputs and outputs captured from the first attention module will be saved in the `module_datasets` directory under the name `vit_base_attn_dataset.pt`.

### Step 3

Use `model_subset.py` to extract only the trained weights corresponding to the subnetwork being used:

```bash
python3 model_subset.py --model base
```
The execution loads the trained weights file from `/trained_models/vit_base_cifar10.pth` and extracts the data corresponding to the layers that make up the first attention module. It then saves this weight subset in the `trained_models` directory under the name `vit_base_part_att.pth`.

### Step 2

Use `confidence.py` to separate the data based on classification confidence:

```bash
python3 confidence.py --model base
```

The execution loads the correct labels for each image from CIFAR10, then retrieves the output logits from `/goldens/golden_base.pt` to obtain the network's predictions. It compares the network's predictions with the ground truth labels and separates the image indices into correct and incorrect predictions.

Using the "golden" reference, the script also calculates the **classification confidence**. This is defined as the difference in probability between the two most likely classes at the network's output. At this stage, two files are generated in the `confidence` directory containing the image indices and their respective confidence scores:
* `base_cnf_corr_out.pt` (for correct classifications)
* `base_cnf_inc_out.pt` (for incorrect classifications)

The process continues using only the correctly classified data. These indices are split into four groups covering different confidence ranges: **0%-25%, 26%-50%, 51%-75%, and 76%-100%**. Finally, the lists representing these groups are saved as `base_cnf_corr_out_cat_4.pt` in the `confidence` directory.

> [!TIP]
> If you do not want the data used in the subnetwork inference to follow this specific distribution based on classification confidence, you can skip **Step 4**.

### Step 5

Use `data_subset.py` to generate the data subset to be used for the subnetwork inference:

```bash
python3 data_subset.py --model base --mode conf --group_sizes 64 64 128 256
```
The execution loads the dataset values created for our subnetwork from `/module_datasets/vit_base_attn_dataset.pt` and the confidence index lists from `/confidence/base_cnf_corr_out_cat_4.pt`. It then reorganizes the data to create a new dataset of 512 values: starting with 64 values from the 0%-25% confidence group, followed by 64 values from the 26%-50% group, and so on.

The values `64 64 128 256` were chosen to total 512 elements in the final dataset. The variation in group sizes is due to the fact that there are not enough elements in all confidence groups to make them equal; correctly classified predictions tend to have higher confidence more often than lower confidence. The new reduced and organized dataset is saved in `/module_datasets/vit_base_attn_dataset_cnf_4.pt`.

In case you do not want to organize the data by confidence and simply wish to obtain a subset of a specific size, you can use:

```bash
python3 data_subset.py --model base --mode range --range 0 512
```
This will result in a new dataset consisting of the first 512 elements (from 0 to 511) of the original dataset. It will be saved in `/module_datasets/vit_base_attn_dataset_0_512.pt`.

> [!TIP]
> Alternatively, you may use the entire dataset without any clipping if desired, which would eliminate the need for **Step 5**.

### Step 6

At this stage, we have all the necessary components to perform the inference with our subnetwork:

1.  **The Subnetwork Model:** Created using `build_submodel()` from `utility.py` (which utilizes `vit.py`).
2.  **Trained Weights for the Subnetwork:** Located at `/trained_models/vit_base_part_att.pth`.
3.  **Input/Output Dataset:** Located at `/module_datasets/vit_base_attn_dataset_cnf_4.pt`.

Use `attention.py` to perform the inference and verify that the subnetwork behaves correctly (i.e., that the outputs match the expected values):

```bash
python3 attention.py --model base --dataset vit_base_attn_dataset_cnf_4.pt --atol 5.0e-6 --batch_size 64 --no-inject
```
The execution will build the subnetwork, load the weights from `/trained_models/vit_base_part_att.pth`, load the dataset from `/module_datasets/vit_base_attn_dataset_cnf_4.pt`, and perform the inference.

It will verify that no errors exist in each batch. Using the `--inject` argument in the command will inject specific fixed errors to test the comparison mechanism. Depending on the chosen tolerance values (`--atol` and `--rtol`), errors might be detected even without injection due to accumulated rounding errors and other numerical uncertainties during model execution.

Although this script is less complex than the software environment where the inference is deployed in the actual experiment, it serves as a tool to explore the generated resources, debug issues, fine-tune tolerance values, and more.
