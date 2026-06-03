# UpTCR: A Progressive Knowledge Transfer Framework with Integrating Any Incomplete Data for TCR-Antigen-HLA Binding Recognition

This repository contains the source code for the paper: A Progressive Knowledge Transfer Framework with Integrating Any Incomplete Data for TCR-Antigen-HLA Binding Recognition.

![image](./pipeline.png)

UpTCR is a progressive knowledge transfer framework that learn priors from any incomplete data for diverse downstream tasks.

## Installation and Setup
1. **Hardware requirements**:
   This project requires only a standard computer with enough RAM and a NVIDIA GPU to support operations. We ran the demo using the following specs:
   - CPU: 10 cores, 2.5 GHz/core
   - RAM: 40GB
   - GPU: NVIDIA TESLA P40, V100, A100, A6000
   - CUDA: 11.0, 12.0

2. **System requirements**:
   This tool is supported for Linux. The tool has been tested on the following system:

   - CentOS Linux release 8.2.2.2004
   - Ubuntu 22.04
   - Ubuntu 24.04

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/TencentAILabHealthcare/PEPAF.git
   cd UpTCR
   ```

4. **Install Required Packages**:
   The basic environment requirements are:
   - Python: 3.10/3.11
   - CUDA: 11.0/12.0

   Use the following command to install the necessary packages as specified in the `requirements.txt` file:

   ```bash
   conda create -n UpTCR python==3.10
   conda activate UpTCR
   pip install -r requirements.txt
   ```

5. **Download Model Weights**:

   Download the `model_weights.zip` file and extract it to the `UpTCR/model_weights` directory. The model_weights.zip is available on Zenodo: <https://doi.org/10.5281/zenodo.20520000>

   After extraction, the `UpTCR/model_weights` directory should contain the following:

   ```plaintext
   UpTCR/model_weights/
   ├── pretrained/
   └── finetune
   ```
   The `UpTCR/model_weights/pretrained` directory contains the pretrained individual encoders and fusion encoders.

   The `UpTCR/model_weights/finetune` directory should contain the following:
    ```plaintext
   UpTCR/model_weights/finetune/
   ├── TCRABpMHC_fewshot/
   ├── TCRABpMHC_unseen/
   ├── TCRABp_fewshot/
   ├── TCRABp_unseen/
   ├── TCRBpMHC_fewshot/
   ├── TCRBpMHC_unseen/
   ├── TCRBp_fewshot/
   └── TCRBp_unseen/
   ```
   Each sub-directory contains the fine-tuned models for direct test.

6. **Download Processed Data**:

   The processed data (~41 GiB) includes TCR-antigen-HLA binding data, structure data, and pretrained embeddings. We provide two download options:

   **Option A: Single archive (for stable network connections)**

   Download `data.zip` from Zenodo: <https://doi.org/10.5281/zenodo.15128399>, and extract it to the `UpTCR/data` directory:
   ```bash
   unzip data.zip -d ./data
   ```

   **Option B: Split volumes (for unreliable networks with resume support)**

   To prevent network-induced transmission failures, we have also partitioned the dataset into 4 volumes (`data.7z.001` to `data.7z.004`, 10 GiB each), hosted at Hugging Face: <https://huggingface.co/datasets/DDDead/Uptcr_data>, for high-speed access.

   Ensure the `p7zip` package is installed on your system. Navigate to your `UpTCR` root directory, move all 4 downloaded files into the `UpTCR/` folder, and run the following command on the **first** volume (the program will automatically detect and merge the remaining parts):
   ```bash
   7z x data.7z.001 -o./data
   ```

   **Directory structure after extraction:**

   ```plaintext
   UpTCR/data/
   ├── finetune/        # Processed TCR-antigen-HLA binding data for training
   ├── structure/       # Processed TCR-antigen-HLA structure data
   └── pretrained_emb/  # Pretrained embeddings
   ```

   > **Note:** The `pretrained_emb` directory is large, consisting of 41,599 antigen embeddings, 27,066 TCRa embeddings, and 27,946 TCRb embeddings. Decompression may take a considerable amount of time.

## Quick Start for prediction
Here we provide diverse settings for result reproduction. Please ensure the model weights (**finetune**) and data have been properly added. Because our UpTCR is able to predict for complete or modality-missing settings, the following scripts for different settings are provided.

1. **Prediction considering all TCRa, TCRb, Antigen, HLA**
    
    Few-shot prediction:
    ```bash
    # few-shot prediction with tetrameric interaction
    python scripts/test/test_TCRABpMHC_fewshot.py
    ```
    Zero-shot prediction:
    ```bash
    # zero-shot predictin with tetrameric interaction interaction
    python scripts/test/test_TCRABpMHC_zeroshot.py
    ```
2. **Prediction only considering TCRb, Antigen, HLA**
    
    Few-shot prediction:
    ```bash
    # few-shot prediction with only TCRb, Antigen, HLA
    python scripts/missing_test/test_TCRBpMHC_fewshot.py
    ```
    Zero-shot prediction:
    ```bash
    # zero-shot prediction with only TCRb, Antigen, HLA
    python scripts/missing_test/test_TCRBpMHC_zeroshot.py
    ```
3. **Prediction only considering TCRa, TCRb, Antigen**
    
    Few-shot prediction:
    ```bash
    # few-shot prediction with only TCRa, TCRb, Antigen
    python scripts/missing_test/test_TCRABp_fewshot.py
    ```
    Zero-shot prediction:
    ```bash
    # zero-shot prediction with only TCRa, TCRb, Antigen
    python scripts/missing_test/test_TCRABp_zeroshot.py
    ```
4. **Prediction only considering TCRb, Antigen**
    
    Few-shot prediction:
    ```bash
    # few-shot prediction with only TCRb, Antigen
    python scripts/missing_test/test_TCRBp_fewshot.py
    ```
    Zero-shot prediction:
    ```bash
    # zero-shot prediction with only TCRb, Antigen
    python scripts/missing_test/test_TCRBp_zeroshot.py
    ```

## Quick Start for training, validation, and testing
Here we provide diverse settings for model training, validation, and testing. Please ensure the model weights (**pretrained**) and data have been added. Our UpTCR is able to predict for complete interaction, modality-missing interaction, Antigen-HLA binding affinity prediction, and contact map prediction.

1. **Binding specificity considering all TCRa, TCRb, Antigen, HLA**
    
    Few-shot learning:
    ```bash
    # few-shot learning with tetrameric interaction
    python scripts/train/train_TCRABpMHC_fewshot.py
    ```
    Zero-shot learning:
    ```bash
    # zero-shot learning with tetrameric interaction
    python scripts/train/train_TCRABpMHC_zeroshot.py
    ```


2. **Binding specificity considering only TCRb, Antigen, HLA**
    
    Few-shot learning:
    ```bash
    # few-shot learning with only TCRb, Antigen, HLA
    python scripts/missing_train/train_TCRBpMHC_fewshot.py
    ```
    Zero-shot learning:
    ```bash
    # zero-shot learning with only TCRb, Antigen, HLA
    python scripts/missing_train/train_TCRBpMHC_zeroshot.py
    ```
3. **Binding specificity considering only TCRa, TCRb, Antigen**
    
    Few-shot learning:
    ```bash
    # few-shot learning with only TCRa, TCRb, Antigen
    python scripts/missing_train/train_TCRABp_fewshot.py
    ```
    Zero-shot learning:
    ```bash
    # zero-shot learning with only TCRa, TCRb, Antigen
    python scripts/missing_train/train_TCRABp_zeroshot.py
    ```
4. **Binding specificity considering only TCRb, Antigen**
    
    Few-shot learning:
    ```bash
    # few-shot learning with only TCRb, Antigen
    python scripts/missing_train/train_TCRBp_fewshot.py
    ```
    Zero-shot learning:
    ```bash
    # zero-shot learning with only only TCRb, Antigen
    python scripts/missing_train/train_TCRBp_zeroshot.py
    ```
5. **Learning antigen-HLA binding affinity**
    
    Run:
    ```bash
    # learning Antigen-HLA binding affinity
    python scripts/train/train_pMHC.py
    ```
6. **Learning contact maps**
    
    Run:
    ```bash
    # learning TCRa-TCRb-Antigen-HLA contact maps
    python scripts/train/train_structure.py
    ```

If you have any questions, please contact us at lvtianxu1@icloud.com.