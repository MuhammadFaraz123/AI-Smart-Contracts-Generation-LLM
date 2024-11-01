# 🧠 AI Smart Contract Generation LLM

This repository contains a **Large Language Model (LLM)** developed to generate smart contracts based on user-defined prompts. The LLM is trained on a comprehensive dataset covering the **crypto**, **blockchain**, and **Web3** industries, enabling it to generate accurate and industry-specific smart contract code. 

---

## 🚀 Key Features

- **Smart Contract Generation**: Generate Solidity-based smart contracts tailored to specific user prompts and contract requirements.
- **Industry-Specific Training**: Trained on a dataset rich in crypto, blockchain, and Web3 knowledge, making it particularly suited for the decentralized ecosystem.
- **End-to-End Pipeline**: The pipeline encompasses scraping, parsing, prompt generation, model training, and deployment for a complete, automated workflow.
- **Distributed Training**: Leveraged **CodeLlama-7b** and **DeepSeek-7b** models with distributed training for enhanced performance on large datasets.
- **Scalable Deployment**: Models deployed on **Google Cloud Platform (GCP)** for scalable and accessible use.

---

## 🚀 Getting Started
Clone the Repository:

```bash
git clone https://github.com/MuhammadFaraz123/AI-Smart-Contracts-Generation-LLM.git
cd AI-Smart-Contracts-Generation-LLM
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run the Inference Pipeline: To generate contracts, run:

```bash
python Inference.py
```

## 📊 Dataset
The training dataset is a collection of smart contracts and relevant documents from the blockchain and Web3 domains. The data includes various smart contract structures, enabling the model to generalize across different contract types.

## 📊 Model Training
The model training is performed using distributed computing on CodeLlama-7b, CodeGen and DeepSeek-7b with full parameter tuning.
The training pipeline leverages different VM resources to scale based on dataset size, enabling efficient training for large-scale datasets.

## 🛠 Configuration
Model and training configurations can be adjusted. Key settings include:

- **Data parameters**: Specify paths and data augmentation options.
- **Training parameters**: Set batch size, learning rate, and other model parameters.



