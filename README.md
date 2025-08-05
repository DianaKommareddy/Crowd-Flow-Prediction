# Crowd Flow Prediction using Restormer
This project uses a transformer-based model called Restormer to predict how crowds move in busy scenes. It takes in the positions of agents, the layout of the environment—including walkable and non-walkable areas—and the agents’ goals. Based on this information, the model learns to estimate the likely movement patterns and generates a flow map of the crowd.

## Project Structure
<pre lang="md">
crowd-flow-prediction-using-restormer/
├── Test Dataset/            # test dataset
│   ├── A/                   # Agent maps
│   ├── E/                   # Scene layout
│   ├── G/                   # Goal maps
│   └── Y/                   # Ground truth
├── checkpoints/
│   └── restormer_best.pth   # Trained model weights
├── Dataset/                 # train dataset
│   ├── A/                   # Agent maps
│   ├── E/                   # Scene layout
│   ├── G/                   # Goal maps
│   └── Y/                   # Ground truth
├── models/
│   └── restormer_crowd_flow.py  # The model architecture
├── requirements.txt         # Python dependencies
├── train.py                 # Training script
├── test.py                  # Evaluation/prediction script
├── utils.py                 # Helper functions
└── README.md                # Project documentation
</pre>

## Model Overview

We use a modified **Restormer**, originally designed for image restoration, to handle crowd flow prediction. It captures spatial relationships and context using self-attention and feedforward blocks with upsampling and downsampling to handle multiscale reasoning.

**Inputs:**
- Agent Map (`A`) — where people are
- Environment Map (`E`) — obstacles, layout
- Goal Map (`G`) — destination/attraction zones

**Output:**
- Predicted flow map

---

## How to Run

### 1. Install Requirements

<pre lang="md">
pip install -r requirements.txt
</pre>

### 2. Train the Model

<pre lang="md">
python train.py --data_dir ./dataset --epochs 50 --batch_size 8
</pre>

This will train the Restormer-based model on your dataset using the provided Agent, Environment, and Goal maps. Checkpoints will be saved to the checkpoints/ directory by default.

### 3. Test the Model

<pre lang="md">
python test.py --model_path ./checkpoints/best_model.pt --data_dir ./dataset
</pre>

This will load the trained model and generate flow predictions. Output predictions will be saved to the predictions/ folder inside the working directory.



