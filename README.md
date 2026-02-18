# Crowd Flow Prediction
This project uses a transformer-based model to predict how crowds move in busy scenes. It takes in the positions of agents, the layout of the environment—including walkable and non-walkable areas—and the agents’ goals. Based on this information, the model learns to estimate the likely movement patterns and generates a flow map of the crowd.

# Pretrained Model Checkpoints

If you don’t want to train from scratch, you can download the pretrained model weights here:

## Pretrained Model Checkpoints

If you don’t want to train from scratch, you can download the pretrained model weights here:

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
This model uses a hierarchical transformer design to predict crowd flow. It combines global and local context information, allowing better learning of patterns and interactions within complex crowd scenes.

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

---

## References

This project builds upon the following foundational works:

**Restormer: Efficient Transformer for High-Resolution Image Restoration**
Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao
[arXiv | GitHub](https://arxiv.org/abs/2111.09881 | https://github.com/swz30/Restormer)

**Laying the Foundations of Deep Long-Term Crowd Flow Prediction**
Seonguk Seo, Seunghoon Hong, Bohyung Han
[Springer Link (ECCV 2020) | GitHub](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_42 | https://github.com/SSSohn/LTCF)



