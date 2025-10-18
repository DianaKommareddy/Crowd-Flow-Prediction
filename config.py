# config.py

class Config:
    # Paths
    train_data_path = "dataset/"
    model_save_path = "hierarchical_cache_attention_model.pth"  # changed name

    # Training Parameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4

    # Image size
    img_height = 256
    img_width = 256

    # Model Parameters
    in_channels = 3  # A, E, G channels
    out_channels = 1  # grayscale flow

    # Device
    device = "cuda"  # or "cpu"
