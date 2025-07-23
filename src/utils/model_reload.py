import torch
import os

# here we should be creating the tglf-SiNN model
def load_prev_model(model, checkpoint_path):
    """
    Restore a previous model from saved weights.
    
    Args:
        cfg: The same configuration object used for training
        checkpoint_path: Path to the saved .pt or .pth file
    
    Returns:
        model: Loaded model with weights
    """
    
    # 1. Sanity check
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2. Load the raw state_dict
    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    # 3. Make sure what you loaded is indeed a dict of tensors
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a dict of parameters, but got {type(state_dict)}")

    # 4. Load into the model
    model.load_state_dict(state_dict)

    # 5. Switch to eval mode
    model.eval()

    return model