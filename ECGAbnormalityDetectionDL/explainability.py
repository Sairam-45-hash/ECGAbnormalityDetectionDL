import torch
import matplotlib.pyplot as plt
import numpy as np

def extract_attention_weights(model, input_tensor):
    model.eval()
    with torch.no_grad():
        # Ensure input_tensor shape is [batch, 1, seq_len] for conv1d
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # [1, seq_len]
        if input_tensor.dim() == 2 and input_tensor.shape[1] != 1:
            input_tensor = input_tensor.unsqueeze(1)  # [batch, 1, seq_len]

        # Forward pass to set model.last_concat
        _ = model(input_tensor)

        # Access combined features saved during forward pass
        combined_features = model.last_concat  # [batch, feature_size]

        # Apply attention layers to combined features
        attention_layer = model.attention
        att_weights = attention_layer(combined_features)  # [batch, 1]

    return att_weights.squeeze().cpu().numpy()

def compute_gradcam(model, input_tensor, target_class):
    # Prepare input tensor for conv1d and GRU: [batch, 1, seq_len]
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len]
    elif input_tensor.dim() == 2 and input_tensor.shape[1] != 1:
        input_tensor = input_tensor.unsqueeze(1)

    input_tensor.requires_grad = True

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on last TCN convolutional layer
    model.tcn.network[-1].conv1.register_forward_hook(forward_hook)
    model.tcn.network[-1].conv1.register_backward_hook(backward_hook)

    output = model(input_tensor.squeeze(1))  # squeeze channel dim for GRU input
    class_score = output[0, target_class]
    class_score.backward()

    grad = gradients[0]  # [batch, channels, length]
    act = activations[0]

    weights = grad.mean(dim=2, keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / torch.max(cam)

    return cam.detach().cpu().numpy()

def plot_ecg_with_attention(ecg_segment, attention_map, title='Attention Map'):
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_segment, label='ECG Signal')
    plt.fill_between(np.arange(len(ecg_segment)), 0, attention_map * np.max(ecg_segment), 
                     alpha=0.4, color='red', label='Attention')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
