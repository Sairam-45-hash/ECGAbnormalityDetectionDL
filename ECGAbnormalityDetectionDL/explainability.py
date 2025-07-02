
import torch
import matplotlib.pyplot as plt
import numpy as np

def extract_attention_weights(model, input_tensor):
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor.unsqueeze(0))
        attention_layer = model.attention
        with torch.no_grad():
            combined_features = model.attention[0](model.last_concat)
            att_weights = model.attention[-1](combined_features)
    return att_weights.squeeze().cpu().numpy()

def compute_gradcam(model, input_tensor, target_class):
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    input_tensor.requires_grad = True

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    model.tcn[-1].conv.register_forward_hook(forward_hook)
    model.tcn[-1].conv.register_backward_hook(backward_hook)

    output = model(input_tensor.squeeze(1))
    class_score = output[0, target_class]
    class_score.backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=2, keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / torch.max(cam)

    return cam.detach().cpu().numpy()

def plot_ecg_with_attention(ecg_segment, attention_map, title='Attention Map'):
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_segment, label='ECG Signal')
    plt.fill_between(np.arange(len(ecg_segment)), 0, attention_map * np.max(ecg_segment), alpha=0.4, color='red', label='Attention')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
