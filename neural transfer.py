import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import os

# Set file paths
content_path = r"C:\Users\Mohammed Haris\OneDrive\Desktop\progidy\Neural style transfer\sketch2.jpg"
style_path = r"C:\Users\Mohammed Haris\OneDrive\Desktop\progidy\Neural style transfer\angel1.jpg"

# Load and transform images
def load_image(image_path, transform, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images
content_img = load_image(content_path, transform)
style_img = load_image(style_path, transform, shape=content_img.shape[-2:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_img = content_img.to(device)
style_img = style_img.to(device)

# Load VGG19
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

def get_features(image, model):
    layers = {
        '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
        '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style loss
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (d * h * w)

# Extract features
content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Generated image
generated = content_img.clone().requires_grad_(True).to(device)

# **EXTREME STYLE TRANSFER TUNING**
style_weights = {
    'conv1_1': 1.0, 'conv2_1': 0.9, 'conv3_1': 0.8,
    'conv4_1': 0.5, 'conv5_1': 0.3
}
content_weight = 1e2
style_weight = 2e8  # Increased drastically

# Optimizer
optimizer = optim.Adam([generated], lr=0.04)  # Higher learning rate
criterion = nn.MSELoss()

# Convert tensor to image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

def save_image(tensor, filename="output.png"):
    image = tensor_to_image(tensor)
    image.save(filename)
    print(f"Image saved as {filename}")

# Training loop
print("Training started...\n")
for epoch in range(1000):  # More epochs
    optimizer.zero_grad()
    gen_features = get_features(generated, vgg)

    # Content loss
    content_loss = criterion(gen_features['conv4_2'], content_features['conv4_2'])

    # Style loss
    style_loss = 0
    for layer in style_weights:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * criterion(gen_gram, style_gram)
        style_loss += layer_loss

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/1000 - Loss: {total_loss.item():.4f}")

# Save the final image
save_image(generated)