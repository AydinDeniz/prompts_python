
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load Pretrained VGG19 Model for Style Transfer
class VGG19Features(nn.Module):
    def __init__(self):
        super(VGG19Features, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layer_mapping = {'1': 'conv1', '6': 'conv2', '11': 'conv3', '20': 'conv4', '29': 'conv5'}
        self.layers = nn.ModuleList([vgg[i] for i in range(30)])  # Extract up to 30 layers

    def forward(self, x):
        features = {}
        for name, layer in enumerate(self.layers):
            x = layer(x)
            if str(name) in self.layer_mapping:
                features[self.layer_mapping[str(name)]] = x
        return features

# Compute Content Loss
def content_loss(generated_features, content_features):
    return torch.mean((generated_features - content_features) ** 2)

# Compute Style Loss
def style_loss(generated_features, style_features):
    loss = 0
    for layer in generated_features:
        G = torch.mm(generated_features[layer].view(-1, generated_features[layer].shape[1]).t(), 
                     generated_features[layer].view(-1, generated_features[layer].shape[1]))
        A = torch.mm(style_features[layer].view(-1, style_features[layer].shape[1]).t(), 
                     style_features[layer].view(-1, style_features[layer].shape[1]))
        loss += torch.mean((G - A) ** 2)
    return loss

# Load 3D Model as Point Cloud
def load_3d_model(file_path):
    print("Loading 3D model...")
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# Apply Style Transfer to 3D Model
def apply_style_transfer(content_path, style_path, output_path, epochs=300, content_weight=1, style_weight=1e6):
    print("Applying neural style transfer...")
    content_image = transforms.ToTensor()(plt.imread(content_path)).unsqueeze(0)
    style_image = transforms.ToTensor()(plt.imread(style_path)).unsqueeze(0)

    model = VGG19Features()
    model.eval()

    content_features = model(content_image)
    style_features = model(style_image)

    generated = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=0.02)

    for epoch in range(epochs):
        optimizer.zero_grad()
        generated_features = model(generated)
        loss = content_weight * content_loss(generated_features, content_features) +                style_weight * style_loss(generated_features, style_features)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    plt.imsave(output_path, transforms.ToPILImage()(generated.squeeze(0)))
    print(f"Styled image saved as {output_path}")

# Apply Texture to 3D Model
def apply_texture_to_3d_model(pcd, texture_path):
    print("Applying texture to 3D model...")
    texture = plt.imread(texture_path)
    colors = np.array(texture[:, :, :3]) / 255.0  # Normalize RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors[:len(pcd.points)])  # Assign colors to points
    return pcd

# Save 3D Model with Style Transfer
def save_3d_model(pcd, output_path):
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Styled 3D model saved as {output_path}")

if __name__ == "__main__":
    content_3d_model = "input_model.obj"
    content_image = "content.jpg"
    style_image = "style.jpg"
    styled_image_output = "styled_texture.jpg"
    styled_model_output = "styled_model.obj"

    content_pcd = load_3d_model(content_3d_model)
    apply_style_transfer(content_image, style_image, styled_image_output)
    styled_pcd = apply_texture_to_3d_model(content_pcd, styled_image_output)
    save_3d_model(styled_pcd, styled_model_output)
