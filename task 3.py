import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess
imsize = 256
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load content and style images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

assert content_img.shape == style_img.shape, "Images must be the same size"

# Initialize generated image from content (NOT noise!)
input_img = content_img.clone().requires_grad_(True)

# Normalization for VGG
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Define Normalization layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Content and style loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Build model
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

def get_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses, style_losses = [], []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

model, style_losses, content_losses = get_model_and_losses(
    cnn, normalization_mean, normalization_std, style_img, content_img
)

# Weights
style_weight = 1e5
content_weight = 1e0

# Optimizer
optimizer = optim.LBFGS([input_img])

print("Optimizing...")

run = [0]
while run[0] <= 300:

    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)

        loss = style_weight * style_score + content_weight * content_score
        if torch.isnan(loss):
            raise ValueError("Loss became NaN")

        loss.backward()
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Style: {style_score.item():.2f}, Content: {content_score.item():.2f}")
        return loss

    optimizer.step(closure)

# Final clamp and save
with torch.no_grad():
    input_img.clamp_(0, 1)

# Save result
unloader = transforms.ToPILImage()
image = input_img.cpu().clone().squeeze(0)
image = unloader(image)
image.save("output.png")
print("Saved as output.png")
