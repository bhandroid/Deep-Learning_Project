import os
from flask import Flask, request, render_template, send_file
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn

# Flask app initialization
app = Flask(__name__)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generator Architecture (from your training code)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        return self.output(x)

# Load the trained generator model
model_path = "best_generator_A2B.pth"
best_generator_A2B = Generator().to(device)
best_generator_A2B.load_state_dict(torch.load(model_path, map_location=device))
best_generator_A2B.eval()

# Define transformations for input preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Preprocess the uploaded image
    img = Image.open(file).convert("RGB")
    input_image = transform(img).unsqueeze(0).to(device)

    # Generate synthetic image
    with torch.no_grad():
        synthetic_image = best_generator_A2B(input_image).squeeze(0).cpu()

    # Post-process the generated image
    synthetic_image = (synthetic_image * 0.5 + 0.5).clamp(0, 1)  # Denormalize to [0, 1]
    synthetic_image = transforms.ToPILImage()(synthetic_image)

    # Save the synthetic image
    output_path = "static/synthetic_image.png"
    synthetic_image.save(output_path)

    return render_template('result.html', synthetic_image=output_path)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

