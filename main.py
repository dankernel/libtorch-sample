
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Get pretrained resnet18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 256, 256)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Inference test
output = traced_script_module(torch.ones(1, 3, 256, 256))
print(output.shape)

# Save model to .pt file
traced_script_module.save("traced_resnet_model.pt")

# XXX : Do `wget https://github.com/pytorch/hub/raw/master/images/dog.jpg`
# Input tensor preparation
input_image = Image.open("./dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Shape check
print(input_batch.shape)

# Inference
output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# XXX : Do `wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt`
# Result output
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

