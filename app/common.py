import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
 
def get_model():
	PATH='static/checkpoint-3.pth'
	model=models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                             nn.ReLU(),
                             nn.Dropout(0.25),
                             nn.Linear(512, 102),
                             nn.LogSoftmax(dim=1))  
    model.load_state_dict(torch.load(PATH,map_location='cpu'),strict=False)
    model.eval()
    return model


def get_tensor(image_bytes):
	test_transforms = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
	image=Image.open(io.BytesIO(image_bytes))
	return test_transforms(image)


	



