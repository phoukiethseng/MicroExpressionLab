import os
from torchvision import transforms
from torchvision.io import read_image

from model.cnn import MicroExpressionCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataset_get(filename):
    prefix ="F:\\CASME2\\Cropped"
    return os.path.join(prefix, filename)

def main():
    model = MicroExpressionCNN(exp_class_size=7, exp_state_size=5)
    model.load_state_dict(torch.load("D:\\CASME2\\2024-10-14_15-55.pt", weights_only=True))
    model.to(device)
    model.eval()

    transformer = transforms.Resize((64, 64))

    X = transformer(read_image(dataset_get("sub09\\EP06_01f\\reg_img95.jpg")))
    X = X.to(device).float()
    X = X[torch.newaxis, :, :, :]
    feature, pred_class, pred_state = model(X)

    print(f'feature: {feature}\npred_class: {pred_class}\npred_state: {pred_state}')

if __name__ == '__main__':
    main()
