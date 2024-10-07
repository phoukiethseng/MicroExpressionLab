import torch

class MicroExpressionCNN(torch.nn.Module):
    def __init__(self, exp_class_size, exp_state_size):
        super(MicroExpressionCNN, self).__init__()

        self.exp_class_size = exp_class_size
        self.exp_state_size = exp_state_size

        # Take in 64 x 64 RGB image
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=512), # infer input feature from output of convolution layer
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.ReLU(),
        )

        self.expression_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=self.exp_class_size),
            torch.nn.Softmax(),
        )
        self.expression_state_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=self.exp_state_size),
            torch.nn.Softmax(),
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        exp_class = self.expression_classifier(x) # Predicted probability of expression class
        exp_state = self.expression_state_classifier(x) # Predicted probability of expression state

        return x, exp_class, exp_state

def test_MicroExpressionCNN():
    model = MicroExpressionCNN()
    x = torch.rand((2, 3, 64, 64))
    feat, exp_class, exp_state = model(x)
    print(f'\nShape: feat = {feat.shape}, exp_class = {exp_class.shape}, exp_state = {exp_state.shape}')
    print(f'Output: {exp_class}, {exp_state}')
