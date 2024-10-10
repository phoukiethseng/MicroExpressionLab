from dataset import MicroExpressionDataset
import torch
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, EPOCH, EXP_STATE_SIZE, EXP_CLASS_SIZE, LEARNING_RATE
from loss import loss_function, half_min_distance
from model import MicroExpressionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    cuda = torch.cuda.current_device()
    if (torch.cuda.is_available()):
        torch.set_default_device(cuda)
    dataset = MicroExpressionDataset("D:\\CASME2\\label.csv", "D:\\CASME2")
    train_dataset, test_dataset = random_split(dataset, [0.85, 0.15], generator=torch.Generator(device=cuda))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=cuda))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=cuda))

    model = MicroExpressionCNN(EXP_CLASS_SIZE, EXP_STATE_SIZE)
    model.train()

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}\n_____________________________________")
        # TODO: Calculate Spatial Feature mean and Half Minimum Distance every epoch instead of every batch
        train_loop(train_dataloader, model)

def train_loop(train_dataloader, model):
    size = len(train_dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    for batch, (X, y_class, y_state) in enumerate(train_dataloader):

        # Move to GPU
        X = X.to(device)
        y_class = y_class.to(device)
        y_state = y_state.to(device)

        # Compute expression class and state prediction along with spatial feature vector
        feat, pred_class, pred_state = model(X.float())
        # TODO: Figure out why predicted predictive probability got pushed to [1.00 1.00 ... 1.00] as we trained the model

        # Using y_class, we know which ground truth class that each sample belong to
        class_feat_means = torch.zeros((EXP_CLASS_SIZE, 512), device=device)
        for k in range(EXP_CLASS_SIZE):
            class_k_index = y_class[:, k]
            class_k_feat_mean = feat.index_select(dim=0, index=class_k_index).mean(dim=0)
            class_feat_means[k] = class_k_feat_mean

        hmd = half_min_distance(class_feat_means)

        # Calculate loss
        loss = loss_function(y_class, y_state, pred_class, pred_state, class_feat_means, feat, hmd)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # for p in model.parameters():
        #     print(p.grad.norm())

        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == '__main__':
    main()
