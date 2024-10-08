from dataset import MicroExpressionDataset
import torch
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, EPOCH, EXP_STATE_SIZE, EXP_CLASS_SIZE, LEARNING_RATE
from loss import loss_function, half_min_distance
from model import MicroExpressionCNN


def main():
    dataset = MicroExpressionDataset("D:\\CASME2\\label.csv", "D:\\CASME2")
    train_dataset, test_dataset = random_split(dataset, [0.85, 0.15])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MicroExpressionCNN(EXP_CLASS_SIZE, EXP_STATE_SIZE)
    model.train()

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}\n_____________________________________")
        train_loop(train_dataloader, model)

def train_loop(train_dataloader, model):
    size = len(train_dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    for batch, (X, y_class, y_state) in enumerate(train_dataloader):
        # Compute prediction and loss
        feat, pred_class, pred_state = model(X.float())

        # Using y_class, we know which ground truth class that each sample belong to
        class_feat_means = torch.zeros((EXP_CLASS_SIZE, 512))
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
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == '__main__':
    main()
