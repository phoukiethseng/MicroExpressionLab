from sympy.stats.sampling.sample_numpy import numpy

from dataset import MicroExpressionDataset
import torch
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, EPOCH, EXP_STATE_SIZE, EXP_CLASS_SIZE, LEARNING_RATE
from loss import loss_function, half_min_distance
from model import MicroExpressionCNN
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
worker_size = 6

def main():
    cuda = torch.cuda.current_device()
    if (torch.cuda.is_available()):
        torch.set_default_device(cuda)
    dataset = MicroExpressionDataset("D:\\CASME2\\label.csv", "D:\\CASME2")
    train_dataset, test_dataset = random_split(dataset, [0.85, 0.15], generator=torch.Generator(device=cuda))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  generator=torch.Generator(device=cuda), num_workers=worker_size)

    epoch_dataloader_batch_size = 2000
    epoch_dataloader = DataLoader(train_dataset, batch_size=epoch_dataloader_batch_size, shuffle=False,
                                  generator=torch.Generator(device=cuda), num_workers=worker_size)

    model = MicroExpressionCNN(EXP_CLASS_SIZE, EXP_STATE_SIZE)
    model.train()

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}\n_____________________________________")
        # TODO: Calculate Spatial Feature mean and Half Minimum Distance every epoch instead of every batch
        print(
            'Calculating feature mean of Expression Class and State and Half-Minimum-Distance'
            'between Feature mean at the start of epoch...')
        class_feature_means, class_state_feature_means = get_exp_class_feature_mean(model, epoch_dataloader,
                                                                                    batch_size=epoch_dataloader_batch_size)
        hmd = half_min_distance(class_feature_means)
        print('Done')
        print(f'half minimum distance = {hmd}\nfeature mean = {class_feature_means}')
        train_loop(train_dataloader, model, class_feature_means, hmd, class_state_feature_means)

    # Save the model
    torch.save(model.state_dict(), f"D:\\CASME2\\{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pt")


def get_exp_class_feature_mean(model, dataloader, batch_size=128):
    class_feat_means = torch.zeros((EXP_CLASS_SIZE, 512), device=device, requires_grad=False)
    class_state_feat_means = torch.zeros((EXP_CLASS_SIZE, EXP_STATE_SIZE, 512), device=device, requires_grad=False)

    a = torch.arange(0, EXP_CLASS_SIZE, requires_grad=False, dtype=torch.float32, device=device)
    b = torch.arange(0, EXP_STATE_SIZE, requires_grad=False, dtype=torch.float32, device=device)

    dataset_size = len(dataloader.dataset)

    for batch, (X, y_class, y_state) in enumerate(dataloader):

        if batch * batch_size % 10000 == 0:
            print(f'Batch: {batch} [{batch * batch_size} / {dataset_size}]')

        X = X.to(device)
        y_class = y_class.to(device)
        y_state = y_state.to(device)

        feature, pred_class, pred_state = model(X.float())

        feature = feature.detach()  # Avoid computational graph build up, which will cause increasing memory consumption until out of memory error

        gt_exp_class_index = torch.matmul(y_class.float(), a)
        gt_exp_state_index = torch.matmul(y_state.float(), b)

        for k in range(EXP_CLASS_SIZE):
            class_k_index = gt_exp_class_index == k
            class_feat = feature[class_k_index, :]
            class_feat_size = class_feat.size()[0]
            class_feat_means[k] += class_feat.sum(dim=0) / dataset_size
            # TODO: calculate expression state feature mean
            for h in range(EXP_STATE_SIZE):
                state_h_index = torch.logical_and(class_k_index, gt_exp_state_index)
                class_state_feat = feature[state_h_index, :]
                class_state_feat_size = class_state_feat.size()[0]
                class_state_feat_means[k, h] += class_state_feat.sum(dim=0) / dataset_size

    return class_feat_means, class_state_feat_means


def train_loop(train_dataloader, model, class_feature_mean, hmd, class_state_feature_means):
    size = len(train_dataloader.dataset)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    for batch, (X, y_class, y_state) in enumerate(train_dataloader):

        # Move to GPU
        X = X.to(device)
        y_class = y_class.to(device)
        y_state = y_state.to(device)

        # Compute expression class and state prediction along with spatial feature vector
        feat, pred_class, pred_state = model(X.float())

        # Calculate loss
        loss = loss_function(y_class, y_state, pred_class, pred_state, class_feature_mean, class_state_feature_means,
                             feat, hmd)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # for p in model.parameters():
        #     print(p.grad.norm())

        # Reset gradient
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    main()
