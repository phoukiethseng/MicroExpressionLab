from dataset import MicroExpressionDataset
import torch
from torch.utils.data import DataLoader, random_split
from loss import cnn_loss_function, half_min_distance
from model.cnn import MicroExpressionCNN
from torch.optim.lr_scheduler import *
from torcheval.metrics import MulticlassAUROC

from utils.config import *
from utils.model import save_model, save_metrics


def main():
    torch.set_default_device(DEVICE)  # Default to CUDA Tensor if GPU compute is available, otherwise use CPU Tensor
    dataset = MicroExpressionDataset(os.path.join(DATASET_PATH_PREFIX, "label.csv"), DATASET_PATH_PREFIX)
    train_dataset, test_dataset = random_split(dataset, [0.85, 0.15], generator=torch.Generator(device=DEVICE))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  generator=torch.Generator(device=DEVICE), num_workers=WORKER_SIZE, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 generator=torch.Generator(device=DEVICE), num_workers=WORKER_SIZE, pin_memory=True)

    # This dataloader is used to calculate the class and state feature means and half minimum distance across the entire dataset
    epoch_dataloader_batch_size = 2000
    epoch_dataloader = DataLoader(train_dataset, batch_size=epoch_dataloader_batch_size, shuffle=False,
                                  generator=torch.Generator(device=DEVICE), num_workers=WORKER_SIZE, pin_memory=True)

    model = MicroExpressionCNN(EXP_CLASS_SIZE, EXP_STATE_SIZE)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}\n_____________________________________")
        print(
            'Calculating feature mean of Expression Class and State and Half-Minimum-Distance '
            'between Feature mean at the start of epoch...')
        class_feature_means, class_state_feature_means = get_exp_class_feature_mean(model, epoch_dataloader,
                                                                                    batch_size=epoch_dataloader_batch_size)
        hmd = half_min_distance(class_feature_means)
        print('Done')
        print(f'half minimum distance = {hmd}\nfeature mean = {class_feature_means}')
        train_loop(train_dataloader, model, class_feature_means, hmd, class_state_feature_means, optimizer)
        scheduler.step()

    # Save the model
    save_model(model, model_name='cnn')

    # Compute Area Under ROC
    exp_class_AUC = MulticlassAUROC(num_classes=EXP_CLASS_SIZE, average=None, device=DEVICE)
    exp_state_AUC = MulticlassAUROC(num_classes=EXP_STATE_SIZE, average=None, device=DEVICE)

    def compute_metric_callback(y_class, y_state, pred_class, pred_state):
        # Convert from one hot encoded to class index
        y_class = y_class.argmax(dim=1, keepdim=False)
        y_state = y_state.argmax(dim=1, keepdim=False)

        exp_class_AUC.update(pred_class, y_class)
        exp_state_AUC.update(pred_state, y_state)

    model.eval()
    test_loop(model, test_dataloader, compute_metric_callback)

    # Save metrics
    save_metrics(exp_class_AUC, "exp_class_auc", 'cnn')
    save_metrics(exp_state_AUC, "exp_state_auc", 'cnn')

    print(f"Expression Class AUC: {exp_class_AUC.compute()}")
    print(f"Expression State AUC: {exp_state_AUC.compute()}")


def get_exp_class_feature_mean(model, dataloader, batch_size=128):
    class_feat_means = torch.zeros((EXP_CLASS_SIZE, 512), device=DEVICE, requires_grad=False)
    class_state_feat_means = torch.zeros((EXP_CLASS_SIZE, EXP_STATE_SIZE, 512), device=DEVICE, requires_grad=False)

    a = torch.arange(0, EXP_CLASS_SIZE, requires_grad=False, dtype=torch.float32, device=DEVICE)
    b = torch.arange(0, EXP_STATE_SIZE, requires_grad=False, dtype=torch.float32, device=DEVICE)

    dataset_size = len(dataloader.dataset)

    exp_class_feature_sample_count = 0
    exp_class_state_feature_sample_count = 0

    for batch, (X, y_class, y_state) in enumerate(dataloader):

        if batch * batch_size % 10000 == 0:
            print(f'Batch: {batch} [{batch * batch_size} / {dataset_size}]')

        X = X.to(DEVICE)
        y_class = y_class.to(DEVICE)
        y_state = y_state.to(DEVICE)

        feature, pred_class, pred_state = model(X.float())

        # DO NOT DELETE THIS LINE
        feature = feature.detach()  # Avoid computational graph build up, which will cause increasing memory consumption until out of memory error

        gt_exp_class_index = torch.matmul(y_class.float(), a)
        gt_exp_state_index = torch.matmul(y_state.float(), b)

        for k in range(EXP_CLASS_SIZE):
            class_k_index = gt_exp_class_index == k
            class_feat = feature[class_k_index, :]
            exp_class_feature_sample_count += class_feat.size()[0]
            class_feat_means[k] += class_feat.sum(dim=0) / dataset_size
            for h in range(EXP_STATE_SIZE):
                state_h_index = torch.logical_and(class_k_index, gt_exp_state_index)
                class_state_feat = feature[state_h_index, :]
                exp_class_state_feature_sample_count += class_state_feat.size()[0]
                class_state_feat_means[k, h] += class_state_feat.sum(dim=0) / dataset_size

        exp_class_feature_sample_count = (class_feat_means * dataset_size) / exp_class_feature_sample_count
        exp_class_state_feature_sample_count = (
                                                       class_state_feat_means * dataset_size) / exp_class_state_feature_sample_count

    return class_feat_means, class_state_feat_means


def train_loop(train_dataloader, model, class_feature_mean, hmd, class_state_feature_means, optimizer):
    size = len(train_dataloader.dataset)

    for batch, (X, y_class, y_state) in enumerate(train_dataloader):

        # Move to GPU
        X = X.to(DEVICE)
        y_class = y_class.to(DEVICE)
        y_state = y_state.to(DEVICE)

        # Compute expression class and state prediction along with spatial feature vector
        feat, pred_class, pred_state = model(X.float())

        # Calculate loss
        loss = cnn_loss_function(y_class, y_state, pred_class, pred_state, class_feature_mean,
                                 class_state_feature_means,
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


def test_loop(model, test_loader, compute_metric_callbacks):
    with torch.no_grad():
        for batch, (X, y_class, y_state) in enumerate(test_loader):
            # Move to GPU
            X = X.to(DEVICE)
            y_class = y_class.to(DEVICE)
            y_state = y_state.to(DEVICE)

            # Compute expression class and state prediction along with spatial feature vector
            feat, pred_class, pred_state = model(X.float())

            compute_metric_callbacks(y_class, y_state, pred_class, pred_state)


if __name__ == '__main__':
    main()
