import torch
from model import MicroExpressionCNN
from config import *

softplus = torch.nn.Softplus()

def half_min_distance(feat_means):
    half_min_distance = torch.zeros(EXP_CLASS_SIZE)
    for i in range(EXP_CLASS_SIZE):
        # Exclude the current class feature mean
        excluded_feat_means = torch.concat((feat_means[:i], feat_means[i+1:]))
        diff =  excluded_feat_means - feat_means[i]
        d = torch.linalg.norm(diff, dim=1, dtype=torch.float64, ord=2)
        half_min_distance[i] = d.detach().min() / 2
    return half_min_distance

def loss_function(gt_exp_class, gt_exp_state, pred_exp_class, pred_exp_state, class_feat_mean, sample_feat, half_min_distance):
    """

    :param half_min_distance: half_min_distance[i] is half distance between feature mean i and closest feature mean
    :param sample_feat: Extracted feature from last fully connected layer, has shape of (batch_size, feature_size)
    :param class_feat_mean: Spatial feature vector mean of all training examples for all class (Has to be updated every beginning of Epoch), has shape of (7, feature_size)
    :param gt_exp_class: Expression Class ground truth, it should have shape of (batch_size, EXP_CLASS_SIZE), which gt_exp_class[i,j] = 1 means i sample contain expression class j
    :param gt_exp_state: Expression State ground truth, it should have shape of (batch_size, EXP_STATE_SIZE), which gt_exp_state[i,j] = 1 means i sample contain expression state j
    :param pred_exp_class: Expression Class predicted probability, it should have shape of (batch_size, EXP_CLASS_SIZE), which gt_exp_class[i,j] = 0.7 means i sample has probability 0.7 of being class j
    :param pred_exp_state: Expression State predicted probability, it should have shape of (batch_size, EXP_STATE_SIZE), which gt_exp_class[i,j] = 0.7 means i sample has probability 0.7 of being state j

    """
    # E1 expression class classification error
    E1  = - (gt_exp_class * torch.log(pred_exp_class + 1e-5)).sum() # we add predicted probability with 1e-5 to avoid probability being zero, which will cause log to goes to minus infinity

    # E2 Intra class variation
    sample_class_feat_mean = torch.matmul(gt_exp_class.float(), class_feat_mean) # We do this so that each sample feature of class c will be subtracted by corresponding feature mean of the same class c
    sample_hmd = torch.matmul(gt_exp_class.float(), half_min_distance) # Same thing with half minimum distance, each sample will operate with its corresponding HMD of the same class as sample ground truth class
    d = torch.linalg.norm(sample_feat - sample_class_feat_mean, dim=1, dtype=torch.float64, ord=2).pow(2) - sample_hmd.pow(2)
    beta = 1 # Sharpness parameter
    E2 = softplus(d).sum() / 2
    # TODO: Debug E2 value too big

    # E3 expression state classification error
    E3 = - (gt_exp_state * torch.log(pred_exp_state + 1e-5)).sum()

    return E1 + E2 + E3

def test_loss_function():

    # Generate ground truth label for both expression class and expression state, both represented as indicator vector, ex: ExpressionClass(0,0,0,1,0,0,0) ExpressionState(1,0,0,0,0)
    gt_exp_class = torch.zeros((BATCH_SIZE, EXP_CLASS_SIZE), dtype=torch.int)
    exp_class_index = torch.randint(0, EXP_CLASS_SIZE, (BATCH_SIZE,))  # Randomize index of expression class that will be set to 1
    gt_exp_state = torch.zeros((BATCH_SIZE, EXP_STATE_SIZE), dtype=torch.int)
    exp_state_index = torch.randint(0, EXP_STATE_SIZE, (BATCH_SIZE,)) # Randomize index of expression state that will be set to 1

    for k in range(BATCH_SIZE):
        gt_exp_class[k, exp_class_index[k]] = 1
        gt_exp_state[k, exp_state_index[k]] = 1

    # Generate a single batch of samples so that we can get the model output feature vector along with expression class and state predicted probability
    x = torch.rand((BATCH_SIZE, 3, 64, 64))
    model = MicroExpressionCNN(EXP_CLASS_SIZE, EXP_STATE_SIZE)
    model.train()
    sample_feat, pred_exp_class, pred_exp_state = model(x)

    # Using gt_exp_class, we know which ground truth class that each sample belong to
    feat_means = torch.zeros((EXP_CLASS_SIZE, 512))
    for k in range(EXP_CLASS_SIZE):
        class_k_index = gt_exp_class[:, k]
        class_k_feat_mean = sample_feat.index_select(dim=0, index=class_k_index).mean(dim=0)
        feat_means[k] = class_k_feat_mean

    hmd = half_min_distance(feat_means)

    loss = loss_function(gt_exp_class, gt_exp_state, pred_exp_class, pred_exp_state, feat_means, sample_feat, hmd)

    print(f'sample_feat = {sample_feat}')
    print(f'pred_exp_class = {pred_exp_class}')
    print(f'pred_exp_state = {pred_exp_state}')

    print(f'loss = {loss}')
