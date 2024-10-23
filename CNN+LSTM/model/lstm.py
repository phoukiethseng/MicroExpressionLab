from torch import nn

class MicroExpressionLSTM(nn.LSTM):
    def __init__(self, in_feature, num_cell_unit):
        super(MicroExpressionLSTM, self).__init__(input_size=in_feature, hidden_size=num_cell_unit, num_layers=2, bias=True, batch_first=True)
