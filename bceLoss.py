import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BCE_loss(pred,label):
    bce_loss = nn.BCELoss(size_average=True)
    bce_out = bce_loss(pred, label)
    print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out

# focal loss code taken from: https://amaarora.github.io/2020/06/29/FocalLoss.html
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma
        # print("self.alpha: ",self.alpha)

    def forward(self, inputs, targets):
        # print("inputs.shape: ",inputs.shape)
        # print("targets.shape: ",targets.shape)
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.squeeze(inputs), torch.squeeze(targets), reduction='none')
        targets = targets.type(torch.long)
        #at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        #F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()