from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models import resnet18, resnet34, resnet50
    

    
class Model( nn.Module ):

    def __init__(self, basetype='res18'):
        
        super().__init__()
        self.basetype = basetype
        self.representation = self.get_base_model()            
        return

    
    def get_base_model(self):

        if 'res18' == self.basetype:
            net = resnet18(pretrained=True)
        if 'res34' == self.basetype:
            net = resnet34(pretrained=True)
        if 'res50' == self.basetype:
            net = resnet50(pretrained=True)

        net = nn.Sequential( *list(net.children())[:-1] )
        return net


    def forward(self, inp):
        rep_out = self.representation(inp)
        rep_out = torch.flatten(rep_out, 1)
        return rep_out



def Train(model, trainloader, criterion, optimizer, logger=None):

    train_loss = 0.
    for batchid, data in enumerate(trainloader):
        
        img, trues, _, _ = data
        if torch.cuda.is_available():
            img, trues = img.cuda(), trues.cuda()
        data = {'trues':trues}

        embed = model(img)
        data['loss'] = criterion(embed, trues)

        optimizer.zero_grad()        
        data['loss'].backward()
        optimizer.step()
        
        data['preds'] = embed

        if logger!=None:
            logger.add_step('train', data)

    return 



def Test(model, testloader, prototypes, logger=None):

    model.eval()
    features, labels = [], []

    for data in testloader:

        img, trues, _, _ = data
        if torch.cuda.is_available():
            img, trues = img.cuda(), trues.cuda()

        with torch.no_grad():
            embed = model(img)
            embed = embed.detach().cpu()

        if features==[]:
            features = embed
            labels = trues
        else:
            features=torch.cat((features,embed))
            labels = torch.cat((labels,trues))

    p, num_classes = [], max(prototypes.keys())
    for i in range(num_classes):
        p.append(prototypes[i])
    prototypes = torch.stack(p)

    dist = torch.cdist(features, prototypes)
    preds = torch.argmin(dist, dim=1)

    data = {
        'trues':labels,
        'preds':preds
    }            

    if logger != None:
        logger.add_step('test', data)

    model.train()
    return data


def GetPrototypes(model, trainloader):

    model.eval()
    features, labels = [], []
    prototypes = {}
    
    for data in trainloader:

        img, trues, _, _ = data
        if torch.cuda.is_available():
            img, trues = img.cuda(), trues.cuda()

        with torch.no_grad():
            embed = model(img)
            embed = embed.detach().cpu()

        if features==[]:
            features = embed
            labels = trues
        else:
            features=torch.cat((features,embed))
            labels = torch.cat((labels,trues))

    srt, end = torch.min(labels), torch.max(labels)
    for i in range(srt, end+1):
        prototypes[i] = torch.mean(features[labels==i].t(), dim=1)

    model.train()
    return prototypes



class TripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    
    def forward(self, inputs, targets):
        
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(targets)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss