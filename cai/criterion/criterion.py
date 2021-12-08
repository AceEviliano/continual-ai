import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


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
    
    
class EWC(): 

    def __init__(self, model, dataloader):
        self.mean = {}
        self.precision = {}
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        for n, p in deepcopy(self.params).items():
            self.mean[n] = p
            self.precision[n] = torch.zeros_like(p)            
        
        for data in dataloader:

            img, target, original, taskid = data
            if torch.cuda.is_available():
                img, target = img.cuda(), target.cuda()
            
            model.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()

            for n, p in model.named_parameters():
                self.precision[n].data += p.grad.data ** 2 / len(dataloader.dataset)

        self.precision = {n: p for n, p in self.precision.items()}
    
        return


    def penalty(self, model):
        
        loss = 0.
        for n, p in model.named_parameters():
            _loss = self.precision[n] * (p - self.mean[n]) ** 2
            loss += _loss.sum()

        return loss

    def altered_construct(self, model, dataloader):
        
        self.new_mean = {}
        self.new_precision = {}
        self.new_params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        for n, p in deepcopy(self.new_params).items():
            self.new_mean[n] = p
            self.new_precision[n] = torch.zeros_like(p)            
        
        for data in dataloader:

            img, target, original, taskid = data
            if torch.cuda.is_available():
                img, target = img.cuda(), target.cuda()
            
            model.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()

            for n, p in model.named_parameters():
                self.new_precision[n].data += p.grad.data ** 2 / len(dataloader.dataset)

        self.new_precision = {n: p for n, p in self.precision.items()}
        model.train()
        
        for k in self.mean.keys():
            prec = torch.stack([self.precision[k], self.new_precision[k]], dim=0)
            mean = torch.stack([self.mean[k], self.new_mean[k]], dim=0)
            inx = torch.argmax(prec, dim=0, keepdim=True)
            self.precision[k] = torch.gather(prec,0,inx)
            self.mean[k] = torch.gather(mean,0,inx)

        return