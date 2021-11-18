import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50


class Identity(nn.Module):
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, x):
        return x
    
    

class Model( nn.Module ):

    def __init__(self, class_size, basetype='res18'):
        
        super().__init__()
        self.class_size = class_size
        self.basetype = basetype
        self.representation = self.get_base_model()

        with torch.no_grad():
            out = self.representation(torch.rand(5, 3, 224, 224))
            rep_dim = out.shape[1]

        self.classifier = nn.Linear(rep_dim, class_size)
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
        clsf_out = self.classifier(rep_out)
        return clsf_out



def Train(model, trainloader, criterion, optimizer, logger=None):

    train_loss = 0.
    for batchid, data in enumerate(trainloader):
        
        img, trues, _, _ = data
        if torch.cuda.is_available():
            img, trues = img.cuda(), trues.cuda()
        data = {'trues':trues}

        probs = model(img)
        data['loss'] = criterion(probs, trues)

        optimizer.zero_grad()        
        data['loss'].backward()
        optimizer.step()
        
        data['preds'] = torch.argmax(probs, dim=1)

        if logger!=None:
            logger.add_step('train', data)

    return



def Test(model, testloader, criterion, logger=None):

    model.eval()
    loss = 0.

    with torch.no_grad():
        for data in testloader:

            img, trues, _, _ = data
            if torch.cuda.is_available():
                img, trues = img.cuda(), trues.cuda()
            data = {'trues': trues}

            probs = model(img)
            data['preds'] = torch.argmax(probs, dim=1)

            if logger != None:
                logger.add_step('test', data)

    model.train()
    return