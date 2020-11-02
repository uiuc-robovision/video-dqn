import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch

import torchvision.models as models

class HabitatDQNMultiAction(nn.Module):
    def __init__(self, action_dim,num_classes=5,extra_capacity=False,panorama=True):
        super(HabitatDQNMultiAction, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.extra_capacity = extra_capacity
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.panorama = panorama
        if panorama:
            self.num_frames = 4
        else:
            self.num_frames = 1
        # try resnet as eval mode during training
        # log predicted q values to check for huge gradients
        # recreate old results with new network
        # log norm of features after conv layer
        # check l1 loss for q value
        # try huber loss
        # geodesic distance 
        if extra_capacity:
            print("Model loading with extra_capacity")
            #freeze resnet params
            self.features = nn.Sequential(*list(self.resnet.children())[:-2],nn.Conv2d(512,64,(3,3)),nn.ReLU(),nn.Flatten())
            self.top=nn.Sequential(nn.Linear(1600*self.num_frames,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,action_dim*self.num_classes))
        else:
            self.features = nn.Sequential(*list(self.resnet.children())[:-1])
            self.top = nn.Linear(in_features=512 * self.num_frames, out_features=action_dim*self.num_classes)

    # set training mode, which has resnet in eval mode
    def set_train(self):
        self.train()
        if self.extra_capacity:
            self.resnet.eval()
            # check this is working
            # print(list(self.features.children())[-5].training)

    def forward(self, inp):
        if self.num_frames == 1 and len(inp.shape) == 4:
            inp = inp.unsqueeze(1)
        if inp.shape[1] != self.num_frames:
            raise Exception("bad shape")
        feats = [
            self.features(inp[:, i, ...]) for i in range(0, self.num_frames)
        ]
        combined = torch.cat(feats, 1).squeeze()
        out= self.top(combined)
        return out.view((-1,self.num_classes,self.action_dim))

if __name__ == "__main__":
    model = HabitatDQNMultiAction(3,extra_capacity=True,panorama=False)
    # x = torch.ones((2,3,224,224))
    x = torch.ones((2,4,3,224,224))
    print(model(x))
    exit()
    x = torch.ones((4,3,224,224))
    self.sub_model(x).shape
    self.resnet
