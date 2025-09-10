import numpy as np
import oneflow.nn as nn
from sdmodel import ClientImageEncoder
from tqdm import tqdm
import oneflow.nn.functional as F
from sklearn.cluster import KMeans
import torchvision


class Client: # as a user
    def __init__(self, dataloader, classes,noise_level = 500 , beta=0):
        self.dataloader = dataloader
        self.model = ClientTune(classes,noise_level).cuda()
        self.ori_features = None
        self.beta = beta
        self.K = 5
        self.classes = classes
        
    def get_ori_features(self):
        self.ori_features = self.__get_all_fea__()
    
    def __get_all_fea__(self):
        
        ori_features,labels = [],[]
        for i, (image, label) in enumerate(tqdm(self.dataloader)):
            image = image.cuda()
            #label = label.cuda()
            feature = self.model(image,True)
            del image
            ori_features.append(feature)    
            #labels.append(label)
        ori_features = torch.cat(ori_features,0)
        #labels = torch.cat(labels,0)
        #self.labels = labels
        ori_features = torch.tensor(ori_features,dtype=torch.float16)
        return ori_features

    def post_precess(self,beta,proto):
        
        dis = -2*self.ori_features@proto.T+\
        torch.sum(self.ori_features**2,dim=1).unsqueeze(1)+\
        torch.ones((self.ori_features.shape[0],proto.shape[0]),dtype = torch.float16).cuda()*torch.sum(proto**2,dim=1).unsqueeze(1).T
        
        pseudo_label = torch.argmin(dis,dim=1)
        #pseudo_label accuracy
        #print((pseudo_label == self.labels).sum().item()/pseudo_label.shape[0])
        dtype = self.ori_features.dtype
        new_features = {}
        for c in range(self.classes):
            tempfeature = self.ori_features[torch.where(pseudo_label==c)]
            if len(tempfeature)==0:
                new_features[c] = None
            elif tempfeature.shape[0]>self.K:
                #randidx = torch.randint(low=0,high=tempfeature.shape[0],size=(self.K,))
                
                #new_features[c] = torch.tensor(tempfeature[randidx],dtype=dtype).cuda()
                
                km = KMeans(n_clusters=self.K, max_iter=100).fit(tempfeature.cpu())#,random_state=0
                if beta>0:
                    new_features[c] = torch.tensor(km.cluster_centers_,dtype=dtype).cuda()+beta*torch.randn(tempfeature.shape,dtype=dtype).cuda()
                else:
                    new_features[c] = torch.tensor(km.cluster_centers_,dtype=dtype).cuda()
                
                
            else:
                new_features[c] = tempfeature+beta*torch.randn(tempfeature.shape,dtype=dtype).cuda()

        return new_features
    
    def get_features(self,proto):
        return self.post_precess(self.beta,proto)
    
    def train(self,lr,epochs):
        self.model.final_proj.train()
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.final_proj.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)#weight_decay=0.001
        for _ in tqdm(range(epochs)):
            for i, (image, label) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.model(image)
                loss = task_criterion(output,label)
                loss.backward()
                optimizer.step()

class ClientTune(nn.Module):
    def __init__(self, classes=345,noise_level=0):
        super(ClientTune, self).__init__()
        
        self.encoder = ClientImageEncoder()
        self.noise_level = noise_level
        self.final_proj = nn.Sequential(
            nn.Linear(2048,classes,dtype = torch.float16)
            # nn.Linear(768,512),
            # nn.ReLU(),
            # nn.Linear(512,classes)
        )
    
    def forward(self, x, get_fea=False):
        with torch.no_grad():
            fea = self.encoder(x,self.noise_level)
            # fea = fea+1*torch.randn(fea.shape).cuda()
        if get_fea:
            return fea.view(fea.shape[0],-1)
        out = self.final_proj(fea.view(fea.shape[0],-1))
        return out