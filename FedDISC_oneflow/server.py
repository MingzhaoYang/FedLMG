import numpy as np
import oneflow.nn as nn
from torchvision import models
import os
from PIL import Image
import oneflow.nn.functional as F
import torchvision.utils as tvu
from sdmodel import ImageGenerator
from oneflow.utils.data import DataLoader, Dataset
from torchvision import transforms
from sdmodel import ClientImageEncoder
from tqdm import tqdm
import os
from PIL import Image
import oneflow
from oneflow.utils.data import Dataset
import torchvision
from datasets.openimage import get_openimage_classes
from utils import partition,Truncated,evaluation

class Server: # as a user
    def __init__(self,server_labeled_loader,transform,bs,classes,beta=0,repeat=1,steps=50,imgpath=''):
        self.client_features = None
        self.server_labeled_loader = server_labeled_loader
        self.agg_features = None
        self.dataset = None
        self.args = {'trans':transform,'bs':bs,'beta':beta}
        self.dataloader = None
        
        self.model = ServerTune(classes = classes).cuda()
        
        self.tempmodel = TempModel(classes = classes)
        self.new_features = None
        self.repeat = repeat
        self.steps = steps
        self.num_classes = classes
        self.imgpath = imgpath
        self.start_class = 0
        
        path = r"/home/share/DomainNet/clipart"
        f = os.listdir(path)
        for i in range(len(f)):
            f[i] = f[i].lower()
        self.class_prompts = sorted(f)   
        
        
        nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
        f = os.listdir(nicopp_path)
        for i in range(len(f)):
            f[i] = f[i].lower()
        self.nicopp_class_prompts = sorted(f)        
        
        self.open_image_class_prompts,self.open_image_rough_classes = get_openimage_classes()
        
        
    def get_class_proto(self):
        proto = torch.zeros([self.num_classes,2048],dtype = torch.float16).cuda()
        ori_features,labels =[],[] 
        
        for i, (image, label) in enumerate(tqdm(self.server_labeled_loader)):
            image = image.cuda()
            label = label.cuda()
            feature = self.model(image,True)
            ori_features.append(feature)
            labels.append(label)
            
        ori_features = torch.cat(ori_features,0)
        labels = torch.cat(labels)
        allclass = list(set(labels.tolist()))
        
        self.ori_features = ori_features
        
        feadict = {}
        for i in allclass:
            proto[i-self.start_class]=torch.mean(ori_features[torch.where(labels==i)],dim=0)
        return proto
    
    
    def update_features(self,features, do_generate = True, directtrain=False):
        
        self.client_features = features
        self.global_features = self.__aggregation_fea__()
        
        if directtrain:
            return 
        if do_generate:
            self.dataset = self.Generator(self.global_features,self.client_features,self.args['trans'],path = self.imgpath,repeat = self.repeat)
            
        #DomainNet
        #self.dataset = ServerData_read(f'/home/share/gen_data/{self.imgpath}',self.args['trans'])
        
        #OpenImage
        self.dataset = ServerData_read_openimage(f'/home/share/gen_data/{self.imgpath}',self.args['trans'])
        
        #NICO++
        #self.dataset = ServerData_read_nico(f'/home/share/gen_data/{self.imgpath}',self.args['trans'])
        
        self.dataloader = DataLoader(self.dataset,batch_size=self.args['bs'],shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
        
    #class_global
    def __aggregation_fea__(self,):
        
        classes = []
        for fea in self.client_features:
            classes += list(fea.keys())
        classes = list(set(classes))

        global_features = [{i:None for i in range(self.num_classes)} for j in range(len(self.client_features))]
        
        for cidx in range(len(self.client_features)):
            for i in self.client_features[cidx].keys():
                if self.client_features[cidx][i]!=None:
                    global_features[cidx][i] = torch.mean(self.client_features[cidx][i],dim=0)
        
        return global_features
    
    
    
    def Generator(self,global_features,client_features,transform,path='domainnet_0',repeat=1):
        
        classes = list(client_features[0].keys())
        img_gen = ImageGenerator()

        gener = torch.Generator("cuda")

        #create data dirs
        
        #DomainNet
        #for i in classes:
        #    if not os.path.exists(f'/home/share/gen_data/{path}/{self.class_prompts[i+self.start_class]}/'):
        #        os.makedirs(f'/home/share/gen_data/{path}/{self.class_prompts[i+self.start_class]}/')
        
        #OpenImage
        for i in classes:
            if not os.path.exists(f'/home/share/gen_data/{path}/{self.open_image_rough_classes[i+self.start_class]}/'):
                os.makedirs(f'/home/share/gen_data/{path}/{self.open_image_rough_classes[i+self.start_class]}/')        
        
        #NICO++
        #for i in classes:
        #    if not os.path.exists(f'/home/share/gen_data/{path}/{self.nicopp_class_prompts[i+self.start_class]}/'):
        #        os.makedirs(f'/home/share/gen_data/{path}/{self.nicopp_class_prompts[i+self.start_class]}/')
        
                
        client = len(global_features)
        datapath = []
        for c in tqdm(classes):
            idx = 0
            for client_idx in range(client):
                if client_features[client_idx][c]!=None:
                    #for imgfea in tqdm(self.client_features[client_idx][c]):
                    for imgfea in tqdm(client_features[client_idx][c]):
                        
                        ###cluster centers###
                        imgfea = imgfea.unsqueeze(0) 

                        for i in range(repeat):
                            imgfea =imgfea.cuda()
                            
                            ###domain-specific features###
                            global_fea = self.global_features[idx%client][c]
                            #global_fea = None
                            # if global_fea !=None:
                            #    imgfea = 0.5*imgfea +0.5*global_fea.unsqueeze(0)
                            #    global_fea = None
                            
                            noised_imgfea = img_gen.noise_image_embeddings(imgfea,noise_level=200)
                            input_fea = noised_imgfea
                            input_fea = torch.cat([torch.zeros_like(input_fea),input_fea],0)
                            
                            #DomainNet
                            #output = img_gen(prompt = self.class_prompts[c+self.start_class],image_embeddings = input_fea,global_embeddings = global_fea, generator =gener,num_inference_steps =self.steps)
                            
                            #NICO++
                            #output = img_gen(prompt = self.nicopp_class_prompts[c+self.start_class],image_embeddings = input_fea,global_embeddings = global_fea, generator =gener,num_inference_steps =self.steps)
                            
                            #OpenImage
                            output = img_gen(prompt = self.open_image_rough_classes[c],image_embeddings = input_fea,global_embeddings = global_fea, generator =gener,num_inference_steps =self.steps)
                            
                            #No Prompt
                            #output = img_gen(prompt = '',image_embeddings = input_fea,global_embeddings = global_fea, generator =gener,num_inference_steps =self.steps)
                            
                            
                            if output["nsfw_content_detected"][0] == False:
                                image = output["images"][0]

                                #OpenImage
                                tvu.save_image(torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255., f'/home/share/gen_data/{path}/{self.open_image_rough_classes[c]}/{self.open_image_class_prompts[client_idx][c]}_{self.open_image_class_prompts[i%client][c]}_{i}_{idx}.jpg')
                                datapath.append((f'/home/share/gen_data/{path}/{self.open_image_rough_classes[c]}/{self.open_image_class_prompts[client_idx][c]}_{self.open_image_class_prompts[i%client][c]}_{i}_{idx}.jpg',c))
                                
                                #DomainNet
                                #tvu.save_image(torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255., f'/home/share/gen_data/{path}/{self.class_prompts[i+self.start_class]}_domain{j}.jpg')
                                #datapath.append((f'/home/share/gen_data/{path}/{self.class_prompts[i+self.start_class]}_domain{j}.jpg',c))
                                
                                #NICO++
                                #tvu.save_image(torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255., f'/home/share/gen_data/{path}/{self.nicopp_class_prompts[c+self.start_class]}/{self.nicopp_class_prompts[c+self.start_class]}_domain{idx%client}_{idx}.jpg')
                                #datapath.append((f'/home/share/gen_data/{path}/{self.nicopp_class_prompts[c+self.start_class]}/{self.nicopp_class_prompts[c+self.start_class]}_domain{idx%client}_{idx}.jpg',c))
                                
                            idx+=1
                    
                    
        torch.save(datapath,'datapath.pth')

        ServerDataset = ServerData(datapath,transform)
        return ServerDataset
    
     
    def train(self,lr,epochs,test_data):
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)#weight_decay=0.001
        for _ in tqdm(range(epochs)):
            for i, (image, label) in enumerate((self.dataloader)):
                self.model.train()
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.model(image)
                
                loss = task_criterion(output,label)
                loss.backward()
                optimizer.step()
            top1, topk = evaluation(self.model,test_data)
            print(f'final server model: top1 {top1}, top5 {topk}')
                
    def get_client_features(self):
        return [self.client_features,]      
    
    def directtrain(self,lr,epochs,test_data):
        self.tempmodel = self.tempmodel.cuda()
        dataset = FeatureData(self.get_client_features(),self.num_classes)
        dataloader = DataLoader(dataset,batch_size=self.args['bs'],shuffle=True,num_workers=8,pin_memory=True,drop_last=False)
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.tempmodel.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)#weight_decay=0.001
        for _ in tqdm(range(epochs)):
            self.tempmodel.train()
            for i, (image, label) in enumerate((dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.tempmodel(image,input_image=False)
                loss = task_criterion(output,label)
                loss.backward()
                optimizer.step()
            
            top1, topk = evaluation(self.tempmodel,test_data)
            print(f'final server model: top1 {top1}, top5 {topk}')
        
    
class ServerTune(nn.Module):
    def __init__(self, classes=345,noise_level = 0):
        super(ServerTune, self).__init__()
        
        self.encoder = ClientImageEncoder()
        
        self.noise_level = noise_level 
        
        self.final_proj = nn.Sequential(
            nn.Linear(2048,classes,dtype = torch.float16)
        )
    
    def forward(self, x, get_fea=False,return_1024=False):
        with torch.no_grad():
            fea = self.encoder(x,noise_level = self.noise_level)
        if get_fea:
            return fea.view(fea.shape[0],-1)
        
        out = self.final_proj(fea.view(fea.shape[0],-1))
        return out
    
class TempModel(nn.Module):
    def __init__(self, classes=345):
        super(TempModel, self).__init__()
        
        self.encoder = ClientImageEncoder()
        self.final_proj = nn.Sequential(
            nn.Linear(2048,classes,dtype = torch.float16)
        )
    
    def forward(self, x,get_fea=False,input_image=True):
        
        if input_image:
            with torch.no_grad():
                x =  self.encoder(x)
                    
        out = self.final_proj(x)
        
        return out

class ServerData_read(Dataset):
    def __init__(self, root_dir,transforms=None):
        super(ServerData_read, self).__init__()
        self.root_dir = root_dir
        
        path = r"/home/share/DomainNet/clipart"
        
        f = os.listdir(path)
        for i in range(len(f)):
            f[i] = f[i].lower()
        self.class_prompts = sorted(f) 
        self.classes = {c:i for i,c in enumerate(self.class_prompts) if i<30}
        
        self.images = []
        self.targets = []
        self.transforms = transforms
        for c in self.classes:
            class_dir = os.path.join(self.root_dir, str(c))
            print(class_dir)
            for image_name in os.listdir(class_dir):
                if '.ipynb_checkpoints' in image_name: continue
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.targets.append(self.classes[c])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = self.targets[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

class ServerData_read_openimage(Dataset):
    def __init__(self, root_dir,transforms=None):
        super(ServerData_read_openimage, self).__init__()
        self.root_dir = root_dir
        
        self.images = []
        self.targets = []
        self.transforms = transforms
        self.open_image_class_prompts,self.open_image_rough_classes = get_openimage_classes()
        
        self.classes = {c:i for i,c in enumerate(self.open_image_rough_classes)}
        
        for c in self.open_image_rough_classes:
            class_dir = os.path.join(self.root_dir, str(c))
            for image_name in os.listdir(class_dir):
                if '.ipynb_checkpoints' in image_name: continue
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.targets.append(self.classes[c])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = self.targets[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
class ServerData_read_nico(Dataset):
    def __init__(self, root_dir,transforms=None):
        super(ServerData_read_nico, self).__init__()
        self.root_dir = root_dir
        
        nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
        f = os.listdir(nicopp_path)
        for i in range(len(f)):
            f[i] = f[i].lower()
        self.nicopp_class_prompts = sorted(f)   
        
        self.classes = {c:i for i,c in enumerate(self.nicopp_class_prompts)}
        
        self.images = []
        self.targets = []
        self.transforms = transforms
        for c in self.classes:
            class_dir = os.path.join(self.root_dir, str(c))
            print(class_dir)
            for image_name in os.listdir(class_dir):
                if '.ipynb_checkpoints' in image_name: continue
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.targets.append(self.classes[c])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = self.targets[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
class ServerData(Dataset):
    def __init__(self, data_paths,transforms=None):
        super(ServerData, self).__init__()
        self.data = data_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data[index][0])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data[index][1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data)
    
class FeatureData(Dataset):
    def __init__(self, datas, classes):
        super(FeatureData, self).__init__()
        self.num_classes = classes
        self.data = self.__getalldata__(datas)
        
        
    def __getalldata__(self,datas):
        alldata = []
        for c in range(self.num_classes):
            for data in datas[0]:
                if data[c]==None: continue
                for imgfea in data[c]:
                    alldata.append((imgfea.cpu(),c))
            if len(datas)>1:
                for imgfea in datas[1][c]:
                    alldata.append((imgfea.cpu(),c))
                
        return alldata
            
    def __getitem__(self, index):
        img = self.data[index][0]
        label = self.data[index][1]
        return img, label

    def __len__(self):
        return len(self.data)
    
    