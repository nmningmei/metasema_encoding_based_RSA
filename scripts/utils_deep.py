#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:22:18 2021

@author: nmei
"""
import os

import pandas as pd
import numpy  as np

import torch
from torch          import nn,no_grad,optim
from torch.utils    import data
from torch.nn       import functional as F
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision          import transforms
from  torchvision         import models as Tmodels


softmax_dim = 1

#candidate models
def candidates(model_name,pretrained = True,):
    picked_models = dict(
            resnet18        = Tmodels.resnet18(pretrained           = pretrained,
                                              progress              = False,),
            alexnet         = Tmodels.alexnet(pretrained            = pretrained,
                                             progress               = False,),
            # squeezenet      = Tmodels.squeezenet1_1(pretrained      = pretrained,
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(pretrained           = pretrained,
                                              progress              = False,),
            densenet169     = Tmodels.densenet169(pretrained        = pretrained,
                                                 progress           = False,),
            inception       = Tmodels.inception_v3(pretrained       = pretrained,
                                                  progress          = False,),
            # googlenet       = Tmodels.googlenet(pretrained          = pretrained,
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(pretrained = pretrained,
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(pretrained       = pretrained,
                                                  progress          = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]
def hidden_activation_functions(activation_func_name):
    funcs = dict(relu = nn.ReLU(),
                 selu = nn.SELU(),
                 elu = nn.ELU(),
                 sigmoid = nn.Sigmoid(),
                 tanh = nn.Tanh(),
                 linear = None,
                 )
    return funcs[activation_func_name]
class easy_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {classifier} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrain_model: nn.Module, pretrained model object
    hidden_units: int, hidden layer units
    hidden_activation: nn.Module, activation layer
    hidden_dropout: float (0,1), dropout rate
    output_units: int, output layer units

    Outputs
    --------------------
    model: nn.Module, a modified model with new {classifier} component with
    {feature} frozen untrainable <-- this is done prior to feed to this function
    """
    def __init__(self,
                 pretrain_model,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 in_shape = (1,3,128,128),
                 ):
        super(easy_model,self).__init__()
        torch.manual_seed(12345)
        in_features             = nn.AdaptiveAvgPool2d((1,1))(pretrain_model.features(torch.rand(*in_shape))).shape[1]
        avgpool                 = nn.AdaptiveAvgPool2d((1,1))
        hidden_layer            = nn.Linear(in_features,hidden_units)
        output_layer            = nn.Linear(hidden_units,output_units)
        if hidden_dropout > 0:
            dropout             = nn.Dropout(p = hidden_dropout)
        
        print(f'feature dim = {in_features}')
        self.features           = nn.Sequential(pretrain_model.features,
                                                avgpool,)
        if (hidden_activation is not None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,
                                                dropout,)
        elif (hidden_activation is not None) and (hidden_dropout == 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,)
        elif (hidden_activation == None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                dropout,)
        elif (hidden_activation == None) and (hidden_dropout == 0):
            self.hidden_layer   = hidden_layer
        
        self.output_layer       = nn.Sequential(output_layer,
                                                nn.LogSoftmax(dim = 1)
                                                )

    def forward(self,x,):
        out     = torch.squeeze(torch.squeeze(self.features(x),3),2)
        hidden  = self.hidden_layer(out)
        outputs = self.output_layer(hidden)
        return outputs,hidden


class resnet_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {fc} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrain_model: nn.Module, pretrained model object
    hidden_units: int, hidden layer units
    hidden_activation: nn.Module, activation layer
    hidden_dropout: float (0,1), dropout rate
    output_units: int, output layer units

    Outputs
    --------------------
    model: nn.Module, a modified model with new {fc} component with
    {feature} frozen untrainable <-- this is done prior to feed to this function
    """

    def __init__(self,
                 pretrain_model,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 ):
        super(resnet_model,self).__init__()
        torch.manual_seed(12345)
        avgpool         = nn.AdaptiveAvgPool2d((1,1))
        in_features     = pretrain_model.fc.in_features
        hidden_layer    = nn.Linear(in_features,hidden_units)
        dropout         = nn.Dropout(p = hidden_dropout)
        output_layer    = nn.Linear(hidden_units,output_units)
        res_net         = torch.nn.Sequential(*list(pretrain_model.children())[:-2])
        print(f'feature dim = {in_features}')
        
        self.features           = nn.Sequential(res_net,
                                      avgpool)
        if (hidden_activation is not None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,
                                                dropout,)
        elif (hidden_activation is not None) and (hidden_dropout == 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,)
        elif (hidden_activation == None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                dropout,)
        elif (hidden_activation == None) and (hidden_dropout == 0):
            self.hidden_layer   = hidden_layer
        self.output_layer       = nn.Sequential(output_layer,
                                                nn.LogSoftmax(dim = 1)
                                                )
        
    def forward(self,x):
        out     = torch.squeeze(torch.squeeze(self.features(x),3),2)
        hidden  = self.hidden_layer(out)
        outputs = self.output_layer(hidden)
        return outputs,hidden

def build_model(pretrain_model_name,
                hidden_units,
                hidden_activation_name,
                hidden_dropout,
                output_units,
                ):
    pretrain_model      = candidates(pretrain_model_name)
    hidden_activation   = hidden_activation_functions(hidden_activation_name)
    for params in pretrain_model.parameters():
        params.requires_grad = False
    if define_type(pretrain_model_name) == 'simple':
        model_to_train = easy_model(
                            pretrain_model      = pretrain_model,
                            hidden_units        = hidden_units,
                            hidden_activation   = hidden_activation,
                            hidden_dropout      = hidden_dropout,
                            output_units        = output_units,
                            )
    elif define_type(pretrain_model_name) == 'resnet':
        model_to_train = resnet_model(
                            pretrain_model      = pretrain_model,
                            hidden_units        = hidden_units,
                            hidden_activation   = hidden_activation,
                            hidden_dropout      = hidden_dropout,
                            output_units        = output_units,
                            )
    return model_to_train

def define_augmentations(image_resize = 128,noise_level = None,do_augmentations = True):
    augmentations = {
        'train':simple_augmentations(image_resize,noise_level,do_augmentations),
        'valid':simple_augmentations(image_resize,noise_level,do_augmentations),
    }
    return augmentations

def noise_fuc(x,noise_level = 1,):
    """
    add guassian noise to the images during agumentation procedures

    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, standard deviation of the gaussian distribution
    """
    generator = torch.distributions.normal.Normal(0,noise_level)
    return x + generator.sample(x.shape)

def simple_augmentations(image_resize = 128,noise_level = None,do_augmentations = True):
    if do_augmentations:
        if noise_level is not None:
            return transforms.Compose([
        transforms.Resize((image_resize,image_resize)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(45,),
        transforms.RandomVerticalFlip(p = 0.5,),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:noise_fuc(x,noise_level)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        else:
            return transforms.Compose([
        transforms.Resize((image_resize,image_resize)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(45,),
        transforms.RandomVerticalFlip(p = 0.5,),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.Grayscale(num_output_channels = 3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
class customizedDataset(ImageFolder):
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path


def data_loader(data_root:str,
                augmentations:transforms    = None,
                batch_size:int              = 8,
                num_workers:int             = 2,
                shuffle:bool                = True,
                return_path:bool            = False,
                drop_last:bool              = False,
                )->data.DataLoader:
    """
    Create a batch data loader from a given image folder.
    The folder must be organized as follows:
        main ---
             |
             -----class 1 ---
                         |
                         ----- image 1.jpeg
                         .
                         .
                         .
            |
            -----class 2 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
            |
            -----class 3 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
    Input
    --------------------------
    data_root: str, the main folder
    augmentations: torchvision.transformers.Compose, steps of augmentation
    batch_size: int, batch size
    num_workers: int, CPU --> GPU carrier, number of CPUs
    shuffle: Boolean, whether to shuffle the order
    return_pth: Boolean, lod the image paths

    Output
    --------------------------
    loader: DataLoader, a Pytorch dataloader object
    """
    if return_path:
        datasets = customizedDataset(
                root                        = data_root,
                transform                   = augmentations
                )
    else:
        datasets    = ImageFolder(
                root                        = data_root,
                transform                   = augmentations
                )
    loader      = data.DataLoader(
                datasets,
                batch_size                  = batch_size,
                num_workers                 = num_workers,
                shuffle                     = shuffle,
                drop_last                   = drop_last,
                )
    return loader

def createLossAndOptimizer(net, learning_rate:float = 1e-4):
    """
    To create the loss function and the optimizer

    Inputs
    ----------------
    net: nn.Module, torch model class containing parameters method
    learning_rate: float, learning rate

    Outputs
    ----------------
    loss: nn.Module, loss function
    optimizer: torch.optim, optimizer
    """
    #Loss function
    loss        = nn.NLLLoss()
    #Optimizer
    optimizer   = optim.Adam([params for params in net.parameters()],
                              lr = learning_rate,
                              weight_decay = 1e-6)

    return(loss, optimizer)



def train_loop(net,
               loss_func,
               optimizer,
               dataloader,
               device,
               categorical          = True,
               idx_epoch            = 1,
               print_train          = False,
               l2_lambda            = 0,
               l1_lambda            = 0,
               ):
    """
    A for-loop of train the autoencoder for 1 epoch

    Inputs
    -----------
    net: nn.Module
        torch model class containing parameters method
    loss_func: nn.Module
        loss function
    optimizer: torch.optim
        optimizer
    dataloader: torch.data.DataLoader
        dataloader to feed
    device:str or torch.device, where the training happens
    idx_epoch:int
        for print
    print_train:Boolean
        debug tool
    l2_lambda:float
        L2 regularization lambda term
    l1_lambda:float
        L1 regularization alpha term

    Outputs
    ----------------
    train_loss: torch.Float, average training loss
    net: nn.Module, the updated model

    """
    from tqdm import tqdm
    train_loss   = 0.
    # set the model to "train"
    net.to(device).train(True)
    # verbose level
    if print_train:
        iterator = tqdm(enumerate(dataloader))
    else:
        iterator = enumerate(dataloader)

    for ii,(features,labels) in iterator:
        
        # shuffle the training batch
        np.random.seed(12345)
        idx_shuffle         = np.random.choice(features.shape[0],features.shape[0],replace = False)
        features            = features[idx_shuffle]
        labels              = labels[idx_shuffle]
        
        # load the data to memory
        inputs      = Variable(features).to(device)
        labels      = labels.to(device)
        # one of the most important steps, reset the gradients
        optimizer.zero_grad()
        # compute the outputs
        outputs,_   = net(inputs)
        # compute the losses
        outputs = outputs
        loss_batch  = loss_func(outputs,labels)
        
        # add L2 loss to the weights
        if l2_lambda > 0:
            weight_norm = torch.norm(list(net.parameters())[-4],2)
            loss_batch  += l2_lambda * weight_norm
        # add L1 loss to the weights
        if l1_lambda > 0:
            weight_norm = torch.norm(list(net.parameters())[-4],1)
            loss_batch  += l1_lambda * weight_norm
        
        # backpropagation
        loss_batch.backward()
        # modify the weights
        optimizer.step()
        # record the training loss of a mini-batch
        train_loss  += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
            
    return train_loss/(ii+1)

def validation_loop(net,
                    loss_func,
                    dataloader,
                    device,
                    verbose = 0,
                    ):
    """
    net:nn.Module, torch model object
    loss_func:nn.Module, loss function
    dataloader:torch.data.DataLoader
    device:str or torch.device
    categorical:Boolean, whether to one-hot labels
    output_activation:string, calling the activation function from an inner dictionary

    """
    from tqdm import tqdm
    # specify the gradient being frozen and dropout etc layers will be turned off
    net.to(device).eval()
    with no_grad():
        valid_loss      = 0.
        if verbose == 0:
            iterator = enumerate(dataloader)
        else:
            iterator    = tqdm(enumerate(dataloader))
        for ii,(features,labels) in iterator:
            # load the data to memory
            inputs      = Variable(features).to(device)
            labels      = labels.to(device)
            # compute the outputs
            outputs,feature_   = net(inputs)
            # compute the losses
            loss_batch  = loss_func(outputs,labels)
            # record the validation loss of a mini-batch
            valid_loss  += loss_batch.data
            denominator = ii
            
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss

def train_and_validation(
        model_to_train,
        f_name,
        loss_func,
        optimizer,
        image_resize    = 128,
        device          = 'cpu',
        batch_size      = 8,
        n_epochs        = int(3e3),
        print_train     = True,
        patience        = 5,
        train_root      = '',
        valid_root      = '',
        noise_level     = None,
        l1_term         = 0,
        l2_term         = 0,
        ):
    """
    This function is to train a new CNN model on clear images
    
    The training and validation processes should be modified accordingly if 
    new modules (i.e., a secondary network) are added to the model
    
    Arguments
    ---------------
    model_to_train:torch.nn.Module
        a nn.Module class
    f_name:string
        the name of the model that is to be trained
    output_activation:torch.nn.activation
        the activation function that is used
        to apply non-linearity to the output layer
    loss_func:torch.nn.modules.loss
        loss function
    optimizer:torch.optim
        optimizer
    image_resize:int, default = 128
        the number of pixels per axis for the image
        to be resized to
    device:string or torch.device
        default = "cpu", where to train model
    batch_size:int, default = 8,
        batch size
    n_epochs:int, default = int(3e3)
        the maximum number of epochs for training
    print_train:bool, default = True
        whether to show verbose information
    patience:int, default = 5
        the number of epochs the model is continuely trained
        when the validation loss does not change
    train_root:string, default = ''
        the directory of data for training
    valid_root:string, default = ''
        the directory of data for validation
    
    Output
    -----------------
    model_to_train:torch.nn.Module, a nn.Module class
    """
    augmentations = {
            'train':simple_augmentations(image_resize,noise_level = noise_level),
            'valid':simple_augmentations(image_resize,noise_level = noise_level),
        }
    
    train_loader        = data_loader(
            train_root,
            augmentations   = augmentations['train'],
            batch_size      = batch_size,
            )
    valid_loader        = data_loader(
            valid_root,
            augmentations   = augmentations['valid'],
            batch_size      = batch_size,
            )
    
    
    model_to_train.to(device)
    model_parameters    = filter(lambda p: p.requires_grad, model_to_train.parameters())
    if print_train:
        params          = sum([np.prod(p.size()) for p in model_parameters])
        print(f'total params: {params:d}')
    
    best_valid_loss     = torch.tensor(float('inf'),dtype = torch.float64)
    losses = []
    for idx_epoch in range(n_epochs):
        # train
        print('\ntraining ...')
        _               = train_loop(
        net                 = model_to_train,
        loss_func           = loss_func,
        optimizer           = optimizer,
        dataloader          = train_loader,
        device              = device,
        idx_epoch           = idx_epoch,
        print_train         = print_train,
        l1_lambda           = l1_term,
        l2_lambda           = l2_term,
        )
        print('\nvalidating ...')
        valid_loss= validation_loop(
        net                 = model_to_train,
        loss_func           = loss_func,
        dataloader          = valid_loader,
        device              = device,
        )
        
        print(f'\nepoch {idx_epoch + 1}, loss = {valid_loss:6f}')
        if valid_loss.cpu().clone().detach().type(torch.float64) < best_valid_loss:
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(model_to_train,f_name)
        else:
            model_to_train = torch.load(f_name)
        losses.append(best_valid_loss)
    
        if (len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    return model_to_train

def extract_cv_features(net,
                        image_dir,
                        image_resize = 128,
                        noise_level = None,
                        do_augmentations = False,
                        ):
    from tqdm import tqdm
    augmentations = define_augmentations(image_resize       = image_resize,
                                         noise_level        = noise_level,
                                         do_augmentations   = False)
    image_loader = data_loader(
        data_root               = image_dir,
        augmentations           = augmentations['valid'],
        batch_size              = 1,
        num_workers             = 1,
        shuffle                 = False,
        return_path             = False,
        )
    features = []
    df = dict(labels = [],
              targets = [])
    for i, (images,labels) in tqdm(enumerate(image_loader),desc = 'extracting'):
        with torch.no_grad():
            path = image_loader.dataset.samples[i][0]
            _,_,_,target,label,image_name = path.split('/')
            
            feature,_ = net(images)
            feature = feature.detach().numpy().flatten()
            features.append(feature)
            
            df['labels'].append(label.lower())
            df['targets'].append(target.lower())
    df = pd.DataFrame(df)
    features = np.array(features)
    return df,features