import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
import random
import sys
import DSN_model
import DANN_Dataset

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def tSNE_img(F_model,L_model,image_name,root_path,pred_file):
    img = cv2.imread(os.path.join(root_path, image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img)
    with torch.no_grad():
        img = Variable(img[:,:,:])
    img = img.cuda()
    img = torch.unsqueeze(img,0)
    tmp_output = F_model(img)
    output = L_model(tmp_output)
    _, output = torch.max(output,1)
    output = output.cpu()
    image_name = image_name.replace("png","png")
    text = "%s,%d" %(image_name,output[0])
    # print(text)
    file.write(text+"\n")
    return True

def mkdir(path):

    path=path.strip()
    path=path.rstrip("\\")

    isExists=os.path.exists(path)

    if not isExists:
        print (path,"successful")
        os.makedirs(path)
        return True
    else:
        print("-"*10)
        print ("Directory already exists.Please remove it.")
        print("-"*10)
        return False

def main():

    t_root_path = sys.argv[1]
    s_root_path = sys.argv[2]

    t_test_dataset = DANN_Dataset.DANN_Dataset(filepath = t_root_path ,csvpath = t_root_path+".csv")
    t_test_loader = DataLoader(t_test_dataset ,batch_size = 1 ,shuffle=False )
    s_test_dataset = DANN_Dataset.DANN_Dataset(filepath = s_root_path ,csvpath = s_root_path+".csv")
    s_test_loader = DataLoader(s_test_dataset ,batch_size = 1 ,shuffle=False )

    net = DSN_model.DSN()

    print('load model...')
    F_model_path = sys.argv[4]
    load_checkpoint(F_model_path,net)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    net.eval()

    pred_file = sys.argv[3]

    iter = 1
    print("creat the log file...")
    mkdir(pred_file)
    print('predicting...')

    source_array = []
    s_label_array = []
    target_array = []
    t_label_array = []
    cut_num = 2000
    print('the dataset has %d size.' % (len(s_test_loader)))
    for i, (s_data,t_data) in enumerate(zip(s_test_loader,t_test_loader)):
        if i == cut_num:
            break
        print("\r%d/%d" %(i,cut_num),end = "")
        s_img, s_label = s_data
        t_img, t_label = t_data
        s_img = Variable(s_img).to(device)
        t_img = Variable(t_img).to(device)
        tmp_output = net(s_img,'source','share')
        privte_feature, share_feature, domain_label, class_label, recon_feature = tmp_output

        s_feature = share_feature.data.cpu().numpy().reshape(-1)

        tmp_output = net(t_img,'source','share')
        privte_feature, share_feature, domain_label, class_label, recon_feature = tmp_output

        t_feature = share_feature.data.cpu().numpy().reshape(-1)
        source_array.append(s_feature)
        s_label_array.append(s_label)
        target_array.append(t_feature)
        t_label_array.append(t_label)

    source_array = np.array(source_array)
    s_label_array = np.array(s_label_array)
    target_array = np.array(target_array)
    t_label_array = np.array(t_label_array)

    print("")
    print("Load data finished!")
    print("test_array : ",source_array.shape)

    tSNE = manifold.TSNE(n_components = 2, init = 'pca', random_state = 312)
    s_img_tSNE = tSNE.fit_transform(source_array)
    t_img_tSNE = tSNE.fit_transform(target_array)
    print("Org data dimension is %d , tSNE data dimension is %d" %(source_array.shape[-1], s_img_tSNE.shape[-1]))
    x_min, x_max = s_img_tSNE.min(0), s_img_tSNE.max(0)
    s_img_norm = (s_img_tSNE - x_min) / (x_max - x_min)
    x_min, x_max = t_img_tSNE.min(0), t_img_tSNE.max(0)
    t_img_norm = (t_img_tSNE - x_min) / (x_max - x_min)
    # t_img_norm = 1. - t_img_norm
    print("img_norm : ",s_img_norm.shape)
    plt.figure(figsize=(16, 16))
    for i in range(t_img_norm.shape[0]):
        plt.text(t_img_norm[i, 0], t_img_norm[i, 1], str(t_label_array[i]), color = plt.cm.Set1(t_label_array[i]),
             fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(pred_file,"DANN_tSNE_target.png"))

    # source
    plt.figure(figsize=(16, 16))
    for i in range(s_img_norm.shape[0]):
        plt.text(s_img_norm[i, 0], s_img_norm[i, 1], str(s_label_array[i]), color = plt.cm.Set1(s_label_array[i]),
             fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(pred_file,"DANN_tSNE_source.png"))

    # figure 2
    plt.figure(figsize=(16, 16))
    for i in range(s_img_norm.shape[0]):
        plt.text(s_img_norm[i, 0], s_img_norm[i, 1], str(s_label_array[i]), color = "blue",
             fontdict={'weight': 'bold', 'size': 10})
        plt.text(t_img_norm[i, 0], t_img_norm[i, 1], str(t_label_array[i]), color = "red",
             fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(pred_file,"DANN_tSNE_st.png"))


if __name__ == '__main__':
    main()
