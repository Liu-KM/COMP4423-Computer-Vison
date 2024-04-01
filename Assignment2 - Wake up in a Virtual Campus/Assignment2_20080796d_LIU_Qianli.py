import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchvision
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
import time
from tqdm import tqdm
# import Daloader
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score


#Step 1: Load the dataset
print("Step 1: Load the dataset")
start_time = time.time()
annotations = []
img_size = (224,224)
with open('./images/annotaions.jsonl') as f:
    for line in f:
        annotations.append(json.loads(line))

annotations = pd.DataFrame(annotations)

#Load the images
annotations['file_name'] = annotations['id'].apply(lambda x: f'./images/{x}.jpg')
def open_img(file_name):
    """
    Opens and preprocesses an image.

    Args:
        file_name (str): The file path of the image.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array, or None if the image cannot be opened.
    """
    try:
        img = Image.open(file_name)
        img = img.resize(img_size)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        return None
annotations['image'] = annotations['file_name'].apply(lambda x: open_img(x))
annotations = annotations.dropna()
annotations.drop('file_name', axis=1, inplace=True)
annotations.drop('id', axis=1, inplace=True)
annotations.rename(columns={'annotation':'label'}, inplace=True)
end_time = time.time()

print(f"Dataset loaded successfully, time used: {end_time-start_time:.2f} seconds")
print('The original dataset has {} images\n\n'.format(annotations.shape[0]))



#Step 2: Data Augmentation
print("Step 2: Data Augmentation")
start_time = time.time()
new_rows = []
for index,row in annotations.iterrows():
    img = row['image']
    label = row['label']
    img = Image.fromarray(img)
    #Horizontal flip
    h_flip_func = torchvision.transforms.RandomHorizontalFlip(p=1)
    # v_flip_func = torchvision.transforms.RandomVerticalFlip(p=1)
    shape_aug = torchvision.transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333))
    # color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    
    new_rows.append({'image':np.array(h_flip_func(img)), 'label':label})
    
    flip_img = h_flip_func(img)
    for _ in range(8):

        new_rows.append({'image':np.array(shape_aug(img)), 'label':label})
        new_rows.append({'image':np.array(shape_aug(flip_img)), 'label':label})
        #show the augmented image and original image in one plot
    if index == 0:
        fig, ax = plt.subplots(1,4)
        ax[0].imshow(img)
        ax[0].set_title('Original')
        ax[1].imshow(np.array(h_flip_func(img)))
        ax[1].set_title('Horizontal Flip')
        ax[2].imshow(np.array(shape_aug(img)))
        ax[2].set_title('Shape Augmentation')
        ax[3].imshow(np.array(shape_aug(flip_img)))
        ax[3].set_title('Shape Augmentation + Horizontal Flip')
        # plt.show()
        plt.savefig('augmented_images.png')

          
#Add the agumented data to the original dataset
new_rows = pd.DataFrame(new_rows)
annotations = pd.concat([annotations, new_rows], ignore_index=True)
# encode the label
annotations['label'] = annotations['label'].apply(lambda x: 0 if x == 'aiart' else 1)
end_time = time.time()

print(f"Data augmented successfully, time used: {end_time-start_time:.2f} seconds")
print('The augmented dataset has {} images'.format(annotations.shape[0]))
print('Demo of the augmented images is saved as \'augmented_images.png\'\n\n')

#Step 3: Prepare the dataloader
print("Step 3: Prepare the dataloader for training and validation")
start_time = time.time()
X = np.array(annotations['image'].tolist())
y = np.array(annotations['label'].tolist())

X = X/255
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
train_set = TensorDataset(torch.tensor(x_train,dtype=torch.float32),torch.tensor(y_train,dtype=torch.float32))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_set = TensorDataset(torch.tensor(x_test,dtype=torch.float32),torch.tensor(y_test,dtype=torch.float32))
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
end_time = time.time()
print(f"Dataloader prepared successfully, time used: {end_time-start_time:.2f} seconds\n\n")
print('The training set has {} images'.format(len(train_set)))
print('The validation set has {} images\n\n'.format(len(test_set)))

#Step 4: Define the model
print("Step 4: Define the model")
models = {}
alexnet = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 2))

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 2))

vgg_11 = vgg(conv_arch)

# models['Alexnet'] = alexnet
models['VGG11'] = vgg_11

print("Alexnet:\n",alexnet)
print('\n\n')

#Step 5: Train the model
print("Step 5: Train the model")
def train_model(net, train_loader,lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
def evaluate_model(net, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        f1 = 0
        recall = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images.permute(0, 3, 1, 2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 += f1_score(labels.cpu(), predicted.cpu())
            recall += recall_score(labels.cpu(), predicted.cpu())
        accuracy = correct / total
        f1 = f1 / len(test_loader)
        recall = recall / len(test_loader)
        print(f'Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Recall: {recall:.2f}')
        return accuracy, f1, recall
eval_results = {}
for model_name in models.keys():
    train_model(models[model_name], train_loader)
    accuracy, f1, recall = evaluate_model(models[model_name], test_loader)
    eval_results[model_name] = {'accuracy': accuracy, 'f1': f1, 'recall': recall}
end_time = time.time()

# print(f"Model trained successfully, time used: {end_time-start_time:.2f} seconds\n\n")

