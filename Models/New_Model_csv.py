import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract image data from DataFrame
        image_data = self.data.iloc[idx, 1:].values

        image_data = image_data.reshape(28, 28)

        # Convert data type from int64 to uint8
        img = np.uint8(image_data)
        #using opencv to resize img from 28x28 to 64x64
        img_resize = cv2.resize(img,(64,64))
        #using merge to clone img_resize from hwx1 to hwx3 
        img_extend = cv2.merge((img_resize,img_resize,img_resize))
        # save image:
        # cv2.imwrite('/content/sample_data/img.jpg', img_extend)
        #convert img_extend from opencv to Pil image using Image.fromarray      
        #image = Image.fromarray(image_data.astype('uint8'), 'L') #convert data
        image = Image.fromarray(img_extend) #convert to Pil image

model = SignLanguageDataset()

def Kaggle_dataset_Model(model,img,save_best_model, train_path, test_path, model):
    #Preprocess data
    model = model(num_classes = 25)
    label = img['label'].values
    image_data = img.iloc[ 0, 1:].values
    image_data = image_data.reshape(28, 28)
    img = np.uint8(image_data)
    img_resize = cv2.resize(img,(64,64))
    img_extend = cv2.merge((img_resize,img_resize,img_resize))
    #convert image fromm height width channel to channel height width
    img_chw = img_extend.transpose(2,0,1)
    img_extend = cv2.cvtColor(img_extend, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_extend) #convert to Pil image
    best_acc = 0.0
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456 , 0.406], std=[0.229, 0.224, 0.225])
    ])
        
        #TRAIN MODEL

        # Apply transformation (if provided)
        if self.transform:
            image = self.transform(image)

        # Extract label
        label = self.data.iloc[idx, 0]

        return image, label

        # Create datasets
    train_dataset = SignLanguageDataset(train_data, transform=transform)
    test_dataset = SignLanguageDataset(test_data, transform=transform)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Initialize model
    model = model.to(device)


    # Set loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    num_epochs = 100
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(f'output={outputs}, labels = {labels}')
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # Evaluate
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data)

        # Calculate average losses and accuracies
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)
        test_acc = test_correct.double() / len(test_loader.dataset)
        if test_acc > best_acc:
          best_acc = test_acc
          torch.save(model.state_dict(), save_best_model)

        # Print results
        a = print(f'Epoch: {epoch+1}/{num_epochs}')
        b = print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        c = print(f'Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}')
        return a,b,c

    
  
    
    
