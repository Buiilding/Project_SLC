import torch
import os
from Classification_Model import Model
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F

def inference(model, weight_path, uploaded_file, num_class):
    model = Model(num_classes=num_class)
    # initialize device
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_statedict = torch.load(weight_path, map_location='cpu')
    # mapping statedict to model
    model.load_state_dict(model_statedict)
    # send model to device
    model = model.to(device)
    # set the model to evaluation mode
    model.eval()
    
    # Reset the position of the uploaded file
    uploaded_file.seek(0)
    # Read the contents of the uploaded file into memory
    file_contents = uploaded_file.read()
    print(f"File contents: {file_contents}")
    file_bytes = np.asarray(bytearray(file_contents), dtype=np.uint8)
    print(f"File bytes: {file_bytes}")
    # Create an OpenCV image from the file bytes
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # display the image
    # plt.imshow(image_RGB)
    
    # Create a PIL Image from the OpenCV image
    image_pil = Image.fromarray(image_RGB)
    
    # Resize the image to a square shape
    size = max(image_pil.size)
    image_resized = image_pil.resize((size, size))
    
    data_transform = transforms.Compose((
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225], )
    ))
    # Apply the transformations and get the image as a tensor
    image_tensor = data_transform(image_resized)
    # change 3 dim to 4 dim before inference
    im = image_tensor.unsqueeze(0)
    # Pass the image through the model
    im = im.to(device)
    output = model(im)
    # apply softmax
    output_softmax = F.softmax(output, dim=1) # because output shape is 1 , 27  which is by column and to calculate column, dim needs to be dim  = 1
    # print output
    print(output_softmax)
    top_k_probs, top_k_classes = torch.topk(output_softmax, k=1)
    print(f'top_k_probs : {top_k_probs} \ntop_k_classes : {top_k_classes}')
    return top_k_classes
