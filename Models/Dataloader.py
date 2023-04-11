#create dataloader
from torchvision import transforms, datasets

train_transform = transforms.Compose((
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),)
)
data_transform = transforms.Compose((
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456 , 0.406],
                         std =[0.229, 0.224, 0.225],)
))

# Intinialize dataset based-on folder path
train_dataset = datasets.ImageFolder(root='/file/path/', transform=train_transform)
# BTVN
val_dataset = datasets.ImageFolder(root = '/file/path/', transform = data_transform) #note val doesn't need RandomHorizontalFlip

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
# BTVN
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 8, num_workers = 4)
