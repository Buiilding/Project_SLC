import torch #operated by facebook with a lot of library
import torch.nn as nn
#import tqdm
class Model(nn.Module):
    def __init__(self, num_classes = 10 ) :
        # super(AlexNet, self). __init__()
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size = 3 , stride = 1, padding=1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride =1 , padding = 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 3, stride =1 , padding = 1),
            # nn.BatchNorm2d(384), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 3 , stride = 1, padding =1),
            # nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2 ))
        self.flatten = nn.Flatten()
        self.fc1 =nn. Sequential( 
            nn.Linear(8192,526),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(526,128),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3= nn.Sequential(
            nn.Linear(128, num_classes))
        
    def forward(self,x):
            out = self.conv_1(x)
            out = self.conv_2(out)
            out = self.conv_3(out)
            out = self.conv_4(out)
            out = self.flatten(out)
            # out = out.reshape(out.size(0), -1 )#flatten
            out = self.fc1(out)
            out = self.fc2 (out)
            out = self.fc3(out)
            #out = nn.Softmax(out)
            return out
def unit_test(b,c,h,w):
    model = Model(num_classes = 27)
    im_input = torch.randn((b,c,w,h))
    out = model(im_input)
    print(f'chick qua qua')
    return out
        