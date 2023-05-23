class Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3 , stride=1 , padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2 , stride=2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(8192 , 526),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5))
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(526 , 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5))
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(128 , num_classes))

    def forward(self , x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out