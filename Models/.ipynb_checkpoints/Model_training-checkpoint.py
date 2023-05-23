from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim

# Evaluation metrics
def accuracy(out, labels):
  _, pred = torch.max(out, dim=1)
  return torch.sum(pred==labels).item()
def train(model, num_epochs, optimizer, criterion, train_loader, val_loader, save_path):
    # Setting device for training
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  # train the model for 10 epochs
  best_acc = 0.0
  for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # move the data to the device (GPU or CPU)
        data, target = data.to(device), target.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # calculate the loss
        loss = criterion(output, target)

        # backward pass
        loss.backward()
        optimizer.step()

        # calculate training accuracy and loss
        train_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        train_correct += predicted.eq(target).sum().item()
        # print(f'train_loss {train_loss}')
        # print(f'train_correct {train_correct}')
        if batch_idx % 10 == 0:
          print(f'ITER [{batch_idx}] / [{len(train_loader)}] train loss = {loss.item() * data.size(0)}')
        
    # calculate average training accuracy and loss for the epoch
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    print(f'train_loss {train_loss}')
    print(f'train_acc {train_acc}')
    # evaluate the model on the test set
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # move the data to the device (GPU or CPU)
            data, target = data.cuda(), target.cuda()

            # forward pass
            output = model(data)

            # calculate the loss
            loss = criterion(output, target)

            # calculate test accuracy and loss
            test_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            test_correct += predicted.eq(target).sum().item()

    # calculate average test accuracy and loss for the epoch
    test_loss /= len(val_loader.dataset)
    test_acc = test_correct / len(val_loader.dataset)
    if test_acc > best_acc:
      best_acc = test_acc
      torch.save(model.state_dict(), save_path)
    # print the results for the epoch
    print('Epoch {}/{}: Train Loss: {:.6f}, Train Acc: {:.6f}, Test Loss: {:.6f}, Test Acc: {:.6f}'.format(
        epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))
