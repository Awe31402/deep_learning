import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# model definition
model = torchvision.models.resnet50(pretrained=False, num_classes=10)
model.to(device)

if device == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scaler = GradScaler()

# training loop
def train(epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.3f}')

    end_time = time.time()
    print(f'Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds, Accuracy: {100.*correct/total:.2f}%')

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print(f'Test Loss after epoch {epoch+1}: {test_loss/len(testloader):.3f}')
    return acc

best_acc = 0
for epoch in range(100):
    start_time = time.time()
    train(epoch)
    acc = test(epoch)
    scheduler.step()

    if acc > best_acc:
        print(f'New best accuracy: {acc:.2f}% (previous: {best_acc:.2f}%)')
        best_acc = acc
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_model.pth')
    print(f'Best accuracy so far: {best_acc:.2f}%')

checkpoint = torch.load('./checkpoint/best_model.pth')
model.load_state_dict(checkpoint['model'])
final_acc = test(checkpoint['epoch'])
print(f'Final accuracy: {final_acc:.2f}%')