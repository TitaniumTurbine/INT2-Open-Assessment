import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchviz import make_dot


class CNN(nn.Module):
    def __init__(self, num_classes=102):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(8, 8)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512 * 8 * 8, 2 ** 12)
        self.fc2 = nn.Linear(2 ** 12, num_classes)

    def forward(self, x):
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
        x = x.view(-1, 512 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_and_validate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, writer,
                       epochs=100):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            labels = labels - 1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        scheduler.step()
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", 100 * correct_train / total_train, epoch)
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                labels = labels - 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        accuracy_val = 100 * correct / total
        accuracy_train = 100 * correct_train / total_train
        writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/val", accuracy_val, epoch)

        # Save model and print accuracy_val
        torch.save(model.state_dict(), f"flowers102_epoch_{epoch + 1}.pth")
        print(f"Epoch {epoch + 1}: Validation accuracy = {accuracy_val:.2f}%")
        print(f"         Train accuracy = {accuracy_train:.2f}%")
        # if epoch % 10 == 0:
        #     ac_cal(model, test_loader, device)


# Testing function
def ac_cal(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # change lables
            labels = labels - 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"         Test accuracy = {(100 * correct / total):.2f}%")


if __name__ == "__main__":
    # Transform with data augmentation
    Transform1 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])
    Transform2 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.05, saturation=0.05),
        transforms.ToTensor()
    ])

    train_data_list = [datasets.Flowers102(root="./dataset", split='train', transform=Transform1, download=True)]
    train_data_list.extend(
        [datasets.Flowers102(root="./dataset", split='train', transform=Transform2, download=True) for _ in range(3)])

    val_data = datasets.Flowers102(root="./dataset", split='val', transform=Transform1, download=True)
    test_data = datasets.Flowers102(root="./dataset", split='test', transform=Transform1, download=True)
    train_data = ConcatDataset(train_data_list)

    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.1)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # TensorBoard writer
    writer = SummaryWriter()

    # Training and validation function
    train_and_validate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, writer)
    ac_cal(model, test_loader, device)
    writer.close()
