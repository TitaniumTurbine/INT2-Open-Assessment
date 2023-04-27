import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
import ssl
from torch.utils.data import ConcatDataset
ssl._create_default_https_context = ssl._create_unverified_context
img_size = (325, 325)
training_data = datasets.Flowers102(root="../flowerData", split="train", download=True, transform=transforms.Compose([transforms.Resize(img_size), ToTensor()]))
test_data = datasets.Flowers102(root="../flowerData", split="test", download=True, transform=transforms.Compose([transforms.Resize(img_size), ToTensor()]))


trainingSet2 = datasets.Flowers102(root="../flowerData", split="test", download=True, transform=transforms.Compose([transforms.Resize(img_size), ToTensor(), torchvision.transforms.ColorJitter(hue=.05, saturation=.05), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(20)]))
trainingSet3 = datasets.Flowers102(root="../flowerData", split="test", download=True, transform=transforms.Compose([transforms.Resize(img_size), ToTensor(), torchvision.transforms.ColorJitter(hue=.02, saturation=.02), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(20), torchvision.transforms.Grayscale(num_output_channels=3)]))

print(training_data)
print("shaope 1")
print(test_data)
print("shape2")
print(trainingSet2)
megaSet =  ConcatDataset([test_data, trainingSet2, trainingSet3])

print("shape 3")
print(megaSet)
labels_map = {
    
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze().permute(1,2,0))
plt.show()
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.BatchNorm2d(100),
            
            nn.Conv2d(100, 200, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8)),
            nn.BatchNorm2d(200),
            
            nn.Flatten(),
    
            nn.Linear(16200, 4000),
            nn.ReLU(),
    
            nn.Linear(4000, 102)
        )
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.layers(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
        #if batch % 200 == 0:
            #loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")
    print(f"Avg train loss: {(total_loss/len(dataloader)):>8f}")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
learning_rate = 2.2e-2
batch_size = 65
epochs = 1000
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1, weight_decay=0)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(megaSet, batch_size=batch_size, shuffle=True)
print(len(training_data))
print(len(test_dataloader))
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(test_dataloader, model, loss_fn, optimiser)
	if (True):
		test_loop(train_dataloader, model, loss_fn)
		nAME = 'model_weights' + str(t) + ".pth"
		torch.save(model.state_dict(), 'model_weights.pth')
	if (t % 3 == 0):
		learning_rate * 0.9
print("Done!")