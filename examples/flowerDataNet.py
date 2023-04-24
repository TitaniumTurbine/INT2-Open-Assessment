import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor ,Lambda
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# This transform takes all of the input images and resizes them all

class reziseImages(object): 
	def __init__(self, size):
		self.size = size
	def __call__(self, inList):
		tempTrans = T.Resize(self.size)
		transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
		newImg = tempTrans(inList)
		newTensor = transform(newImg)
		return newTensor.float()



myTrans = reziseImages((200,200))
flowersTrainingData = datasets.Flowers102(
	root="flowerData",
	split = "train",
	download= True,
	transform=myTrans,
	target_transform=Lambda(lambda y: torch.zeros(102, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

)

flowersTestingData = datasets.Flowers102(
	root="flowerData",
	split = "test",
	download= True,
	transform=myTrans,

)

#Defining the neural network

class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(40000,10000),
			nn.ReLU(),
			nn.Linear(10000,10000),
			nn.ReLU(),
			nn.Linear(10000,102)
		)
	def forward(self,x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
model = NeuralNet().to(device)

def trainNet(dataLoader, model, loss_fn ,optimizer):
	for batch, (x,y) in enumerate(dataLoader):
		print(x.shape)
		pred = model(x)
		loss = loss_fn(pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if batch % 50 == 0:
			#print("my loss is:" + str( loss))


def testNet(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model (X)
            test_loss = loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    



learningRate = 1e-5
batchSize = 64
epochs = 2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
train_dataloader = DataLoader(flowersTrainingData, batch_size=64, shuffle=True)
test_dataloader = DataLoader(flowersTestingData, batch_size=64, shuffle=True)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainNet(train_dataloader, model, loss_fn, optimizer)
    testNet(test_dataloader, model, loss_fn)
