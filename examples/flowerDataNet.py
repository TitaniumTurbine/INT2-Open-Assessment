import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor ,Lambda
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn

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
	transform=ToTensor(),

)

#Defining the neural network

class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(40000,40000),
			nn.ReLU(),
			nn.Linear(40000,40000),
			nn.ReLU(),
			nn.Linear(40000,102)
		)
	def forward(self,x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
model = NeuralNet()

def trainNet(dataLoader, model, lossFunc ,optimizer):
	for batch, (x,y) in enumerate(dataLoader):
		print(x.shape)
		pred = model(x)
		loss = loss_fn(pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("my loss is:" + str( loss))






learningRate = 1e-5
batchSize = 64
epochs = 2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
train_dataloader = DataLoader(flowersTrainingData, batch_size=64, shuffle=True)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainNet(train_dataloader, model, loss_fn, optimizer)