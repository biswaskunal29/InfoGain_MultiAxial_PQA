import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F



class train_dataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
#        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
#        self.n_samples = xy.shape[0]
#
#        # here the first column is the class label, the rest are the features
#        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
#        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        
#        X = np.load('X_train_v3.npy')
#        y = np.load('y_train_v3.npy')
        
        X_double = np.load('full_X_train_30split.npy')
        X = X_double.astype(np.single)
        y_double = np.load('full_y_train_30split.npy')
        y = y_double.astype(np.single)
        
        self.n_samples = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)
#        self.x_data = self.x_data.to(torch.device("cuda"), dtype = torch.float)
#        self.y_data = self.y_data.to(torch.device("cuda"), dtype = torch.float)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
class test_dataset(Dataset):

    def __init__(self):        
#        X = np.load('X_test_v3.npy')
#        y = np.load('y_test_v3.npy')
        
        X_double = np.load('full_X_test_30split.npy')
        X = X_double.astype(np.single)
        y_double = np.load('full_y_test_30split.npy')
        y = y_double.astype(np.single)
        
        self.n_samples = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)
#        self.x_data = self.x_data.to(torch.device("cuda"), dtype = torch.float)
#        self.y_data = self.y_data.to(torch.device("cuda"), dtype = torch.float)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# create dataset
train_dataset = train_dataset()
test_dataset = test_dataset()

# =============================================================================
# # get 4th sample and unpack
# X_curr,y_curr = dataset[4]
# print(X_curr, y_curr)
# =============================================================================

# hyper parameters
num_epochs = 5
total_samples = len(train_dataset)
batch_size = 64
n_iterations = math.ceil(total_samples/batch_size)
lr  = 0.02
lr2 = 0.0001
lr3 = 0.00001
input_size = 39072
hidden_size = 512 


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

# =============================================================================
# convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)
# =============================================================================

# =============================================================================
# examples = iter(test_loader)
# example_data, example_targets = examples.next()
# print(example_data.shape, example_targets.shape)
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(39072, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
#model = Net()
model = Net().to(device)
  
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model.parameters(), lr=lr2)
optimizer3 = torch.optim.Adam(model.parameters(), lr=lr3)


# =============================================================================
# Train the model
# =============================================================================

n_total_steps = len(train_loader)
#print(n_total_steps)
for epoch in range(num_epochs):
    for i, (X_curr, y_curr) in enumerate(train_loader): 
        
        X_curr = X_curr.reshape(-1, input_size).to(device)
        y_curr = y_curr.to(device)

        # Forward pass
        outputs = model(X_curr)        
        loss = criterion(outputs, y_curr)
        
        # Backward and optimize
        optimizer.zero_grad()
        optimizer2.zero_grad()        
        loss.backward()
        
# =============================================================================
#         if epoch >= 10 and epoch < 20:
#             optimizer2.step()
# #            print("Used optimiser 2")
#         elif epoch >= 20:
#             optimizer3.step()
# #            print("Used optimiser 3")
#         else :
#             optimizer.step()  
# #            print("Used optimiser 1")
# =============================================================================
        optimizer.step() 
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], \tStep [{i+1}/{n_total_steps}], \tLoss: {loss.item():.10f}')

# =============================================================================
# Test Model
# =============================================================================

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        X, y = data
        output = model(X.view(-1,input_size))
        #print(output)
        for idx, i in enumerate(output):
#            print(f"{i}\n{torch.argmax(i)}\n{torch.argmax(y[idx])}")
#            print(torch.argmax(i), y[idx])
            if torch.argmax(i) == torch.argmax(y[idx]):
                correct += 1
            total += 1

acc = 100.0 * correct / total
print("Accuracy: ", round(acc, 10))

# =============================================================================
# Save model
# =============================================================================

#torch.save(model.state_dict(), "big5model_bad.dat")
































