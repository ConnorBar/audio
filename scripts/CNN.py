from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from data.constants import *
from models.BasicCNN import ChineseToneCNN, ChineseTonesDataset

# Load in data
X = np.load(os.path.join(DATA_DIR, 'features.npy'))
y = np.load(os.path.join(DATA_DIR, 'labels.npy'))

X = torch.FloatTensor(X)
y = torch.LongTensor(y)
  
# CNN's want (batch_size, channels, height, width)
# so we make it to be: (batch_size, channels, frequency_bands, time_steps).
X = X.unsqueeze(1)
X = X.permute(0, 1, 3, 2)

# Data Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=True)

# Hyperparameters
num_classes = len(torch.unique(y)) # is just 4
learning_rate = 0.001
momentum= 0.9
num_epochs = 50
batch_size = 32

# Create data loaders
train_dataset = ChineseTonesDataset(X_train, y_train)
val_dataset = ChineseTonesDataset(X_val, y_val)
test_dataset = ChineseTonesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# init model, loss function, and optimizer
model = ChineseToneCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# set device 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def train_one_epoch(epoch_index, tb_writer):
  running_loss = 0
  last_loss = 0
  
  for i, data in enumerate(train_loader):
    X_batch, y_batch = data
    
    # zero grads every batch
    optimizer.zero_grad()
    # inference & calc loss
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    
    # back prop & adjust learning weights
    loss.backward()
    optimizer.step()
    
    # report loss
    running_loss += loss.item()
    if i % 1000 == 999:
      last_loss = running_loss / 1000 # loss per bacth
      print('  batch {} loss: {}'.format(i + 1, last_loss))
      tb_x = epoch_index * len(train_loader) + i + 1
      tb_writer.add_scaler('Loss/train', last_loss,  tb_x)
      running_loss = 0
    # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader)}")


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0
best_vloss = 1_000_000.

# Training loop
for epoch in range(num_epochs):
  print('EPOCH {}:'.format(epoch + 1))
  
  model.train(True)
  avg_loss = train_one_epoch(epoch, writer)
  running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
  model.eval()
  with torch.no_grad():
      for i, vdata in enumerate(val_loader):
          vinputs, vlabels = vdata
          voutputs = model(vinputs)
          vloss = criterion(voutputs, vlabels)
          running_vloss += vloss

  avg_vloss = running_vloss / (i + 1)
  print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

  # Log the running loss averaged per batch
  # for both training and validation
  writer.add_scalars('Training vs. Validation Loss',
                  { 'Training' : avg_loss, 'Validation' : avg_vloss },
                  epoch_number + 1)
  writer.flush()

  # Track best performance, and save the model's state
  if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      model_path = 'model_{}_{}'.format(timestamp, epoch_number)
      torch.save(model.state_dict(), model_path)

  epoch_number += 1

    

