import yfinance as yf 
import mplfinance as mpf
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np

apple = "AAPL"
stock_data = yf.Ticker(apple)
stock_data_hist = stock_data.history(start="2011-01-02", end="2023-12-31")
# print(pd.DataFrame(stock_data_hist[['Open', 'Close']]))

data = pd.DataFrame(stock_data_hist[['Open', 'Close']])
data.index = pd.to_datetime(data.index)
store = data.index

number_lags = 70 #hyperparameter --- 1

indices_train = np.array(store[number_lags : int((data.shape[0]*4)/5)])
indices_test = np.array(store[int((data.shape[0]*4)/5) : data.shape[0]])
print(len(indices_train))
print(len(indices_test))

# Plotting candlestick chart
mpf.plot(stock_data_hist,
         type='candle',
         style='charles',
         volume=True,
         ylabel='Price',
         ylabel_lower='Volume',
         figsize=(20,12))
        #  mav=(50, 150, 200))
plt.show()

#required scaling of return prices
scaler = MinMaxScaler()
data.iloc[:,0:] = scaler.fit_transform(data.iloc[:,0:])
# print(data.iloc[:,0:])


x_train = []
y_train = []

x_test = []
y_test = []

for i in range(number_lags , int((data.shape[0]*4)/5)):
        x_train.append(data.iloc[i - number_lags : i,0])
        y_train.append(data.iloc[i,0])
        
for i in range(int((data.shape[0]*4)/5) , data.shape[0]):
        x_test.append(data.iloc[i - number_lags : i,0])
        y_test.append(data.iloc[i,0])
        
x_train = torch.tensor(x_train, dtype = torch.float32)
x_train = x_train.view(x_train.shape[0],x_train.shape[1],1)
y_train = torch.tensor(y_train, dtype = torch.float32)
y_train = y_train.view(-1,1)

x_test = torch.tensor(x_test , dtype = torch.float32)
x_test = x_test.view(x_test.shape[0],x_test.shape[1],1)
y_test = torch.tensor(y_test , dtype = torch.float32)
y_test = y_test.view(-1,1)

class stock_data(Dataset):
        def __init__(self):
                self.x  = x_train
                self.y  = y_train
                self.len = x_train.shape[0]
                
        def __getitem__(self,index):  
                return self.x[index], self.y[index]
        
        def __len__(self):
                return self.len
        
class stock_data_test(Dataset):
        def __init__(self):
                self.x  = x_test
                self.y  = y_test
                self.len = x_test.shape[0]
                
        def __getitem__(self,index):
                return self.x[index], self.y[index]
        
        def __len__(self):
                return self.len


prices = stock_data()
test = stock_data_test()

n_epochs = 28
batch_size = 5
input_size = 1
hidden_size = 32
num_layers = 1
output_size = 1
learning_rate = 0.02
# dropout_rate = 0.3

# print(input_size)
# print(output_size)

train_loader = DataLoader(dataset = prices, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test, batch_size = batch_size, shuffle = False)

okk,now = next(iter(train_loader))
# print(okk.shape)
# print(now.shape)

class stock_predictor(nn.Module):
        def __init__(self,input_size,num_layers,hidden_size,output_size):
                super(stock_predictor,self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers , batch_first = True)
                # self.dropout = nn.Dropout(dropout_rate)
                self.linear = nn.Linear(hidden_size,output_size)
                
        def forward(self,x):
                batch_size = x.size(0)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                x,_ = self.lstm(x, (h0,c0)) # _ is modified (h0,c0)
                # x = self.dropout(x)
                x = self.linear(x[:,-1,:])
                return x
                
                
model = stock_predictor(input_size,num_layers,hidden_size,output_size)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss = nn.MSELoss()

for epoch in range(n_epochs):
        running_loss = 0
        for i,(data,labels) in enumerate(train_loader):
                y_prediction = model(data)
                losses = loss(y_prediction, labels)
                running_loss += losses
                
                losses.backward()
                optimizer.step()
                
                optimizer.zero_grad()
                
                if(i%2 == 0):
                        print(f"Epochs {epoch + 1}/{n_epochs} : {i+1}/{len(train_loader)} = loss : {running_loss:.3f}")
                        running_loss = 0
                        
with torch.no_grad():
        for i,(data,labels) in enumerate(test_loader):    
                y_pred = model(data)
                losses = loss(y_pred, labels)
               
        print(f"Loss on testing model : {losses:.3f}") 
                               
     
result_train = model(x_train).squeeze().detach().numpy()
result_test = model(x_test).squeeze().detach().numpy()
print(result_train.shape)
real_train = y_train.squeeze().detach().numpy()
real_test = y_test.squeeze().detach().numpy()

fig,ax = plt.subplots(figsize = (12,12),nrows = 2)
ax[0].plot(indices_train,real_train, color = "blue")
ax[0].plot(indices_train,result_train, color = "red")

ax[1].plot(indices_test,real_test, color = "blue")
ax[1].plot(indices_test, result_test, color = "red")

plt.show()



                
                






