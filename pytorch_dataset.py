#%%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self,file_name, label_col):
        super().__init__()
        self.label_col=label_col
        self.data=pd.read_csv(file_name).select_dtypes(exclude=[object])
        column_to_move = self.data.pop(label_col)
        self.data.insert(0, label_col, column_to_move)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example=self.data.iloc[idx]
        features=torch.tensor(example[1:10])
        label=example[0]
        return features, label


mydata=AirbnbNightlyPriceImageDataset('tabular_data/clean_tabular_data.csv', 'Price_Night')

train_count = int(0.7 *mydata.__len__())
valid_count = int(0.2*mydata.__len__())
test_count = mydata.__len__ ()- train_count - valid_count
train_set, valid_set, test_set = random_split(mydata, (train_count, valid_count, test_count))

batch_size=4
train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=True)


class LinearRegression(torch.nn.Module):
    def __init__(self)->None:
        super().__init__()
        #initialise parameters
        self.linear_layer=torch.nn.Linear(9,1)
        self.double()

     

    def forward(self, features):
    #use layers to process features
        return self.linear_layer(features)


model=LinearRegression()

def train(model, epochs=10):
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            predicted=model(features)
            loss=F.mse_loss(predicted, labels)
            loss.backward()
            print(loss)
            #optimization

train(model, epochs=5)


#%%