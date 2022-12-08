from torch.optim import Adam
import argparse
import torch.nn as nn 
import util
import numpy as np
import os 
import torch
from torch.utils.data import DataLoader,TensorDataset
from model import TCPredict


def rmse(prediction,truth):
    return torch.sqrt(torch.mean((prediction - truth)**2))
 
def train(model,loader,optimizer,loss_fn,batch_size):
    model.train()
    running_loss = 0.0
    for batch_idx,(X,y) in enumerate(loader):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/(len(loader)*batch_size)

def val(model,loader,loss_fn):
    model.eval()
    running_loss = 0.0
    for _,(X,y) in enumerate(loader):
        out = model(X)
        loss = loss_fn(out,y)
        running_loss += loss.item()
    return running_loss/len(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Train the TCPredict model"
)
    parser.add_argument("output_dir",type=str)
    parser.add_argument("experiment_name",type=str)
    parser.add_argument("--timepoints",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--fc-width",type=int,default=50)
    args = parser.parse_args()

    ibtracs_ds = util.get_dataset()
    sst_ds = util.get_sst_ds()
    train_storms,valid_storms,test_storms = util.train_validation_test(ibtracs_ds,seed=42)

    X_train,y_train,group_train = util.make_X_y(ibtracs_ds,sst_ds,train_storms,timesteps=args.timepoints)
    X_validation,y_validation,group_val = util.make_X_y(ibtracs_ds,sst_ds,valid_storms,timesteps=args.timepoints)

    X_train = np.transpose(X_train,axes=(0,2,1))
    X_validation = np.transpose(X_validation,axes=(0,2,1))
    ### Data cleansing

    model = TCPredict(initial_timesteps=args.timepoints,fc_width=args.fc_width)
    model = model.cuda().double()


    train_dataset= TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=16)

    validation_dataset = TensorDataset(torch.from_numpy(X_validation),torch.from_numpy(y_validation))
    validation_loader = DataLoader(validation_dataset)

    optimizer = Adam(model.parameters(),lr=3e-5)

    for e in range(args.epochs):
        print(f"---Epoch {e} ---")
        train_loss = train(model,train_dataset,optimizer,rmse,batch_size=16)
        val_loss = val(model,validation_dataset,rmse)
        print(f"Training Loss: {train_loss:.3f}")
        print(f"Validation Loss:{val_loss:.3f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir,args.experiment_name+'.model'))


