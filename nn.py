import pdb
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cv_model import col_split, col_sample

def network(input_dim):
    # simple 2 layer network
    hidden_layers = [10, 2]
    m = nn.Sequential(
        nn.Linear(input_dim, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.LogSoftmax()
    )
    return m

def print_status(epoch, loss, model, _time, batch=None):
    if batch:
        print "    Batch {} | test loss: {:.2f} | duration: {:.2f}".format(batch, loss, _time)
    else:
        print "Epoch {:<3} | test loss: {:.2f} | duration: {:.2f}".format(epoch, loss, _time)

def main(max_epochs=100, nrows=None):
    df = pd.read_csv('data/numerai_training_data.csv', nrows=nrows)
    feat_cols = df.columns[3:-1]
    target_col = df.columns[-1]

    sampled_df = col_sample(df, 'era', 0.01)
    sampled_df = sampled_df[sampled_df.columns[3:]]
    train, test = col_split(df, 'era', test_size=0.5)

    train_feats = torch.Tensor(train[feat_cols].values)
    train_targ = torch.LongTensor(train[target_col].values)

    test_feats = torch.Tensor(test[feat_cols].values)
    test_targ = torch.LongTensor(test[target_col].values)

    # variables for validation
    val_feats = Variable(test_feats)
    val_targ = Variable(test_targ)

    dataset = TensorDataset(train_feats, train_targ)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    print("Done making data loader")
    model = network(len(feat_cols))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.NLLLoss()
    print("Starting training")

    val_losses = []
    for epoch in range(max_epochs):
        start_time = time.time()
        for batch_idx, (bdata, btargs) in enumerate(dataloader):

            batch_start_time = time.time()
            xs = Variable(bdata)
            ys = Variable(btargs)
            model.zero_grad()

            output = model(xs)
            loss = loss_func(output, ys)
            loss.backward()
            optimizer.step()
            batch_elapsed = time.time() - batch_start_time
            #print_status(epoch, loss.data[0], model, batch_elapsed, batch_idx)

        elapsed = time.time() - start_time
        # validate
        val_output = model(val_feats)
        loss = loss_func(val_output, val_targ)
        val_losses.append(loss.data[0])
        print_status(epoch, loss.data[0], model, elapsed)

    plt.plot(val_losses)
    plt.savefit('validation_losses.png')

if __name__ == '__main__':
    main()
