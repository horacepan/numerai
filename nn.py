import pdb
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
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

def print_status(epoch, loss, _time, y_true, model_output, batch=None):
    '''
    Args:
        epoch: int
        loss: float
        model_output: variable of tensor
        _time: float
        batch: None or int batch number
    '''
    if batch:
        print "    Batch {} | test loss: {:.2f} | time: {:.2f}".format(batch, loss, _time)
    else:
        report = predict_analysis(model_output, y_true)
        print "Epoch {:<3} | test loss: {:.2f} | time: {:.2f} | {}".format(epoch, loss, _time,
                                                                               report)

def predict(output):
    return output.data.numpy().argmax(axis=1)

def predict_analysis(model_output, y_true):
    '''
    Return a string with stuff like accuracy, number of predicted zeros, ones, true zeros, true ones
    '''
    y_predict = predict(model_output)
    accuracy = np.mean(y_predict == y_true)

    pred_zeros = np.mean(y_predict == 0)
    pred_ones = 1.0 - pred_zeros
    cm = confusion_matrix(y_true, y_predict)
    tn, fp, fn, tp = cm.ravel() / float(len(y_true))

    # TODO: fix this gross line breaking
    report = "acc: {:.3f} | pred zeros: {:.3f} "
    report += "| pred ones: {:.3f} | tn: {:.3f} "
    report += "| fn: {:.3f} | fp: {:.3f} | tp: {:.3f}"
    report = report.format(accuracy, pred_zeros, pred_ones, tn, fn, fp, tp)
    return report

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
    y_test = test[target_col].values

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

            model_output = model(xs)
            loss = loss_func(model_output, ys)
            loss.backward()
            optimizer.step()

            batch_elapsed = time.time() - batch_start_time
            #print_status(epoch, loss.data[0], batch_elapsed,  ys, model_output, batch_idx)

        elapsed = time.time() - start_time
        # validate
        val_output = model(val_feats)
        y_pred = predict(val_output)
        accuracy = np.mean(y_pred == y_test)
        loss = loss_func(val_output, val_targ)
        val_losses.append(loss.data[0])
        print_status(epoch, loss.data[0], elapsed, y_test, val_output)

    plt.plot(val_losses)
    plt.savefig('validation_losses.png')

if __name__ == '__main__':
    main(nrows)
