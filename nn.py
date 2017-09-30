import pdb
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
from cv_model import col_split, col_sample

def network(input_dim, hidden_dim):
    # simple 2 layer network
    m = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2),
        nn.LogSoftmax()
    )
    return m

def print_status(epoch, loss, _time, y_true, model_output, data='val', batch=None):
    '''
    Args:
        epoch: int
        loss: float
        model_output: variable of tensor
        _time: float
        data: string of "train", test"
    '''
    if data == 'batch':
        print "    Batch {} | test loss: {:.2f} | time: {:.2f}".format(batch, loss, _time)
    elif data == 'test':
        report = predict_analysis(model_output, y_true)
        print "Epoch {:<3} | {:<5} | test loss: {:.2f} | time: {:.2f} | {} |".format(epoch, data, loss,
                                                                                _time, report)
    elif data == 'train':
        y_predict = predict(model_output)
        accuracy = np.mean(y_predict == y_true)
        print "Epoch {:<3} | {:<5} | test loss: {:.2f} | time: {:.2f} | acc: {:.3f}".format(epoch, data, loss,
                                                                                   _time, accuracy)

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
    report += "| tp: {:.3f}"
    report = report.format(accuracy, pred_zeros, pred_ones, tn, tp)
    return report

def main(hidden_dim, max_epochs=1000, nrows=None):
    df = pd.read_csv('data/numerai_training_data.csv', nrows=nrows)
    feat_cols = df.columns[3:-1]
    target_col = df.columns[-1]

    sampled_df = col_sample(df, 'era', 0.01)
    sampled_df = sampled_df[sampled_df.columns[3:]]
    train, test = col_split(df, 'era', test_size=0.5)

    train_feats = torch.Tensor(train[feat_cols].values)
    y_train = train[target_col].values
    train_targ = torch.LongTensor(y_train)

    test_feats = torch.Tensor(test[feat_cols].values)
    y_test = test[target_col].values
    test_targ = torch.LongTensor(y_test)

    # variables for validation
    var_train_feats = Variable(train_feats)
    var_train_targ = Variable(train_targ)

    # variables for validation
    val_feats = Variable(test_feats)
    val_targ = Variable(test_targ)

    dataset = TensorDataset(train_feats, train_targ)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    model = network(len(feat_cols), hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.NLLLoss()
    print("Starting training for model with hidden dim: {}".format(hidden_dim))

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
            #print_status(epoch, loss.data[0], batch_elapsed,  ys, model_output, 'batch', batch_idx)

        elapsed = time.time() - start_time

        if epoch > 0 and epoch % 10 == 0:
            # training error
            train_start = time.time()
            train_output = model(var_train_feats)
            y_pred_train = predict(train_output)
            accuracy = np.mean(y_pred_train == y_train)
            loss = loss_func(train_output, var_train_targ)
            val_losses.append(loss.data[0])
            train_elapsed = time.time() - train_start
            print_status(epoch, loss.data[0], elapsed, y_train, train_output, 'train')

            # validation error
            val_start = time.time()
            val_output = model(val_feats)
            y_pred = predict(val_output)
            accuracy = np.mean(y_pred == y_test)
            loss = loss_func(val_output, val_targ)
            val_losses.append(loss.data[0])
            val_elapsed = time.time() - val_start
            print_status(epoch, loss.data[0], elapsed, y_test, val_output, 'test')


if __name__ == '__main__':
    dims = [2*i for i in range(2, 6)]
    for hidden_dim in dims[::-1]:
        main(hidden_dim)
        print '=' * 80
