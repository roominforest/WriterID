import torch
import numpy as np


def save_model(model, epoch, config, class_num):
    dicts = {'state_dict': model.state_dict(),
             'epoch': epoch,
             'config': config,
             }
    save_file = './' + str(class_num) + '.pkl'
    torch.save(dicts, save_file)
    return


def compute_train_acc(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    correct = pred == label
    return 1. * np.sum(correct)


def compute_vote(pred):
    pred = pred.cpu().numpy()
    # print('pred:',pred.shape)
    # print('pre:',pred)
    counts = np.bincount(pred)
    return np.argmax(counts)