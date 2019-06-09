'''
Read the notes carefully
1. Note you should only replace 'somewhere' and 'YourFunction' with your own model but WITHOUT any other modifications when you submit this file
2. Run this file on validation set by '' python test.py --testfolder ../Validation_with_labels --num_class 10'', where '../Validation_with_labels' is the path of validation set, 10/107 is number of classes
'''

from model import LSTM
import os
import argparse
from dataset import Writer_Dataset
from utils import *
from config import Config


def label2id(table_, pre):
    for k in table_.keys():
        if table_[k] == pre + 1:
            return int(k)


def test(table_):
    predict_id = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data
            if use_gpu:
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                vote_pre = compute_vote(preds)
                pre_id = label2id(table_, vote_pre)
                predict_id.append(pre_id)
    return predict_id


if __name__ == "__main__":

    # read the test folder
    parser = argparse.ArgumentParser(description='parameters setting')
    parser.add_argument('--testfolder', type=str, default=None)
    parser.add_argument('--num_class', type=int, default=10)
    # parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--config', '-c', metavar='CONFIG', default='config.yml', help='configs')

    args = parser.parse_args()
    config = Config.load(args.config)

    config.NumOfCategory = args.num_class
    config.test_dir = args.testfolder
    use_gpu = torch.cuda.is_available()

    if config.NumOfCategory == 10:
        table = np.load(config.onehot_10_file).item()
        model_file = '10.pkl'
    else:
        table = np.load(config.onehot_107_file).item()
        model_file = '107.pkl'

    print(args.testfolder)

    NumOfCategory = config.NumOfCategory

    len_Sample = config.LenofSample

    test_dataset = Writer_Dataset(config, 'test', reread=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len_Sample,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(model_file)
    state_dict = checkpoint['state_dict']
    # different sample type may have different in_dim
    model = LSTM(in_dim=3, hidden_dim=100, n_layer=1, n_class=NumOfCategory)
    model.load_state_dict(state_dict)
    # for model resume
    # print(model)
    if use_gpu:
        model = model.cuda()
    # read true_ids
    true_ids = np.load(os.path.join(args.testfolder, 'true_ids.npy'))

    predict_ids = test(table)
    test_accuracy = np.mean(np.array(predict_ids) == np.array(true_ids))
    print('Test Accuracy: {:.2f}'.format(test_accuracy))
