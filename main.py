import argparse
from dataset import Writer_Dataset
from config import Config
from model import LSTM,BiLSTM
from torch.optim import lr_scheduler
import torch.utils.data
from utils import *

parser = argparse.ArgumentParser(description='WriterID task')
parser.add_argument('--config', '-c', metavar='CONFIG', default='config.yml', help='configs')


def train(epoch):
    print('epoch:', str(epoch))
    model.train()
    correct = 0
    num_train = len(Train_loader)*BATCH_SIZE
    # print(num_train)
    for data in Train_loader:
        input, label = data

        if use_gpu:
            input = input.cuda().type(torch.cuda.FloatTensor)
            label = label.cuda()

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            _, preds = torch.max(outputs.data, 1)
            correct += compute_train_acc(preds,label)
            loss.backward()
            optimizer.step()
    train_acc = correct/num_train
    print('train_acc:',train_acc)


def val():
    model.eval()
    correct = 0
    num = len(Validation_loader)  # num is the number of the Writers
    # print("val len:",len(Validation_loader))
    with torch.no_grad():
        for data in Validation_loader:
            input, label = data  # data is a batch


            if use_gpu:
                input = input.cuda().type(torch.cuda.FloatTensor)
                label = label.cuda()
                outputs = model(input)
                _, preds = torch.max(outputs.data, 1)
                vote_pre = compute_vote(preds)
                true_id = label.cpu().numpy()[0]
                if vote_pre == true_id:
                    correct += 1.
        acc = correct / num
        print("val acc:", acc)


if __name__ == '__main__':
    args = parser.parse_args()
    config = Config.load(args.config)
    BATCH_SIZE = config.BATCH_SIZE
    EPOCH = config.EPOCH
    use_gpu = torch.cuda.is_available()
    NumOfCategory = config.NumOfCategory
    len_Sample = config.LenofSample

    print("class num:",NumOfCategory)
    re = False
    # prepare dataset
    train_dataset = Writer_Dataset(config, 'Train', reread=re,save = True)
    val_dataset = Writer_Dataset(config, 'val', reread=re,save = True)

    Train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    Validation_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=len_Sample,
        shuffle=False,
        num_workers=0,
    )

    # model
    # Hidden_dim is determined by the needs
    if config.model_type == 'single':
        model = LSTM(in_dim=3, hidden_dim=100, n_layer=1, n_class=NumOfCategory)
    elif config.model_type == 'bi':
        model = BiLSTM(input_size = 3, hidden_size=128, num_layers=2, num_classes=NumOfCategory)
    if use_gpu:
        model = model.cuda()

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay= 1e-8)  # weight_decay=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # train and validation
    for epoch in range(EPOCH):
        scheduler.step()
        train(epoch)
        val()
    # save_model(model, epoch, config, config.NumOfCategory)

