from torch.utils.data import Dataset
import glob
import numpy as np
import os


class Writer_Dataset(Dataset):

    def __init__(self, config, phase, reread=True,save = False):
        # print('init')
        self.reread = reread
        self.save = save
        self.phase = phase
        self.lenofSample = config.LenofSample
        self.numofSamples = config.NumofSamples
        self.test_dir = config.test_dir

        self.root = config.data_dir
        self.NumOfCategory = config.NumOfCategory
        self.sample_type = config.sample_type
        # self.data_rhs = {}
        if self.NumOfCategory == 107:
            self.train_dir = self.root + config.data_107 + 'Train/'
            self.val_dir = self.root + config.data_107 + 'Validation/'
            self.one_hot = config.onehot_107_file
        if self.NumOfCategory == 10:
            self.train_dir = self.root + config.data_10 + 'Train/'
            self.val_dir = self.root + config.data_10 + 'Validation/'
            self.one_hot = config.onehot_10_file
            # self.test_dir = self.root + config.data_10 + 'Validation_with_labels/'
        self.one_hot_dict = np.load(self.one_hot).item()

        if self.phase == 'Train':
            self.file = glob.glob(self.train_dir + "*" + '.npy')
        #
        #     # self.train_file.sort()
        if self.phase == 'val':
            self.file = glob.glob(self.val_dir + '*' + '.npy')

        # if self.phase == 'test':
        #     self.file = glob.glob(self.test_dir + '*' + '.npy')
        # print('done')
        if self.phase != 'test':
            if not os.path.exists(self.phase + '_' + str(self.NumOfCategory) + '_rhs.npy') or self.reread:
                self.data_rhs = self.gen_data(self.file)
            else:
                self.data_rhs = np.load(self.phase + '_' + str(self.NumOfCategory) + '_rhs.npy')
        else:
            self.data_test = self.gen_test(self.test_dir)

    def gen_test(self, test_dir):
        data_rhs = []
        label_file = os.path.join(test_dir + 'true_ids.npy')
        if test_dir[-1] == '/':
            data_file = glob.glob(test_dir + '*' + '.npy')
        else:
            data_file = glob.glob(test_dir + '/' + '*' + '.npy')
        data_file.sort()
        for file in data_file:
            if 'true_ids' in file:
                pass
            else:
                data = np.load(file)
                if self.sample_type == 'stroke':
                    data_RHS = self.CreatRHS_stroke(data)
                if self.sample_type == 'word':
                    data_RHS = self.CreatRHS_word(data)
                if self.sample_type == 'all':
                    data_RHS = self.CreatRHS_all(data)
                for i in range(self.lenofSample):
                    data_sample = self.random_sample(data_RHS)
                    data_rhs.append(data_sample)
        data_rhs = np.array(data_rhs)
        if self.save:
            np.save(self.phase + '_' + str(self.NumOfCategory) + '_rhs.npy', data_rhs)
        print('data done')
        return data_rhs

    def gen_data(self, file_dirs):
        data_rhs = []
        for file_dir in file_dirs:
            data = np.load(file_dir)
            label = file_dir.split('/')[-1].split('.')[0]
            label = self.trans(label)
            if self.sample_type == 'stroke':
                data_RHS = self.CreatRHS_stroke(data)
            if self.sample_type == 'word':
                data_RHS = self.CreatRHS_word(data)
            if self.sample_type == 'all':
                data_RHS = self.CreatRHS_all(data)
            for i in range(self.lenofSample):
                info = {}
                data_sample = self.random_sample(data_RHS)
                info['data'] = data_sample
                info['label'] = label
                data_rhs.append(info)
        data_rhs = np.array(data_rhs)
        if self.save:
            np.save(self.phase + '_' + str(self.NumOfCategory) + '_rhs.npy', data_rhs)
        print('data done')
        return data_rhs

    def trans(self, label):
        index = self.one_hot_dict[label]
        return index - 1

    def CreatRHS_all(self, data): # sample method 3
        data_RHS = []
        new_data = []
        for word in data:
            for i in range(len(word)):
                stroke = word[i]
                for j in range(len(stroke)):
                    point = stroke[j]
                    new_data.append(point)
        for k in range(1,len(new_data)):
            rhs_ = list(np.array(new_data[k]) - np.array(new_data[k-1]))
            data_RHS.append(np.array(rhs_))
        data_RHS = np.array(data_RHS)
        return data_RHS

    def CreatRHS_word(self, data): # sample method 2
        data_RHS = []
        for word in data:
            new_word = []
            for i in range(len(word)):
                stroke = word[i]
                for j in range(len(stroke)):
                    point = stroke[j]
                    new_word.append(point)
            for k in range(1,len(new_word)):
                rhs_ = list(np.array(new_word[k])- np.array(new_word[k-1]))
                data_RHS.append(np.array(rhs_))
        data_RHS = np.array(data_RHS)
        return data_RHS

    def CreatRHS_stroke(self, data): # sample method 1
        data_RHS = []
        for word in data:
            # word = np.array(word)
            if len(word) == 1:
                pass
            else:
                for i in range(0, len(word)):
                    for j in range(len(word[i])):  # stroke = word[i]
                        if j == 0:
                            rhs_ = list(word[i][j])
                            rhs_.append(0)
                        else:
                            rhs_ = list(np.array(word[i][j]) - np.array(word[i][j - 1]))
                            rhs_.append(1)
                        rhs_ = np.array(rhs_)
                        data_RHS.append(rhs_)
        data_RHS = np.array(data_RHS)
        return data_RHS

    def random_sample(self, data_RHS):
        for i in range(self.lenofSample):
            sample_point = np.random.random_integers(0, len(data_RHS) - self.numofSamples)
            data_sample = data_RHS[sample_point:(sample_point + self.numofSamples)]
        return data_sample

    def __getitem__(self, idx):
        if self.phase != 'test':
            info = self.data_rhs[idx]
            data_RHS_sample = info['data']
            label = info['label']
            return data_RHS_sample,label
        else:
            return self.data_test[idx]

    def __len__(self):
        return self.lenofSample * self.NumOfCategory
