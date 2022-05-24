from sklearn.model_selection import train_test_split
import pandas as pd
import torch


class Dataset:
    def __init__(self, test_size=0.3, random_state=100, train=True):
        self.df = pd.read_csv('./data/TMNIST_Data.csv').drop(['names'], axis=1)
        self.df2 = pd.read_csv('./data/mnist_train.csv')

        columns = ['labels'] + [str(i) for i in range(1, 785)]
        self.df.columns = columns
        self.df2.columns = columns

        self.df = pd.concat([self.df, self.df2], ignore_index=True).sample(frac=1).reset_index(drop=True)

        self.__imgs = self.df.loc[:, '1':].astype('float32') / 255
        self.__labels = self.df['labels'].values
        self.x = None
        self.y = None

        x_train, x_test, y_train, y_test = train_test_split(self.__imgs,
                                                            self.__labels,
                                                            test_size=test_size,
                                                            random_state=random_state)

        if train:
            self.x, self.y = x_train, y_train
        else:
            self.x, self.y = x_test, y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x.iloc[idx]).reshape(1, 28, 28), self.y[idx]

    def get_imgs(self):
        return self.__imgs

    def get_labels(self):
        return self.__labels

