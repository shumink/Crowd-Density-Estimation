import numpy as np
import random
from scipy.misc import imread, imsave, imresize
project_folder = "./"
visualisation_path = project_folder + 'visualise/'
NEW_SHAPE = [3, 160, 120 ]
class Data:
    def __init__(self,file_path, label_path, validation_size, ninput, nout=4):
        self.train_set = []
        self.train_label = []
        self.eval_set = []
        self.eval_label = []
        self.file_path = file_path
        self.label_path = label_path
        self.ninput = ninput
        self.validation_size = validation_size
        self.load(ninput, nout, validation_size, [i for i in range(ninput)])
        print(np.sum(self.eval_label, axis=0))
        print(len(self.train_label))
        self.randomise()
        self.cnt = 0

    def load(self, ninput, nout, validation_size, order):
        dataset = self.load_data_set(ninput, nout)
        dataset = self.shuffle(dataset, order)
        step = 0
        eval_set = []
        eval_label = []
        for data in dataset:
            if step >= validation_size:
                if step == validation_size:
                    self.eval_set = np.asarray(eval_set).astype(np.float64)
                    self.eval_label = np.reshape(np.asarray(eval_label), (validation_size, nout))
                self.train_set.append(data[0])
                self.train_label.append(data[1])
            else:
                eval_set.append(data[0])
                eval_label.append(data[1])
            step += 1
            self.order.append(data[2])
        self.train_set = np.asarray(self.train_set).astype(np.float64)
        meanval = np.mean(self.train_set, axis=0)
        self.train_set -= meanval
        self.eval_set -= meanval
        self.train_set = np.asarray(self.train_set).astype(np.float64)
        self.train_label = np.asarray(self.train_label).astype(np.float64)



    def load_data_set(self, ninput, nout=4):
        data_set = []
        label = []
        label_file = open(self.label_path)
        for each in label_file:
            vector = np.zeros((1, nout))
            vector[(0, int(each))] = 1
            label.append(vector)
        for i in range(ninput):
            file = self.file_path[i]
            vector = label[i]
            data_set.append([imresize(imread(file), (NEW_SHAPE[2], NEW_SHAPE[1])), vector, i])
        return data_set

    def next_batch(self, size=10):
        idx = np.array(np.random.randint(0, len(self.train_set), size=size))
        return self.train_set[idx], self.train_label[idx]

    def all(self):
        self.cnt += len(self.train_set)
        return self.train_set, np.reshape(self.train_label, (170, 4))

    def hasNext(self):
        return self.cnt < len(self.train_set)

    def getOrder(self):
        return self.order

    def shuffle(self, list, order):
        if self.order is not None or not self.order:
            return [list[i] for i in order]
        else:

            random.shuffle(list)
            return list

    def randomise(self):
        set = []
        for i in range(len(self.train_set)):
            set.append([self.train_set[i], self.train_label[i]])
        random.shuffle(set)
        self.train_label = []
        self.train_set = []
        for newdata in set:
            self.train_set.append(newdata[0])
            self.train_label.append(newdata[1])
        self.refresh()
        self.train_set = np.asarray(self.train_set).astype(np.float64)
        self.train_label = np.asarray(self.train_label).astype(np.float64)

    def refresh(self):
        self.cnt = 0
