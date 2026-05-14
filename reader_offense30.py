import numpy as np
import torch 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalizer(array,player_num):
    
    var_x = 10000
    var_y = 10000
    id = 0
    #find the three closest to  each other (offense line)
    for i in range(6,13):
        if var_x > np.std(array[i:i+4,0]) and var_y > np.std(array[i:i+4,1]):

            var_x = np.std(array[i:i+4,0])
            vay_y = np.std(array[i:i+4,1])
            id = i

    if id >= round(player_num/2)-2:
        mean_x = np.mean(array[id:id+4,0]) - 20
    elif id < round(player_num/2)-2:
        mean_x = np.mean(array[id:id+4,0]) + 20

    mean_y = np.mean(array[:player_num,1])
    #normalize
    for i in range(player_num): 
        array[i,0] = (array[i,0] - mean_x)/450
        array[i,1] = (array[i,1] - mean_y)/450
    return array

def find_index(array):
    index = 0 
    for i in range(len(array[:,2])):
        if array[i,2] != array[i+1,2]:
            index = i+1
            break
    return index 

def split(array):
    index = 0
    for i in range(len(array[:,0])):
        if array[i,0] > 0:
            index = i
            break
    return index

def filp(array):
    if array[0] < 0:
        array[:] = [-x for x in array[:]]
    return array

def shift(array):
    min = np.min(array)
    shift_array = array.copy()
    shift_array = [(x - min) for x in array]
    return shift_array

def padding(array):
    aug_array = array.copy()
    r = 15 - len(aug_array)
    aug_array = np.append(aug_array,np.zeros((r)),0)
    return aug_array

def add_noise(array):
    array = np.squeeze(array)
    player_num = 15
    for i in range(16,30):
        if array[i] == 0:
            player_num = i-14
            break
    i = np.random.randint(10)
    if i == 5:
        j = np.random.randint(player_num)
        for k in range(j,player_num-1):
            array[k] = array[k+1]
            array[k+15] = array[k+15]
    X = array[0:15]
    Y = array[15:30]
    # Generate a noise sample consisting of values that are a little higer or lower than a few randomly selected values in the original data. 
    noise_sample_X = np.random.default_rng().uniform(0.04,0.14, int(0.8*player_num))
    noise_sample_Y = np.random.default_rng().uniform(0.04,0.14, int(0.8*player_num))
    # Generate an array of zeros with a size that is the difference of the sizes of the original data an the noise sample.
    zeros_X = np.zeros(player_num - len(noise_sample_X))
    zeros_Y = np.zeros(player_num - len(noise_sample_Y))
    # Add the noise sample to the zeros array to obtain the final noise with the same shape as that of the original data.
    noise_X = np.append(noise_sample_X, zeros_X,0)
    noise_Y = np.append(noise_sample_Y, zeros_Y,0)
    # Shuffle the values in the noise to make sure the values are randomly placed.
    np.random.shuffle(noise_X)
    np.random.shuffle(noise_Y)

    zeros_X = np.zeros(len(X) - player_num)
    zeros_Y = np.zeros(len(Y) - player_num)

    noise_X = np.append(noise_X, zeros_X,0)
    noise_Y = np.append(noise_Y, zeros_Y,0)

    # Obtain data with the noise added.
    X_noised = X + noise_X
    Y_noised = Y + noise_Y
    #shift
    min = np.min(X_noised[:player_num])
    mean = np.mean(Y_noised[:player_num])
    X_noised[:player_num] = [X - min for X in X_noised[:player_num]]
    Y_noised[:player_num] = [Y - mean for Y in Y_noised[:player_num]]
    aug = np.append(X_noised,Y_noised,0)
    aug = np.append(aug,[array[30]],0)
    return aug

class reader():
    def __init__(self,dic_path):
        #get file names
        file_list = []
        for file_name in os.listdir(dic_path):
            if file_name.endswith(".npy"):
                file_list.append(file_name)
        file_list.sort()
        train_dataSet = np.zeros((1,2,31))
        test_dataSet = np.zeros((1,2,31))
        train_test = 0

        for file in file_list:
            #read file
            file_array = np.load((dic_path + "/" + file))
            if file == "2022 North Clayton 1-1.npy":
                train_test = 1
            #get player number
            self.player_num = file_array[0,1]
            #drop header
            file_array = file_array[1:,:]
            #if label>0 =1
            file_array[:,2] = np.where(file_array[:,2] > 0, 1, 0)
            #make array dtype = float32
            file_array = file_array.astype('float32')
            
            #normalizing x,y
            file_array = normalizer(file_array,self.player_num)

            index = split(file_array)
            #split X Y label
            X_0 = file_array[:index,0].copy()
            X_0 = filp(X_0)
            X_0 = shift(X_0)
            X_0 = padding(X_0)
            
            Y_0 = file_array[:index,1].copy()
            Y_0 = padding(Y_0)
            label_0 = file_array[0,2].copy()
            

            X_1 = file_array[index:self.player_num,0].copy()
            X_1 = shift(X_1)
            X_1 = padding(X_1)

            Y_1 = file_array[index:self.player_num,1].copy()
            Y_1 = padding(Y_1)

            label_1 = file_array[self.player_num-2,2].copy()

            data_1 = np.append(X_0,Y_0,0)
            data_1 = np.append(data_1,[label_0],0)

            data_2 = np.append(X_1,Y_1,0)
            data_2 = np.append(data_2,[label_1],0)
            
            data_1 = np.expand_dims(data_1, 0)
            data_2 = np.expand_dims(data_2, 0)
            data = np.append(data_1, data_2 ,0)

            if train_test == 0:
                train_dataSet = np.append(train_dataSet,[data],0)
            else:
                test_dataSet = np.append(test_dataSet,[data],0)

        train_dataSet = np.delete(train_dataSet,0,0)
        test_dataSet = np.delete(test_dataSet,0,0)

        #augument
        for i in range(len(train_dataSet[:,0])):
            aug = add_noise(train_dataSet[i,0,:])
            aug_2 = add_noise(train_dataSet[i,1,:])
            aug_3 = add_noise(train_dataSet[i,0,:])
            aug_4 = add_noise(train_dataSet[i,1,:])

            aug = np.expand_dims(aug, 0)
            aug_2 = np.expand_dims(aug_2, 0)
            aug_3 = np.expand_dims(aug_3, 0)
            aug_4 = np.expand_dims(aug_4, 0)

            aug = np.append(aug,aug_2,0)
            aug_3 = np.append(aug_3,aug_4,0)
            train_dataSet = np.append(train_dataSet,[aug],0)

        np.random.shuffle(train_dataSet)

        self.train = torch.from_numpy(train_dataSet)
        self.test = torch.from_numpy(test_dataSet)
        self.train = self.train.type(torch.float32)
        self.test = self.test.type(torch.float32)

        save_path = "./football_data/save_datas/off30/"
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        torch.save(self.test[:,:,:30],f'{save_path}test_data.pt')
        torch.save(self.test[:,:,30],f'{save_path}test_label.pt')
        torch.save(self.train[:,:,:30],f'{save_path}train_data.pt')
        torch.save(self.train[:,:,30],f'{save_path}train_label.pt')

if __name__ == "__main__":
    dic_path = "./football_data/fixed_coordinate/"
    data = reader(dic_path)
    
    #visulizer
    # a = np.random.randint(600, size=10)
    # for i in a:
    #     plt.figure(figsize=(6, 12))
    #     plt.plot()
    #     plt.title("Train")
    #     plt.scatter(data.train[i,:15],data.train[i,15:30])
    #     plt.show()
