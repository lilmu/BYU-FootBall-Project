import numpy as np
import torch 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalizer(array,player_num):
    std_x = np.std(array[:player_num,0])
    std_y = np.std(array[:player_num,1])
    mean_y = np.mean(array[:player_num,1])
    #find the X mean of th offense line
    var_x = 10000
    var_y = 10000
    id_x = 0
    for i in range(player_num-4):
        if (var_x > np.std(array[i:i+4,0]) and var_y > np.std(array[i:i+4,1])):

            var_x = np.std(array[i:i+4,0])
            vay_y = np.std(array[i:i+4,1])
            id_x = i

    if id_x >= round(player_num/2)-2:
        mean_x = np.mean(array[id_x:id_x+4,0]) - 20
    elif id_x < round(player_num/2)-2:
        mean_x = np.mean(array[id_x:id_x+4,0]) + 20
   
    for i in range(player_num): 
        array[i,0] = (array[i,0] - mean_x)/(450)
        array[i,1] = (array[i,1] - mean_y)/(450)
    return array


def augmentation(array,player_num):
    aug_array = array.copy()
    aug_array_s = array.copy()

    aug_array[:player_num] = [-x for x in array[:player_num]]
    aug_array[58] = not(array[58])
    aug_array_s[29:29+player_num] = [-y for y in array[29:29+player_num]]
    aug_array_t = aug_array.copy()
    aug_array_t[29:29+player_num] = [-y for y in aug_array[29:29+player_num]]
    return aug_array, aug_array_s, aug_array_t

def add_noise(array):
    array = np.squeeze(array)
    player_num = 29
    for i in range(29,58):
        if array[i] == 0:
            player_num = i-28
            break
    i = np.random.randint(10)
    if i == 5:
        j = np.random.randint(player_num)
        for k in range(j,player_num-1):
            array[k] = array[k+1]
            array[k+29] = array[k+29]
    X = array[0:29]
    Y = array[29:58]
    # Generate a noise sample consisting of values that are a little higer or lower than a few randomly selected values in the original data. 
    noise_sample_X = np.random.default_rng().uniform(-0.02,0.02, int(0.8*player_num))
    noise_sample_Y = np.random.default_rng().uniform(-0.02,0.02,int(0.8*player_num))
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

    aug = np.append(X_noised,Y_noised,0)
    aug = np.append(aug,[array[58]],0)
    return aug

def decide_side(array):
    side = 0
    # offense on the right
    if array[0] == 0:
        side = 1
    # offense on the left
    elif array[0] == 1:
        side = 0
    return side

class reader():
    def __init__(self,dic_path):
        #get file names
        file_list = []
        for file_name in os.listdir(dic_path):
            if file_name.endswith(".npy"):
                file_list.append(file_name)
        train_test = 0
        train_dataSet = np.zeros((1,59))
        test_dataSet = np.zeros((1,59))
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
            #split X Y label
            X = file_array[:,0].copy()
            Y = file_array[:,1].copy()
            label = file_array[:,2].copy()

            # 0 for offense on left
            # 1 for offense on right
            side = decide_side(label)
        
            X = np.append(X,Y,0)
            data = np.append(X,[side],0)

            if train_test == 0:
                train_dataSet = np.append(train_dataSet,[data],0)
            else:
                test_dataSet = np.append(test_dataSet,[data],0)
                
        train_dataSet = np.delete(train_dataSet,0,0)
        test_dataSet = np.delete(test_dataSet,0,0)

        for i in range(len(train_dataSet[:,0])):
            aug_1 = add_noise(train_dataSet[i,:])
            aug_2 = add_noise(train_dataSet[i,:])
            train_dataSet = np.append(train_dataSet,[aug_1],0)
            train_dataSet = np.append(train_dataSet,[aug_2],0)

        np.random.shuffle(train_dataSet)
        
        self.train = torch.from_numpy(train_dataSet)
        self.test = torch.from_numpy(test_dataSet)
        self.train = self.train.type(torch.float32)
        self.test = self.test.type(torch.float32)

        save_path = "/home/ephraimpan/football_data/save_datas/side/"
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        torch.save(self.test[:,:58],f'{save_path}test_data.pt')
        torch.save(self.test[:,58],f'{save_path}test_label.pt')
        torch.save(self.train[:,:58],f'{save_path}train_data.pt')
        torch.save(self.train[:,58],f'{save_path}train_label.pt')

if __name__ == "__main__":
    dic_path = "/home/ephraimpan/football_data/fixed_coordinate/"
    data = reader(dic_path)
    
    #visulizer
    a = np.random.randint(1298, size=10)
    for i in a:
        plt.figure(figsize=(12, 12))
        plt.plot()
        plt.title("Train")
        plt.scatter(data.train[i,:29],data.train[i,29:58])
        plt.show()
        # plt.subplot(1, 2, 2)
        # plt.title("Test")
        # plt.scatter(data.test[i,:29],data.test[i,29:58])
        # plt.show()