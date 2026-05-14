import numpy as np
import torch 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalizer(array,player_num):
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
        array[i,0] = (array[i,0] - mean_x) /(450)
        array[i,1] = (array[i,1] - mean_y) /(450)
    return array

def oneHot_encoder(array):
    #make file [1,29,6]
    zero = np.zeros((29,2))
    array = np.append(array,zero,axis=1)
    #encode
    for i in range(len(array[:,0])):
                if array[i,2] == 0:
                    array[i,2:] = [1,0,0]
                elif array[i,2] == 1:
                    array[i,2:] = [0,1,0]
                elif array[i,2] == 2:
                    array[i,2:] = [0,0,1]
    return array

def unify(array):
    aug_array = array.copy()
    for i in range(len(array[:,0])):
        if array[i,0] < 0:
            aug_array[i,2] = 0
        elif array[i,0]==0 and array[i,1]==0:
            aug_array[i,2] = 2
        elif array[i,0] >= 0:
            aug_array[i,2] = 1
    return aug_array

def add_noise(array):
    array = np.squeeze(array)
    player_num = 29
    for i in range(0,29):
        if array[i,1] == 0:
            player_num = i+1
            break

    i = np.random.randint(10)
    if i == 5:
        j = np.random.randint(player_num)
        for k in range(j,player_num-1):
            array[k,0] = array[k+1,0]
            array[k,1] = array[k+1,1]

    X = array[:,0]
    Y = array[:,1]
    label = array[:,2:]
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
    X_noised = np.expand_dims(X_noised,1)
    Y_noised = np.expand_dims(Y_noised,1)
    aug = np.append(X_noised,Y_noised,1)
    aug = np.append(aug,label,1)
    return aug
class reader():
    def __init__(self,dic_path):
        
        #get file names
        file_list = []
        for file_name in os.listdir(dic_path):
            if file_name.endswith(".npy"):
                file_list.append(file_name)
        file_list.sort()
        train_test = 0
        test_dataSet = np.zeros((1,29,5))
        train_dataSet = np.zeros((1,29,5))
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
            #make ghost player label = 2
            file_array[self.player_num:,2] = 2
            #make array dtype = float32
            file_array = file_array.astype('float32')

            #normalizing x,y
            file_array = normalizer(file_array,self.player_num)
            #do one-hot encode
            file_array = oneHot_encoder(file_array)
            #sup_data to dataSet
            if train_test == 0:
                train_dataSet = np.append(train_dataSet,[file_array],axis=0)
            else:
                test_dataSet = np.append(test_dataSet,[file_array],axis=0)
  
        train_dataSet = np.delete(train_dataSet,0,0)
        test_dataSet = np.delete(test_dataSet,0,0)
        for i in range(len(train_dataSet[:,0,0])):
            aug_1 = add_noise(train_dataSet[i,:,:])
            aug_2 = add_noise(train_dataSet[i,:,:])
            train_dataSet = np.append(train_dataSet,[aug_1],0)
            train_dataSet = np.append(train_dataSet,[aug_2],0)

        np.random.shuffle(train_dataSet)

        self.train = torch.from_numpy(train_dataSet)
        self.test = torch.from_numpy(test_dataSet)
        self.train = self.train.type(torch.float32)
        self.test = self.test.type(torch.float32)

        save_path = "./football_data/save_datas/CE/"
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        torch.save(self.test[:,:,0:2],f'{save_path}test_data.pt')
        torch.save(self.test[:,:,2:],f'{save_path}test_label.pt')
        torch.save(self.train[:,:,0:2],f'{save_path}train_data.pt')
        torch.save(self.train[:,:,2:],f'{save_path}train_label.pt')

if __name__ == "__main__":
    dic_path = "./football_data/fixed_coordinate/"
    data = reader(dic_path)
    
    #visulizer
    a = np.random.randint(559, size=15)
    for i in a:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plt.scatter(data.train[i,:,0],data.train[i,:,1],c=data.train[i,:,2],cmap=plt.cm.RdYlGn)

        plt.subplot(1, 2, 2)
        plt.title("Test")
        plt.scatter(data.test[i,:,0],data.test[i,:,1],c=data.test[i,:,2],cmap=plt.cm.RdYlGn)
        plt.show()
    
       
