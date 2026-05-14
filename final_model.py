import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
from reader import reader

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.layer_1 = nn.Linear(in_features=30,out_features=2048)
        self.layer_2 = nn.Linear(in_features=2048,out_features=16384)
        self.layer_3 = nn.Linear(in_features=16384,out_features=4096)
        self.layer_4 = nn.Linear(in_features=4096,out_features=512)
        self.layer_5 = nn.Linear(in_features=512,out_features=32)
        self.layer_6 = nn.Linear(in_features=32,out_features=1)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        x = x.to(self.device)
        y = self.layer_1(x)
        y = self.ReLU(y)
        y = self.layer_2(y)
        y = self.ReLU(y)
        y = self.layer_3(y)
        y = self.ReLU(y)
        y = self.layer_4(y)
        y = self.ReLU(y)
        y = self.layer_5(y)
        y = self.ReLU(y)
        y = self.layer_6(y)
        return y
    
def normalizer(array,player_num):
    mean_y = np.mean(array[:player_num,1])
    var_x = 10000
    var_y = 10000
    id = 0
    #find the three closest to  each other (offense line)
    for i in range(6,13):
        if var_x > np.std(array[i:i+4,0]) and var_y > np.std(array[i:i+4,1]):
            var_x = np.std(array[i:i+4,0])
            vay_y = np.std(array[i:i+4,1])
            id = i
    #mean_x = line
    if id >= round(player_num/2)-2:
        mean_x = np.mean(array[id:id+4,0]) - 20
    elif id < round(player_num/2)-2:
        mean_x = np.mean(array[id:id+4,0]) + 20
    #normalize
    for i in range(player_num): 
        array[i,0] = (array[i,0] - mean_x)/450
        array[i,1] = (array[i,1] - mean_y)/450
    return array

def split_teams(array,player_num):
    index = 0
    for i in range(len(array[:,0])):
        if array[i,0] > 0:
            index = i
            break
    #split X Y label
    X_L = array[:index,0].copy()
    X_L = filp_shift(X_L)
    X_L = padding(X_L)

    Y_L = array[:index,1].copy()
    Y_L = padding(Y_L)
    label_L = array[index-2,2].copy()
            
    X_R = array[index:player_num,0].copy()
    X_R = filp_shift(X_R)
    X_R = padding(X_R)

    Y_R = array[index:player_num,1].copy()
    Y_R = padding(Y_R)
    label_R = array[index+1,2].copy()

    team_L = np.append(X_L,Y_L,0)
    team_L = np.append(team_L,[label_L],0)

    team_R = np.append(X_R,Y_R,0)
    team_R = np.append(team_R,[label_R],0)
    return team_L,team_R

def filp_shift(array):
    shift_array = array.copy()
    #flip x
    if array[0] < 0:
        shift_array = [-x for x in array]
    #shift left most player to 0
    min = np.min(shift_array)
    shift_array = [(x - min) for x in shift_array]
    return shift_array

def padding(array):
    if len(array) < 15:
        aug_array = array.copy()
        r = 15 - len(aug_array)
        aug_array = np.append(aug_array,np.zeros((r)),0)
    else:
        aug_array = array[-15:].copy()
    return aug_array

def eval(model_path,save_path):
 
    model = classifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    player = torch.load(f"{save_path}test_data.pt")
    labels = torch.load(f"{save_path}test_label.pt")

    with torch.inference_mode():
        correct = 0
        total = 0
        correct_post = 0
        total_post = 0

        for i in range(0,len(player[:,0]),2):
            total += 2
            total_post +=1
            logit_L = model.forward(player[i,:])
            logit_R = model.forward(player[i+1,:])
            #without post process
            pred_L,pred_R = 0,0
            pred_L = torch.round(torch.sigmoid(logit_L))
            pred_R = torch.round(torch.sigmoid(logit_R))
            if pred_L == labels[i]:
                correct+=1
            if pred_R == labels[i+1]:
                correct+=1

            #post process
            logit = torch.cat((logit_L,logit_R),0)
            pred = logit.softmax(dim=0)
            pred = torch.argmax(pred,dim=0)
            if pred == 0:
                pred_L = 1
                pred_R = 0
            else:
                pred_L = 0
                pred_R = 1

            if pred_L == labels[i] and pred_R == labels[i+1]:
                correct_post+=1

    acc = (correct/total)*100
    post_acc = (correct_post/total_post)*100

    print("whitout post process:")
    print(f"total sample:{total}")
    print(f"errors:{total-correct}")
    print(f"acc= {acc}%")
    print("------------------------------------")
    print("after post process:")
    print(f"total sample:{total_post}")
    print(f"errors:{total_post-correct_post}")
    print(f"acc= {post_acc}%")

if __name__ == "__main__":
    # where the coordinate are saved ---------- need to be changed -----------
    input_path = "./football_data/fixed_coordinate"
    # where the prepocessed data are saved ---------- need to be changed -----------
    save_path = "./football_data/save_datas/"
    # choose the best model ---------- need to be changed -----------
    model_path = "./football_project/model/mdl_params_e140.pt"
    
    data = reader(input_path,save_path)
    eval(model_path,save_path)