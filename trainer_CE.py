import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from reader_CE import reader
from football_dataset import football_dataset
from plot import confusion_matrix_dispaly
from plot import visualizer


class team_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
         #create three linear layers
        self.dropout = nn.Dropout(p=0.3)
        self.layer_1 = nn.Linear(in_features=2,out_features=2048)
        self.layer_2 = nn.Linear(in_features=2048,out_features=16384)
        self.layer_3 = nn.Linear(in_features=16384,out_features=4096)
        self.layer_4 = nn.Linear(in_features=4096,out_features=512)
        self.layer_5 = nn.Linear(in_features=512,out_features=32)
        self.layer_6 = nn.Linear(in_features=32,out_features=3)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        x = x.to(self.device)
        y = self.dropout(x)
        y = self.layer_1(y)
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
    
class trainer():
    def __init__(self, Path_test_player, Path_test_label, Path_train_player, Path_train_label):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.batchsize = 64
        self.Epochs = 150
        self.model = team_classifier().to(self.device)
        self.lr = 0.05
        self.eval_every = 10
        self.save_every = 10
        
        data_path = "./football_data/coordinate/"
        self.save_model_path = "./football_project/CE/model/"
        self.save_pic_path = "./football_project/CE/pic/"

        self.data = reader(data_path)
        test_data = football_dataset(Path_test_player, Path_test_label)
        training_data = football_dataset(Path_train_player, Path_train_label)

        self.train_dataloader = DataLoader(training_data, batch_size=self.batchsize, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batchsize, shuffle=True)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,weight_decay=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

        

    def save(self, train_loss, eval_loss, train_acc, eval_acc, eval_epochNum, epochNum):
        def makePlot(firstX, firstY, secondX, secondY, title, ylabel, fileName):
            plt.plot(firstX, firstY)
            plt.plot(secondX, secondY)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.savefig(self.save_pic_path + fileName)
            plt.close()

        # make loss plot
        makePlot(range(0, len(train_loss)), train_loss, eval_epochNum, eval_loss, title="Loss Plot", ylabel="Mean Loss",
                 fileName="lossPlot_e" + str(epochNum) + ".png")
        # make acc plotlogits.softmax(dim=2)
        makePlot(range(0, len(train_acc)), train_acc, eval_epochNum, eval_acc, title="Accuracy Plot",
                 ylabel="Mean Accuracy", fileName="accPlot_e" + str(epochNum) + ".png")

        torch.save(self.model.state_dict(), str(self.save_model_path + "mdl_params_e" + str(epochNum) + ".pt"))

    def eval_helper(self):
        correct = 0
        acc_total = 0
        losses =[]
        self.model.eval()
        pred_arr = torch.zeros(1,29)
        label_arr = torch.zeros(1,29)
        player_arr = torch.zeros(1,29,2)
        with torch.inference_mode():
            for player, label in self.test_dataloader:
                player.to(self.device)
                label.to(self.device)
                logits = self.model.forward(player)
                logits = torch.squeeze(logits)
                loss = self.loss_fn(logits.softmax(dim=2), label)

                losses.append(loss.item())
                #get accuracy
                pred = logits.softmax(dim=2)
                label_item = torch.argmax(label, dim=2)
                pred_item = torch.argmax(pred, dim=2)
                
                for i in range(pred_item.size(dim=0)):
                    for j in range(1,pred_item.size(dim=1)-1):
                        if pred_item[i,j-1] == pred_item[i,j+1] == 1 and pred_item[i,j] != 1:
                            pred_item[i,j] = 1
                        elif pred_item[i,j-1] == pred_item[i,j+1] == 0 and pred_item[i,j] != 0:
                            pred_item[i,j] = 0
                        elif pred_item[i,j] == 2 and (player[i,j,0] != 0 or player[i,j,1] != 0):
                            pred_item[i,j] == pred_item[i,j-1]
                                
                correct += torch.eq(label_item,pred_item).sum().item()            
                acc_total += label_item.size(dim=0)*label_item.size(dim=1)

                pred_arr = torch.cat((pred_arr,pred_item),dim=0)
                label_arr = torch.cat((label_arr,label_item),dim=0)
                player_arr = torch.cat((player_arr,player),dim=0)
                              
        pred_arr = pred_arr[1:,:]
        label_arr = label_arr[1:,:]
        player_arr = player_arr [1:,:,:]
        acc = (correct/acc_total)*100
        print(f"testing loss :{np.mean(losses)} | acc :{acc}%")
        return np.mean(losses), acc, pred_arr, label_arr, player_arr

    def train_helper(self):
        correct = 0
        acc_total = 0
        losses = []
        self.model.train()
        for player, label in self.train_dataloader:
            player.to(self.device)
            label.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model.forward(player)
            logits = torch.squeeze(logits)
            loss = self.loss_fn(logits.softmax(dim=2), label)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

           #get accuracy
            pred = logits.softmax(dim=2)
            label_item = torch.argmax(label, dim=2)
            pred_item = torch.argmax(pred, dim=2)


            correct += torch.eq(label_item,pred_item).sum().item()
            acc_total += label_item.size(dim=0)*label_item.size(dim=1)
                
        acc = (correct/acc_total)*100
        # if acc>70:
        #      print(pred_item)

        return np.mean(losses), acc

    def runAll(self):
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        evaluation_epochNum = []
        for i in range(0, self.Epochs):

            t_loss, t_acc = self.train_helper()
            
            train_loss.append(t_loss)
            train_acc.append(t_acc)

            if i % self.eval_every == 0:
                e_loss,e_acc,pred_arr,label_arr,player_arr = self.eval_helper()

                mat_path = "./football_project/CE/confusion_matrix/"
                pred_arr = torch.flatten(pred_arr, start_dim=0, end_dim=-1)
                label_arr_f = torch.flatten(label_arr,start_dim=0,end_dim=-1)
                confusion_matrix_dispaly(pred_arr, label_arr_f,save_path= mat_path, fileName=f"{i}.jpg")
                
                eval_loss.append(e_loss)
                eval_acc.append(e_acc)
                evaluation_epochNum.append(i)
                print(f"trainig loss :{np.mean(train_loss)} | acc :{np.mean(train_acc)}%")
                print("-----------------------------------------------------------------------")
                visual_path = "./football_project/CE/visualize/"
                visualizer(self.model,player_arr,label_arr,visual_path,i)

            if i%self.save_every == 0:
                self.save(train_loss, eval_loss, train_acc, eval_acc, evaluation_epochNum, i)
                


Path_test_player = Path("./football_data/save_datas/CE/test_data.pt")
Path_test_label = Path("./football_data/save_datas/CE/test_label.pt")
Path_train_player = Path("./football_data/save_datas/CE/train_data.pt")
Path_train_label = Path("./football_data/save_datas/CE/train_label.pt")

classifier = trainer(Path_test_player, Path_test_label, Path_train_player, Path_train_label)
classifier.runAll()