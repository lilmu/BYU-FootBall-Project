import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=2)

def confusion_matrix_dispaly(pred,label,save_path,fileName):
    con_mat = confusion_matrix(label, pred)
    # Plot non-normalized confusion matrix
    
    disp = ConfusionMatrixDisplay(con_mat)
    
    disp.plot()
    plt.title("confusion matrix")
    plt.savefig(save_path + fileName)
    plt.close()