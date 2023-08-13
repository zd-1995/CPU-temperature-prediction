from sklearn.preprocessing import *
from sklearn.metrics import *
from selectfeatures import *
from ReadData import *
from sklearn.model_selection import *
from NN import *
import numpy as np
from Results import *
from sklearn.preprocessing import MultiLabelBinarizer



address = 'data_asli.csv'
data_1 = read_file(address)
print(np.shape(data_1))
# t=[]
# for x in range(2792):
#     for z in x:
#        if np.isnan(data_1[x][z]=='True'):
#            print(x,z)

data = data_1[0:2800,0:19]
# [m,n]=np.shape(data)
# print(m,'*********')
label_1 = data_1[0:2800,19:20]
print(np.shape(label_1))
# print(label_1)


Scaler_x = StandardScaler()
Scaler_y = StandardScaler()

X1= Scaler_x.fit_transform(data)

y = Scaler_y.fit_transform(label_1)

y_1 = y[:,0]


(X_train_1, X_test_1, Y_train_1, Y_test_1) = train_test_split(data,label_1, test_size=0.15, shuffle=False,random_state=False)

X_train_fs_1 = X_train_1
X_test_fs_1 = X_test_1

print(np.shape(X_train_fs_1),np.shape(X_test_fs_1),np.shape(Y_train_1),np.shape(Y_test_1))

sec_2=[]
lab_2=[]
sec_3=[]
lab_3=[]
sec_4=[]
lab_4=[]
sec_5=[]
lab_5=[]
for x in range(420):
    if X_test_fs_1[x][15]==2:
        sec_2.append(X_test_fs_1[x][0:19])
        lab_2.append(Y_test_1[x][:])
    elif X_test_fs_1[x][15]==3:
        sec_3.append(X_test_fs_1[x][0:19])
        lab_3.append(Y_test_1[x][:])
    elif X_test_fs_1[x][15]==4:
        sec_4.append(X_test_fs_1[x][0:19])
        lab_4.append(Y_test_1[x][:])
    elif X_test_fs_1[x][15]==5:
        sec_5.append(X_test_fs_1[x][0:19])
        lab_5.append(Y_test_1[x][:])
sec_2=np.array(sec_2)
lab_2=np.array(lab_2)
sec_3=np.array(sec_3)
lab_3=np.array(lab_3)
sec_4=np.array(sec_4)
lab_4=np.array(lab_4)
sec_5=np.array(sec_5)
lab_5=np.array(lab_5)
print(np.shape(sec_2))

# CNN

# cnn_1 = cnn_model(X_train_fs_1,Y_train_1,X_test_fs_1,Y_test_1)
# For Second 2
cnn_2 = cnn_model(X_train_fs_1,Y_train_1,sec_2,lab_2)
# For Second 3
cnn_3 = cnn_model(X_train_fs_1,Y_train_1,sec_3,lab_3)
# For Second 4
cnn_4 = cnn_model(X_train_fs_1,Y_train_1,sec_4,lab_4)
# For Second 5
cnn_5 = cnn_model(X_train_fs_1,Y_train_1,sec_5,lab_5)

# LSTM
lstm_1 = lstm_model(X_train_fs_1,Y_train_1,X_test_fs_1,Y_test_1,256)
# For Second 2
lstm_2 = lstm_model(X_train_fs_1,Y_train_1,sec_2,lab_2,256)
# For Second 3
lstm_3 = lstm_model(X_train_fs_1,Y_train_1,sec_3,lab_3,256)
# For Second 4
lstm_4 = lstm_model(X_train_fs_1,Y_train_1,sec_4,lab_4,256)
# For Second 5
lstm_5 = lstm_model(X_train_fs_1,Y_train_1,sec_5,lab_5,256)

print('CNN')
print('For Second 2:')
print('\n','MAE:',cnn_2[4],'\n','Pe:',cnn_2[3],'\n','SDAE:',cnn_2[5])
print('For Second 3:')
print('\n','MAE:',cnn_3[4],'\n','Pe:',cnn_3[3],'\n','SDAE:',cnn_3[5])
print('For Second 4:')
print('\n','MAE:',cnn_4[4],'\n','Pe:',cnn_4[3],'\n','SDAE:',cnn_4[5])
print('For Second 5:')
print('\n','MAE:',cnn_5[4],'\n','Pe:',cnn_5[3],'\n','SDAE:',cnn_5[5])
print('LSTM')
print('For Second 2:')
print('\n','MAE:',lstm_2[4],'\n','Pe:',lstm_2[3],'\n','SDAE:',lstm_2[5])
print('For Second 3:')
print('\n','MAE:',lstm_3[4],'\n','Pe:',lstm_3[3],'\n','SDAE:',lstm_3[5])
print('For Second 4:')
print('\n','MAE:',lstm_4[4],'\n','Pe:',lstm_4[3],'\n','SDAE:',lstm_4[5])
print('For Second 5:')
print('\n','MAE:',lstm_5[4],'\n','Pe:',lstm_5[3],'\n','SDAE:',lstm_5[5])

# h = [16,32,64,128,256]
# mae = []
# for k in h:
#     lstm_1 = lstm_model(X_train_fs_1,Y_train_1,X_test_fs_1,Y_test_1,k)
#     mae.append(lstm_1[4])
# # lstm_2 = lstm_model(X_train_fs_2,Y_train_2,X_test_fs_2,Y_test_2)
# mae = np.array(mae)
# print(np.shape(mae))
# print('MLP','\n','MAE:',mlp_1[0],'Pe:',mlp_1[1],'SDAE:',mlp_1[2],'\n')
# print('CNN','\n','MAE:',cnn_1[4],'\n','Pe:',cnn_1[3],'\n','SDAE:',cnn_1[5],'\n')
print('LSTM','\n','MAE:',lstm_1[4],'\n','Pe:',lstm_1[3],'\n','SDAE:',lstm_1[5])

hist_df = pd.DataFrame(Y_test_1)
hist_csv_file = 'xtest.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
hist = pd.DataFrame(lstm_1[6])
hist_file = 'result_diff.csv'
with open(hist_file, mode='w') as z:
    hist.to_csv(z)
hist_p = pd.DataFrame(lstm_1[7])
hist_filep = 'result_pre.csv'
with open(hist_filep, mode='w') as zp:
    hist_p.to_csv(zp)

hist2 = pd.DataFrame(lstm_2[6])
hist_file2 = 'result_diff_2.csv'
with open(hist_file2, mode='w') as z2:
    hist2.to_csv(z2)
hist3 = pd.DataFrame(lstm_3[6])
hist_file3 = 'result_diff_3.csv'
with open(hist_file3, mode='w') as z3:
    hist3.to_csv(z3)
hist4 = pd.DataFrame(lstm_4[6])
hist_file4 = 'result_diff_4.csv'
with open(hist_file4, mode='w') as z4:
    hist4.to_csv(z4)
hist5 = pd.DataFrame(lstm_5[6])
hist_file5 = 'result_diff_5.csv'
with open(hist_file5, mode='w') as z5:
    hist5.to_csv(z5)


# df = pd.DataFrame(cnn_1[3:6])
# hist_csv = 'result_cnn.csv'
# with open(hist_csv, mode='w') as k:
#     df.to_csv(k)


lstm_2[1].save('l1.h5')
cnn_1[1].save('c1.h5')