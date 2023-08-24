import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import lightgbm as lgb
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)
import matplotlib.pyplot as plt # グラフ描画用
import mojimoji
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanAbsolutePercentageError



train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
label=train['price']
train=train.drop('price',axis=1)
train=train.drop('id',axis=1)
test=test.drop('id',axis=1)


#---------------------------------------------------------------------------------------------------------
# #前処理odometer
for i in range(train.shape[0]):
    if train['odometer'][i]==-131869:
        train['odometer'][i]=131869
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#前処理year:~2022
# print(type(train['year'][0]))
for i in range(train.shape[0]):
    if train['year'][i]>=2023:
        train['year'][i]=train['year'][i]-1000

for i in range(test.shape[0]):
    if test['year'][i]>=2023:
        test['year'][i]=test['year'][i]-1000
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#Feature Engineering：年間走行距離
train['annual_mileage']=train['odometer'].div(2023-train['year'])
test['annual_mileage']=test['odometer'].div(2023-test['year'])
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#欠損値がある行を01で示すfeature作成
train_miss=train.isnull().any(axis=1)
train_miss=pd.DataFrame(train_miss)
train_miss.columns=['miss']
train_miss=train_miss.astype(int)
# train=pd.concat([train,miss],axis=1)
test_miss=test.isnull().any(axis=1)
test_miss=pd.DataFrame(test_miss)
test_miss.columns=['miss']
test_miss=test_miss.astype(int)
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#前処理manufacturer 全角半角変換
train["manufacturer"]=train["manufacturer"].str.lower()
train["manufacturer"]=train["manufacturer"].apply(mojimoji.zen_to_han)
test["manufacturer"]=test["manufacturer"].str.lower()
test["manufacturer"]=test["manufacturer"].apply(mojimoji.zen_to_han)

#前処理cylinders　カテゴリ変数から数値変数
train['cylinders']=train['cylinders'].str.replace('6 cylinders','6')#11504
train['cylinders']=train['cylinders'].str.replace('8 cylinders','8')#5727
train['cylinders']=train['cylinders'].str.replace('4 cylinders','4')#10071
train['cylinders']=train['cylinders'].str.replace('10 cylinders','10')#60
train['cylinders']=train['cylinders'].str.replace('12 cylinders','12')#22
train['cylinders']=train['cylinders'].str.replace('5 cylinders','5')#46
train['cylinders']=train['cylinders'].str.replace('3 cylinders','3')#31
train['cylinders']=train['cylinders'].str.replace('other','-1')#71
train['cylinders']=train['cylinders'].astype(int)
test['cylinders']=test['cylinders'].str.replace('6 cylinders','6')#11504
test['cylinders']=test['cylinders'].str.replace('8 cylinders','8')#5727
test['cylinders']=test['cylinders'].str.replace('4 cylinders','4')#10071
test['cylinders']=test['cylinders'].str.replace('10 cylinders','10')#60
test['cylinders']=test['cylinders'].str.replace('12 cylinders','12')#22
test['cylinders']=test['cylinders'].str.replace('5 cylinders','5')#46
test['cylinders']=test['cylinders'].str.replace('3 cylinders','3')#31
test['cylinders']=test['cylinders'].str.replace('other','-1')#71
test['cylinders']=test['cylinders'].astype(int)

#前処理size　カテゴリ変数から数値変数
train['size']=train['size'].str.replace('full−size','full-size')
train['size']=train['size'].str.replace('fullーsize','full-size')
train['size']=train['size'].str.replace('full-size','4')
train['size']=train['size'].str.replace('mid−size','mid-size')
train['size']=train['size'].str.replace('midーsize','mid-size')
train['size']=train['size'].str.replace('mid-size','3')
train['size']=train['size'].str.replace('subーcompact','sub-compact')
train['size']=train['size'].str.replace('sub-compact','2')
train['size']=train['size'].str.replace('compact','1')
train['size']=train['size'].astype(int)
test['size']=test['size'].str.replace('full−size','full-size')
test['size']=test['size'].str.replace('fullーsize','full-size')
test['size']=test['size'].str.replace('full-size','4')
test['size']=test['size'].str.replace('mid−size','mid-size')
test['size']=test['size'].str.replace('midーsize','mid-size')
test['size']=test['size'].str.replace('mid-size','3')
test['size']=test['size'].str.replace('subーcompact','sub-compact')
test['size']=test['size'].str.replace('sub-compact','2')
test['size']=test['size'].str.replace('compact','1')
test['size']=test['size'].astype(int)

#前処理condition　カテゴリ変数から数値変数
train['condition']=train['condition'].str.replace('salvage','1')
train['condition']=train['condition'].str.replace('fair','2')
train['condition']=train['condition'].str.replace('good','3')
train['condition']=train['condition'].str.replace('excellent','4')
train['condition']=train['condition'].str.replace('like new','5')
train['condition']=train['condition'].str.replace('new','6')
train['condition']=train['condition'].astype(int)

test['condition']=test['condition'].str.replace('salvage','1')
test['condition']=test['condition'].str.replace('fair','2')
test['condition']=test['condition'].str.replace('good','3')
test['condition']=test['condition'].str.replace('excellent','4')
test['condition']=test['condition'].str.replace('like new','5')
test['condition']=test['condition'].str.replace('new','6')
test['condition']=test['condition'].astype(int)
#---------------------------------------------------------------------------------------------------------

train_lenc=train.copy()
test_lenc=test.copy()

#---------------------------------------------------------------------------------------------------------
#target encoding
enc_auto = TargetEncoder(smooth="auto",target_type="continuous")
enc_auto.fit(train.loc[:,['region']], label)
train.loc[:,['region']]=enc_auto.transform(train.loc[:,['region']])
test.loc[:,['region']]=enc_auto.transform(test.loc[:,['region']])
enc_auto.fit(train.loc[:,['state']], label)
train.loc[:,['state']]=enc_auto.transform(train.loc[:,['state']])
test.loc[:,['state']]=enc_auto.transform(test.loc[:,['state']])
enc_auto.fit(train.loc[:,['type']], label)
train.loc[:,['type']]=enc_auto.transform(train.loc[:,['type']])
test.loc[:,['type']]=enc_auto.transform(test.loc[:,['type']])
enc_auto.fit(train.loc[:,['paint_color']], label)
train.loc[:,['paint_color']]=enc_auto.transform(train.loc[:,['paint_color']])
test.loc[:,['paint_color']]=enc_auto.transform(test.loc[:,['paint_color']])
enc_auto.fit(train.loc[:,['manufacturer']], label)
train.loc[:,['manufacturer']]=enc_auto.transform(train.loc[:,['manufacturer']])
test.loc[:,['manufacturer']]=enc_auto.transform(test.loc[:,['manufacturer']])
enc_auto.fit(train.loc[:,['fuel']], label)
train.loc[:,['fuel']]=enc_auto.transform(train.loc[:,['fuel']])
test.loc[:,['fuel']]=enc_auto.transform(test.loc[:,['fuel']])
enc_auto.fit(train.loc[:,['title_status']], label)
train.loc[:,['title_status']]=enc_auto.transform(train.loc[:,['title_status']])
test.loc[:,['title_status']]=enc_auto.transform(test.loc[:,['title_status']])
enc_auto.fit(train.loc[:,['transmission']], label)
train.loc[:,['transmission']]=enc_auto.transform(train.loc[:,['transmission']])
test.loc[:,['transmission']]=enc_auto.transform(test.loc[:,['transmission']])
enc_auto.fit(train.loc[:,['drive']], label)
train.loc[:,['drive']]=enc_auto.transform(train.loc[:,['drive']])
test.loc[:,['drive']]=enc_auto.transform(test.loc[:,['drive']])
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#neural net用のone-hot encoding
import pickle
for i in ["paint_color","type","fuel","title_status"]:
    le = LabelEncoder()
    enctra = le.fit_transform(train_lenc[i])
    train_lenc[i]=enctra
    with open('label.pickle', 'wb') as web:
        pickle.dump(le , web)
    with open('label.pickle', 'rb') as web:
        le = pickle.load(web)
    enctes = le.transform(test_lenc[i])
    test_lenc[i]=enctes
    

train['paint_color_lenc']=train_lenc['paint_color']
train['type_lenc']=train_lenc['type']
train['fuel_lenc']=train_lenc['fuel']
train['title_status_lenc']=train_lenc['title_status']
test['paint_color_lenc']=test_lenc['paint_color']
test['type_lenc']=test_lenc['type']
test['fuel_lenc']=test_lenc['fuel']
test['title_status_lenc']=test_lenc['title_status']


cat_cols = ['paint_color_lenc', 'type_lenc', 'fuel_lenc', 'title_status_lenc']
# 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う
all_x = pd.concat([train, test])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 学習データとテストデータに再分割
train = all_x.iloc[:train.shape[0], :].reset_index(drop=True)
test = all_x.iloc[train.shape[0]:, :].reset_index(drop=True)
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
ms = MinMaxScaler()
#標準化(それぞれでfit_transform)
train = ms.fit_transform(train)
test = ms.fit_transform(test)
# #---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#FSFW
mi=mutual_info_regression(train, label)
train=train*mi
test=test*mi
#---------------------------------------------------------------------------------------------------------

#ここで、missのfeatureを追加
train_miss = ms.fit_transform(train_miss)
test_miss=ms.fit_transform(test_miss)
train=np.hstack([train,train_miss])
test=np.hstack([test,test_miss])
train_data,valid_data,train_label,valid_label=train_test_split(train,label,train_size=0.7,random_state=1)
train_data = torch.Tensor(train_data)
valid_data = torch.Tensor(valid_data)
train_label = torch.Tensor(train_label.to_numpy())
valid_label = torch.Tensor(valid_label.to_numpy())

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(valid_data, valid_label)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
class NNModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Identity()
        )
    
    def forward(self, x):
        y = self.layers(x)
        return y
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
model = NNModel(train_data.size()[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
n_epochs = 100
for epoch in range(1, n_epochs + 1):
    model.train()
    i=-1
    for x_batch, y_batch in train_loader:
        i=i+1
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        yhat = model(x_batch)
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{10}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluate the model on the test data
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
    mean_loss = total_loss / len(test_loader)
    print(f'Test Loss: {mean_loss:.4f}')


t_pred=model(train_data).detach().cpu().numpy()
v_pred=model(valid_data).detach().cpu().numpy()
plt.scatter(t_pred, train_label)
plt.xlabel("train_predict")
plt.ylabel("train_label")
plt.show()
plt.scatter(v_pred, valid_label)
plt.xlabel("valid_predict")
plt.ylabel("valid_label")
plt.show()

#---------------------------------------------------------------------------------------------------------
# モデル評価
#mean_absolute_percentage_error
t_m = mean_absolute_percentage_error(train_label,t_pred)
print('train error :',t_m)
v_m = mean_absolute_percentage_error(valid_label,v_pred)
print('valid error :',v_m)
#---------------------------------------------------------------------------------------------------------

# pred=model(test).detach().cpu().numpy()
# sub = pd.read_csv('submit_sample.csv', encoding = 'UTF-8', names=['id', 'ans'])
# sub['ans'] = pred
# sub.to_csv("first.csv", header=False, index=False)