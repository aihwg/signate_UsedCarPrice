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
import seaborn as sns; sns.set() # グラフ描画用

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print(train.shape)
print(test.shape)
# # # 欠損値を表示
# # print(train.shape[0]-train.count())#fuel 1239,title_status 456,type 456,state 3304

#labelencoder
# train.columns.values
import pickle
for i in ['region','manufacturer','condition','cylinders','fuel','title_status','transmission','drive','size','type','paint_color','state']:
    le = LabelEncoder()
    enctra = le.fit_transform(train[i])
    train[i]=enctra
    with open('label.pickle', 'wb') as web:
        pickle.dump(le , web)
    with open('label.pickle', 'rb') as web:
        le = pickle.load(web)
    enctes = le.transform(test[i])
    test[i]=enctes

label=train['price']
train=train.drop('price',axis=1)
train=train.drop('id',axis=1)
test=test.drop('id',axis=1)

ms = MinMaxScaler()
train = ms.fit_transform(train)
test = ms.fit_transform(test)

mi=mutual_info_regression(train, label)
train=train*mi
test=test*mi
# z=0
# zz=[]
# for i in range(len(mi)):
#     if 0.1>=mi[i]:
#         zz.append(i)
#     z=z+1
# train=np.delete(train,zz,1)
# test=np.delete(test,zz,1)

train_data,valid_data,train_label,valid_label=train_test_split(train,label,train_size=0.7)
lgb_train = lgb.Dataset(train_data, train_label)
lgb_eval = lgb.Dataset(valid_data, valid_label, reference=lgb_train) 

# LightGBM parameters
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression', # 目的 : 回帰  
        'metric': {'rmse'}, # 評価指標 : rsme(平均二乗誤差の平方根) 
}

# モデルの学習
model = lgb.train(params,
                  train_set=lgb_train, # トレーニングデータの指定
                  valid_sets=lgb_eval, # 検証データの指定
                  )

# テストデータの予測
lgb_pred = model.predict(valid_data)
df_pred = pd.DataFrame({'CRIM':valid_label,'CRIM_pred':lgb_pred})
print(df_pred)

# モデル評価
# rmse : 平均二乗誤差の平方根
mse = mean_squared_error(valid_label, lgb_pred) # MSE(平均二乗誤差)の算出
rmse = np.sqrt(mse) # RSME = √MSEの算出
print('RMSE :',rmse)

#r2 : 決定係数
r2 = r2_score(valid_label,lgb_pred)
print('R2 :',r2)

pred=model.predict(test)
sub = pd.read_csv('submit_sample.csv', encoding = 'UTF-8', names=['id', 'ans'])
# print(sub)
sub['ans'] = pred
sub.to_csv("first.csv", header=False, index=False)