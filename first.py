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
import mojimoji
from sklearn.preprocessing import TargetEncoder

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
label=train['price']
train=train.drop('price',axis=1)
train=train.drop('id',axis=1)
test=test.drop('id',axis=1)
# print(train.info())
# print(train.shape[0]-train.count())
# train.hist()
# plt.tight_layout()
# plt.show()
# print(train['odometer'].min())

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
# test=pd.concat([test,miss],axis=1)

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


#target encoding
enc_auto = TargetEncoder(smooth="auto",target_type="continuous")
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


# #########
# train=train.drop('price',axis=1)
# train=train.drop('id',axis=1)
# train=train.drop('year',axis=1)
# train=train.drop('odometer',axis=1)

# train=train.to_numpy()
# # print(train.info())
# for i in range(train.shape[1]):
#     dictT={}
#     k=0
#     for j in range(train.shape[0]):
#         if train[j][i] not in dictT.keys():
#             dictT[train[j][i]]=k
#             k=k+1
#     print(dictT.keys(),'\n')
# #########
# #########
# test=test.drop('id',axis=1)
# test=test.drop('year',axis=1)
# test=test.drop('odometer',axis=1)

# test=test.to_numpy()
# # print(train.info())
# for i in range(test.shape[1]):
#     dictT={}
#     k=0
#     for j in range(test.shape[0]):
#         if test[j][i] not in dictT.keys():
#             dictT[test[j][i]]=k
#             k=k+1
#     print(dictT.keys(),'\n')
# #########



# # 欠損値を表示
# print(train.shape[0]-train.count())#fuel 1239,title_status 456,type 456,state 3304

#labelencoder
import pickle
for i in ['region','condition','fuel','title_status','transmission','drive']:
    le = LabelEncoder()
    enctra = le.fit_transform(train[i])
    train[i]=enctra
    with open('label.pickle', 'wb') as web:
        pickle.dump(le , web)
    with open('label.pickle', 'rb') as web:
        le = pickle.load(web)
    enctes = le.transform(test[i])
    test[i]=enctes



#標準化（trainでfit）
ms = MinMaxScaler()
# ms.fit(train)
# train=ms.transform(train)
# test=ms.transform(test)
#標準化(それぞれでfit_transform)
train = ms.fit_transform(train)
test = ms.fit_transform(test)

#FSFW
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

#ここで、missのfeatureを追加
train_miss = ms.fit_transform(train_miss)
test_miss=ms.fit_transform(test_miss)
train=np.hstack([train,train_miss])
test=np.hstack([test,test_miss])

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