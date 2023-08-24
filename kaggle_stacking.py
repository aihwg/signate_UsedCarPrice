# ---------------------------------
# スタッキング
# ----------------------------------
from sklearn.metrics import log_loss,mean_absolute_percentage_error
from sklearn.model_selection import KFold
# models.pyにModel1Xgb, Model1NN, Model2Linearを定義しているものとする
# 各クラスは、fitで学習し、predictで予測値の確率を出力する
from models import Model1Xgb, Model1NN, Model2Linear
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
# print(train['region'].nunique())
# print(train['type'])
# print(train.info())
# print(train.shape[0]-train.count())
# train.hist()
# plt.tight_layout()
# plt.show()


# #########
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

train=pd.concat([train,train_miss],axis=1)
test=pd.concat([test,test_miss],axis=1)
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
#labelencoder
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
#---------------------------------------------------------------------------------------------------------
cat_cols = ['paint_color_lenc', 'type_lenc', 'fuel_lenc', 'title_status_lenc']
# 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う
all_x = pd.concat([train, test])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 学習データとテストデータに再分割
train_nn = all_x.iloc[:train.shape[0], :].reset_index(drop=True)
test_nn = all_x.iloc[train.shape[0]:, :].reset_index(drop=True)


#---------------------------------------------------------------------------------------------------------
ms = MinMaxScaler()

scaling_columns = ['region','state','type','paint_color','manufacturer','fuel','title_status','transmission','drive','cylinders','size','condition','annual_mileage','year','odometer','paint_color_lenc','type_lenc','fuel_lenc','title_status_lenc'] 
ms = MinMaxScaler().fit(train[scaling_columns])
scaled_train = pd.DataFrame(ms.transform(train[scaling_columns]), columns=scaling_columns, index=train.index)
train.update(scaled_train)
scaling_columns = ['region','state','type','paint_color','manufacturer','fuel','title_status','transmission','drive','cylinders','size','condition','annual_mileage','year','odometer','paint_color_lenc','type_lenc','fuel_lenc','title_status_lenc'] 
ms = MinMaxScaler().fit(test[scaling_columns])
scaled_test = pd.DataFrame(ms.transform(test[scaling_columns]), columns=scaling_columns, index=test.index)
test.update(scaled_train)
scaling_columns = ['region','state','manufacturer','transmission','drive','cylinders','size','condition','annual_mileage','year','odometer'] 
ms = MinMaxScaler().fit(test_nn[scaling_columns])
scaled_test_nn = pd.DataFrame(ms.transform(test_nn[scaling_columns]), columns=scaling_columns, index=test_nn.index)
test_nn.update(scaled_train)
scaling_columns = ['region','state','manufacturer','transmission','drive','cylinders','size','condition','annual_mileage','year','odometer'] 
ms = MinMaxScaler().fit(train_nn[scaling_columns])
scaled_train_nn = pd.DataFrame(ms.transform(train_nn[scaling_columns]), columns=scaling_columns, index=train_nn.index)
train_nn.update(scaled_train_nn)
# #---------------------------------------------------------------------------------------------------------


# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test


# 1層目のモデル
# pred_train_1a, pred_train_1bは、学習データのクロスバリデーションでの予測値
# pred_test_1a, pred_test_1bは、テストデータの予測値
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train, label, test)

model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_nn, label, test_nn)

# 1層目のモデルの評価
print(f'logloss: {log_loss(label, pred_train_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(label, pred_train_1b, eps=1e-7):.4f}')

# 予測値を特徴量としてデータフレームを作成
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

# 2層目のモデル
# pred_train_2は、2層目のモデルの学習データのクロスバリデーションでの予測値
# pred_test_2は、2層目のモデルのテストデータの予測値
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, label, test_x_2)
print(f'logloss: {log_loss(label, pred_train_2, eps=1e-7):.4f}')