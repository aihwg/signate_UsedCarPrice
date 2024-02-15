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
from sklearn.metrics import mean_absolute_percentage_error
import umap
from geopy.geocoders import OpenCage
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

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
# #enbedding word2vec
# # from gensim.models import KeyedVectors  #GoogleNews-vectors-negative300.bin
# # # Load vectors directly from the file
# # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# # print(model["statecollege"])
# # print(model["statecollege"].shape)
# # # print(model[train['region'].to_list()])

# from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
# train_resion = sbert_model.encode(train['region'].to_list())
# mapper = umap.UMAP(random_state=0)
# train_resion = mapper.fit_transform(train_resion)
# test_resion = sbert_model.encode(test['region'].to_list())
# test_resion = mapper.fit_transform(test_resion)

# train=train.drop('region',axis=1)
# test=test.drop('region',axis=1)
# train_resion=pd.DataFrame(train_resion,columns=['region_0','region_1'])
# test_resion=pd.DataFrame(test_resion,columns=['region_0','region_1'])
# train=train.assign(r特徴量生成特徴量生成_0=train_resion['region_0'],resion_1=train_resion['region_1'])
# test=test.assign(resion_0=test_resion['region_0'],resion_1=test_resion['region_1'])

# cities = np.array(['florence / muscle shoals', 'New York', 'London'])
# # cities=train['region'].to_numpy()
# geolocator = GoogleV3(api_key='YOUR_API_KEY_HERE')
# def get_location(city):
#     try:
#         return geolocator.geocode(city)
#     except (GeocoderTimedOut, GeocoderUnavailable):
#         return None
# with ThreadPoolExecutor() as executor:
#     locations = list(executor.map(get_location, cities))
# df = pd.DataFrame({'City': cities, 'Latitude': [location.latitude if location else None for location in locations], 'Longitude': [location.longitude if location else None for location in locations]})
# print(df)
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
# cat_cols = ['paint_color_lenc', 'type_lenc', 'fuel_lenc', 'title_status_lenc']
# # 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う
# all_x = pd.concat([train, test])
# all_x = pd.get_dummies(all_x, columns=cat_cols)

# # 学習データとテストデータに再分割
# train = all_x.iloc[:train.shape[0], :].reset_index(drop=True)
# test = all_x.iloc[train.shape[0]:, :].reset_index(drop=True)


# #---------------------------------------------------------------------------------------------------------
# #umapで特徴量生成
# mapper = umap.UMAP(random_state=0)
# train_umap = mapper.fit_transform(train)
# train_umap=pd.DataFrame(train_umap,columns=['umap_0','umap_1'])
# train=train.assign(umap_0=train_umap['umap_0'],umap_1=train_umap['umap_1'])
# test_umap = mapper.fit_transform(test)
# test_umap=pd.DataFrame(test_umap,columns=['umap_0','umap_1'])
# test=test.assign(umap_0=test_umap['umap_0'],umap_1=test_umap['umap_1'])
# #---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#数値データをrankgaus
#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
# #標準化（trainとtestをconcatしてから、標準化）
# train_test=pd.concat([train,test])
ms = MinMaxScaler()
# ms.fit(train_test)
# train=ms.transform(train)
# test=ms.transform(test)

# # 標準化（trainでfit）
# ms.fit(train)
# train=ms.transform(train)
# test=ms.transform(test)
# print(train.min(),train.max())
# print(test.min(),test.max())
#標準化(それぞれでfit_transform)
train = ms.fit_transform(train)
test = ms.fit_transform(test)
# #---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#FSFW
mi=mutual_info_regression(train, label)
train=train*mi
test=test*mi
# print(mi)
# z=0
# zz=[]
# for i in range(len(mi)):
#     if 0.05>=mi[i]:
#         zz.append(i)
#     z=z+1
# train=np.delete(train,zz,1)
# test=np.delete(test,zz,1)
#---------------------------------------------------------------------------------------------------------

#ここで、missのfeatureを追加
train_miss = ms.fit_transform(train_miss)
test_miss=ms.fit_transform(test_miss)
train=np.hstack([train,train_miss])
test=np.hstack([test,test_miss])
train_data,valid_data,train_label,valid_label=train_test_split(train,label,train_size=0.7,random_state=1)



reg = StackingRegressor(
    estimators=[
                ('lgb_model0', lgb.LGBMRegressor(random_state=42)),
                ('lgb_model1', lgb.LGBMRegressor(random_state=42,max_depth=5,num_leaves=28)),
                ('lgb_model2', lgb.LGBMRegressor(random_state=42,max_depth=6,num_leaves=60)),
                ('regr0', RandomForestRegressor(random_state=0,max_depth=8,n_estimators=200)),
                ('regr1', RandomForestRegressor(random_state=0,max_depth=10,n_estimators=300)),
                ],
    final_estimator=lgb.LGBMRegressor(random_state=42)
)
reg.fit(train_data, train_label)
#---------------------------------------------------------------------------------------------------------
print('train error :',mean_absolute_percentage_error(train_label,reg.predict(train_data)))
print('valid error :',mean_absolute_percentage_error(valid_label,reg.predict(valid_data)))
#---------------------------------------------------------------------------------------------------------
plt.scatter(reg.predict(train_data), train_label)
plt.xlabel("train_predict")
plt.ylabel("train_label")
plt.show()
plt.scatter(reg.predict(valid_data), valid_label)
plt.xlabel("valid_predict")
plt.ylabel("valid_label")
plt.show()

pred=reg.predict(test)
print(pred)
sub = pd.read_csv('submit_sample.csv', encoding = 'UTF-8', names=['id', 'ans'])
sub['ans'] = pred
sub.to_csv("first.csv", header=False, index=False)