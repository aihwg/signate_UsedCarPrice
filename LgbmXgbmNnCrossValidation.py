import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense,  Activation, Dropout, BatchNormalization
from sklearn import preprocessing
# from keras.initializers import he_normal,glorot_normal
# import xgboost as xgb
from keras.utils import np_utils
import lightgbm as lgb
from sklearn.model_selection import KFold

#accuracyを出すための関数
def acc(a,b):
    tt=0
    for i in range(a.size):
        if b[i]==a[i]:
            tt=tt+1
    c=tt/a.size
    return c


train = pd.read_table("train.tsv", encoding="utf-8")
test = pd.read_table("test.tsv", encoding="utf-8")
x = train.iloc[:,1:10]
x=x.drop("Na", axis=1)
x=x.drop("Fe", axis=1)
x=x.drop("Si", axis=1)
x=x.drop("K", axis=1)
x=x.drop("Ca", axis=1)
y = train["Type"]
test = test.iloc[:,1:10]
test=test.drop("Na", axis=1)
test=test.drop("Fe", axis=1)
test=test.drop("Si", axis=1)
test=test.drop("K", axis=1)
test=test.drop("Ca", axis=1)
x=x.values#pandasからndarray
y=y.values
test=test.values
x = preprocessing.scale(x)  # 標準化
y = np_utils.to_categorical(y)  # one-hotへ。Typeは1~7(6はなし)になってて、one-hotにすると0が含まれるから、狂う
test=preprocessing.scale(test)


#lgb
lgb_model0= lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)
lgb_model1= lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)
lgb_model2= lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)
lgb_model3= lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)
lgb_model4= lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)


callbacks = []
callbacks.append(lgb.early_stopping(stopping_rounds=10))
callbacks.append(lgb.log_evaluation())

# lp=lgb_model.predict(x_test)
# lptr=lgb_model.predict(x_train)

#xgb
xgb_model0 = xgb.XGBClassifier()
xgb_model1 = xgb.XGBClassifier()
xgb_model2= xgb.XGBClassifier()
xgb_model3 = xgb.XGBClassifier()
xgb_model4 = xgb.XGBClassifier()

#nn
model0=Sequential()
model0.add(Dense(10, input_dim=4, kernel_initializer=he_normal()))    # 入力層4ノード, 隠れ層に10ノード, 全結合
model0.add(Activation("relu"))
model0.add(BatchNormalization())
model0.add(Dense(12))
model0.add(Activation("relu"))
model0.add(BatchNormalization())
model0.add(Dense(10))
model0.add(Activation("relu"))
model0.add(BatchNormalization())
model0.add(Dense(8))
model0.add(Activation("softmax"))
model0.compile(loss="categorical_crossentropy",   # 誤差関数
              optimizer="adam",     # 最適化手法
              metrics=['accuracy'])
model1=model0
model2=model0
model3=model0
model4=model0


# #Not One-hot
# ytr_l=np.argmax(y_train,axis=1)
# yte_l=np.argmax(y_test,axis=1)
# lgb_model.fit(x_train, ytr_l, eval_set=eval_set, callbacks=callbacks)
# xgb_model.fit(x_train, y_train)
# model.fit(x_train, y_train, epochs=1000, batch_size=32)

cv = KFold(n_splits=5, random_state=0, shuffle=True)
mse_list = []
i=-1
for train_index, test_index in cv.split(x):
    # get train and test data
    X_train, X_test = x[train_index], x[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    Y_train_noh=np.argmax(Y_train,axis=1)
    print(Y_train_noh)
    Y_test_noh=np.argmax(Y_test,axis=1)
    eval_set = [(X_test, Y_test_noh)]
    i=i+1
    # predict test data
    if i==0:
        model0.fit(X_train, Y_train, epochs=50, batch_size=16)
        xgb_model0.fit(X_train,Y_train)
        lgb_model0.fit(X_train,Y_train_noh,eval_set=eval_set, callbacks=callbacks)
        pretes_m_0=model0.predict(X_test)
        pretes_m_0=np.argmax(pretes_m_0,axis=1)
        pretes_x_0=xgb_model0.predict(X_test)
        pretes_x_0=np.argmax(pretes_x_0,axis=1)
        pretes_l_0=lgb_model0.predict(X_test)
        acc_m_0=acc(pretes_m_0,Y_test_noh)
        acc_x_0=acc(pretes_x_0,Y_test_noh)
        acc_l_0=acc(pretes_l_0,Y_test_noh)
    elif i==1:
        model1.fit(X_train, Y_train, epochs=50, batch_size=16)
        xgb_model1.fit(X_train,Y_train)
        lgb_model1.fit(X_train,Y_train_noh,eval_set=eval_set, callbacks=callbacks)
        pretes_m_1=model1.predict(X_test)
        pretes_m_1=np.argmax(pretes_m_1,axis=1)
        pretes_x_1=xgb_model1.predict(X_test)
        pretes_x_1=np.argmax(pretes_x_1,axis=1)
        pretes_l_1=lgb_model1.predict(X_test)
        acc_m_1=acc(pretes_m_1,Y_test_noh)
        acc_x_1=acc(pretes_x_1,Y_test_noh)
        acc_l_1=acc(pretes_l_1,Y_test_noh)
    elif i==2:
        model2.fit(X_train, Y_train, epochs=50, batch_size=16)
        xgb_model2.fit(X_train,Y_train)
        lgb_model2.fit(X_train,Y_train_noh,eval_set=eval_set, callbacks=callbacks)
        pretes_m_2=model2.predict(X_test)
        pretes_m_2=np.argmax(pretes_m_2,axis=1)
        pretes_x_2=xgb_model2.predict(X_test)
        pretes_x_2=np.argmax(pretes_x_2,axis=1)
        pretes_l_2=lgb_model2.predict(X_test)
        acc_m_2=acc(pretes_m_2,Y_test_noh)
        acc_x_2=acc(pretes_x_2,Y_test_noh)
        acc_l_2=acc(pretes_l_2,Y_test_noh)
    elif i==3:
        model3.fit(X_train, Y_train, epochs=50, batch_size=16)
        xgb_model3.fit(X_train,Y_train)
        lgb_model3.fit(X_train,Y_train_noh,eval_set=eval_set, callbacks=callbacks)
        pretes_m_3=model3.predict(X_test)
        pretes_m_3=np.argmax(pretes_m_3,axis=1)
        pretes_x_3=xgb_model3.predict(X_test)
        pretes_x_3=np.argmax(pretes_x_3,axis=1)
        pretes_l_3=lgb_model3.predict(X_test)
        acc_m_3=acc(pretes_m_3,Y_test_noh)
        acc_x_3=acc(pretes_x_3,Y_test_noh)
        acc_l_3=acc(pretes_l_3,Y_test_noh)
    else:
        model4.fit(X_train, Y_train, epochs=50, batch_size=16)
        xgb_model4.fit(X_train,Y_train)
        lgb_model4.fit(X_train,Y_train_noh,eval_set=eval_set, callbacks=callbacks)
        pretes_m_4=model4.predict(X_test)
        pretes_m_4=np.argmax(pretes_m_4,axis=1)
        pretes_x_4=xgb_model4.predict(X_test)
        pretes_x_4=np.argmax(pretes_x_4,axis=1)
        pretes_l_4=lgb_model4.predict(X_test)
        acc_m_4=acc(pretes_m_4,Y_test_noh)
        acc_x_4=acc(pretes_x_4,Y_test_noh)
        acc_l_4=acc(pretes_l_4,Y_test_noh)
# # 各モデル評価
print("acc_m")
print(acc_m_0)
print(acc_m_1)
print(acc_m_2)
print(acc_m_3)
print(acc_m_4)
print("acc_x")
print(acc_x_0)
print(acc_x_1)
print(acc_x_2)
print(acc_x_3)
print(acc_x_4)
print("acc_l")
print(acc_l_0)
print(acc_l_1)
print(acc_l_2)
print(acc_l_3)
print(acc_l_4)
# # score = model.evaluate(x_test, y_test, verbose=1)
# # print("nn:Test score", score[0])
# # print("nn:Test accuracy", score[1])
# # print("xgb:train acc",xgb_model.score(x_train,y_train))
# # print("xgb:test acc",xgb_model.score(x_test,y_test))
# # print("lgb:train acc",acc(lptr,ytr_l))
# # print("lgb:test acc",acc(lp,yte_l))

predicted_M_0=model0.predict(test)
predicted_M_1=model1.predict(test)
predicted_M_2=model2.predict(test)
predicted_M_3=model3.predict(test)
predicted_M_4=model4.predict(test)
predicted_X_0=xgb_model0.predict(test)
predicted_X_1=xgb_model1.predict(test)
predicted_X_2=xgb_model2.predict(test)
predicted_X_3=xgb_model3.predict(test)
predicted_X_4=xgb_model4.predict(test)
predicted_L_0=lgb_model0.predict(test)
predicted_L_1=lgb_model1.predict(test)
predicted_L_2=lgb_model2.predict(test)
predicted_L_3=lgb_model3.predict(test)
predicted_L_4=lgb_model4.predict(test)

predicted_X_0=np.argmax(predicted_X_0,axis=1)
predicted_X_1=np.argmax(predicted_X_1,axis=1)
predicted_X_2=np.argmax(predicted_X_2,axis=1)
predicted_X_3=np.argmax(predicted_X_3,axis=1)
predicted_X_4=np.argmax(predicted_X_4,axis=1)
predicted_M_0=np.argmax(predicted_M_0,axis=1)
predicted_M_1=np.argmax(predicted_M_1,axis=1)
predicted_M_2=np.argmax(predicted_M_2,axis=1)
predicted_M_3=np.argmax(predicted_M_3,axis=1)
predicted_M_4=np.argmax(predicted_M_4,axis=1)

# print(predicted_L_0)
# print(predicted_M_0)
# print(predicted_X_0)
# print(predicted_L_1)
# print(predicted_M_1)
# print(predicted_X_1)

pred=np.array([],dtype='int64')
#voting
for i in range(predicted_X_0.size):
    preds=[*map(lambda x:np.argmax(np.bincount(x)), np.array([[predicted_X_0[i]],
                                                            [predicted_X_1[i]],
                                                            [predicted_X_2[i]],
                                                            [predicted_X_3[i]],
                                                            [predicted_X_4[i]],
                                                            [predicted_M_0[i]],
                                                            [predicted_M_1[i]],
                                                            [predicted_M_2[i]],
                                                            [predicted_M_3[i]],
                                                            [predicted_M_4[i]],
                                                            [predicted_M_0[i]],
                                                            [predicted_L_0[i]],
                                                            [predicted_L_1[i]],
                                                            [predicted_L_2[i]],
                                                            [predicted_L_3[i]],
                                                            [predicted_L_4[i]]]).T)]
    pred=np.append(pred,preds[0])
print("pred")
print(pred)

sub = pd.read_csv('sample_submit.csv', encoding = 'UTF-8', names=['id', 'ans'])
# print(sub)
sub['ans'] = pred
sub.to_csv("LXNCV.csv", header=False, index=False)