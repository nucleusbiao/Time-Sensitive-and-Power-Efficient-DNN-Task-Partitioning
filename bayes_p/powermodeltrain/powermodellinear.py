#24.11.1   powermodel 
#训练设备的功耗模型
#功耗和cpu利用率数据由powercollection 获得,记录在：/latency/bayeslog*.py。
#使用前，将记录的设备功耗数据整理至excel。给出的例子里，为用户设备u的数据，在：data25.1.6powermodelu.xlsx
#注意各区间的数据比重均衡。

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import numpy as np
#from sklearn.externals import joblib
import joblib


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# data=pd.read_excel(“data24.12.19powermodelk.xlsx")
# data=pd.read_excel("data24.12.20powermodelly.xlsx")
data=pd.read_excel("data25.1.6powermodelu.xlsx") #读取设备的功耗数据

print(data.describe())

corr_matrix=data.corr()
print("corr_matrix: ",corr_matrix)

#构建训练集,测试集
train_set, test_set = split_train_test(data, 0.2)
train_cpuu= train_set.drop("power", axis=1)
train_power = train_set["power"].copy()
test_cpu=test_set.drop("power", axis=1)
test_power = test_set["power"].copy()

#训练和评估训练集
lin_reg = LinearRegression()
lin_reg.fit(train_cpuu, train_power)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(train_cpuu)
print(housing_predictions[:20])
print(train_power[:20])
lin_mse = mean_squared_error(train_power, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

housing_predictions = lin_reg.predict(test_cpu)
lin_mse = mean_squared_error(test_power, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#模型保存
savepath='powermodel_u.pkl'
joblib.dump(lin_reg, savepath)
#模型加载
loaded_model = joblib.load(savepath)
housing_predictions = loaded_model.predict(test_cpu)
lin_mse = mean_squared_error(test_power, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

x=range(0,len(housing_predictions))
plt.figure(figsize=(10, 4), dpi=80)
plt.plot(x,housing_predictions,marker='.', markersize=5,label='predicted power')
plt.plot(x,test_power,marker='.',markersize=5,label='real power')
plt.legend()
plt.show()
