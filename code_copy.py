#(0)Problem
#input=Admission_Predict.csv
 #feature=["GRE_Score","TOEFL_Score","University_Rating","SOP","LOR","CGPA","Research"]
 #label=["Chance_of_Admit"]
#ouput은 해당 input데이터를 통해 훈련한 모델의 성능
#label은 1개
#데이터는 모든 컬럼이 다 숫자형으로 이루어진 dataframe이다.

#(1)Feature
#I choose feature like that
#feature=["GRE_Score","TOEFL_Score","University_Rating","SOP","LOR","CGPA","Research"]
#데이터 전처리: 공백이 있는 열의 이름을 모두 바꾸어줌
#데이터 전처리: 모델 성능에 방해될 수 있는 요소인 컬럼을 제외함

#(2)Model
#선택한 모델은 Ridge_Regression
#왜냐하면 주어진 Regression모델 중 가장 모델의 성능이 좋게 나타남

#(3)Measure
 #1. 우선 예측할 값인 label과 그 예측을 하기 위해 필요한 data인 feature를정한다.
 #2. 정해진 값을 각각 traindata/testdata에 나누어 넣고 모델을 생성한다.
 #3. 이때 훈련시키는 방법은 교차검증으로 데이터 전체를 분할하여 test/train data로 나누고반복해서  모델을 훈련시킨다. 
 #4. 훈련된 모델이 예측한 값과 주어진 라벨의 정답값을 비교하여 모델의 성능을 평가한다.

#(4)Model parameter engineering
#모델의 parameter인 alpha값을 0.1,0.001,0.0001,0.00001등을 넣어 조정해봤을때 0.001의 성능이 근소하지만 가장 좋게 나왔음으로 alpha=0.001을 선택.
#그치만 [0.1,0.00001]까지 10단위로 parameter를 조정해보았을때  모델 성능의 차이는 매우 작음
#데이터의 불균형이나 희소성은 두드러지지 않았음.

#02번
#regression model
#따라서 모델성능평가에 MAE(mean absolute error) 사용

import numpy as np
import pandas as pd
import pickle as pi

#데이터 위치설정 및 불러오기
datapath="../homework8/Admission_Predict.csv"
original_data=pd.read_csv(datapath, sep=",")
print("---------------------------------------------------------------------------------------------")
print("\n")
print("데이터정보")
a=original_data.info()
b=original_data.describe()
c=original_data.corr()
print("\n")
print("데이터요약")
print(b)
print("\n")
print("각 featre들의 상관관계")
print(c)

final_data=original_data.copy()
#'serial No'같이 정답을 맞추는데 필요한 feature가 아니고 정답에 영향을 미치지 않는 feature는 제외
final_data=final_data.drop(columns=['Serial No.'])

#결측지확인하기
check_for_nan=final_data.isnull().values.any()
print("결측치데이터= {}".format(check_for_nan))
print("\n")
print("---------------------------------------------------------------------------------------------")
print("\n")
print("모델성능")

#사용할 데이터 열이름 바꾸기
final_data.columns=["GRE_Score","TOEFL_Score","University_Rating","SOP","LOR","CGPA","Research","Chance_of_Admit"]

#사용할패키지 로딩
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

#사용할feature
features=["GRE_Score","TOEFL_Score","University_Rating","SOP","LOR","CGPA","Research"]
#예측할 label
labels=["Chance_of_Admit"]

#교차검증을 통해서 모델훈련,10번의 교차검증
kf=KFold(n_splits=10, shuffle=True)

mae=[]
fold_idx=1
a=0
for train_idx, test_idx in kf.split(final_data):
    train_d, test_d =final_data.iloc[train_idx],final_data.iloc[test_idx]

    train_y=train_d[labels]
    train_x=train_d[features]

    test_y=test_d[labels]
    test_x=test_d[features]
    #모델생성
    model=Ridge(alpha=0.001)
    model.fit(train_x,train_y)
    #예측하기
    pred_y=model.predict(test_x)
    #모델 성능 평가
    mean_mae=mean_absolute_error(test_y,pred_y)
    mae.append(mean_mae)

    print('Fold {} : MAE = {}'.format(fold_idx,mae[a]))
    fold_idx += 1
    a += 1

print('Total (Average) MAE = {}'.format(np.average(mae)))


