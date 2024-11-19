import pandas as pd
from holoviews.plotting.bokeh.styles import alpha
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

from ai_KNN import DATA_FILE
#貌似无需map映射
SPECIAL_LABEL_DICT={
    0:"non-diabetic", #非糖尿病
    1:"pre-diabetes",  #糖尿病前期
    2:"diabetes" #糖尿病
}
#所关联的特征
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

def plot_fitting_line(linear_reg_model,X,y,feat):
    w=linear_reg_model.coef_
    b=linear_reg_model.intercept_

    plt.figure()
    plt.scatter(X,y,alpha=0.5)

    plt.plot(X,w*X+b,c='red')
    plt.title(feat)
    plt.show()


def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values
    for feat in FEAT_COLS:
        X=diabetes_data[feat].values.reshape(-1,1)
        y=diabetes_data['Label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)
        Linear_reg_model=LinearRegression()
        Linear_reg_model.fit(X_train, y_train)
        #y_pred = Linear_reg_model.predict(X)
        #print('评测分数', f1_score(y,y_pred, average='macro'))
        plot_fitting_line(Linear_reg_model,X_test,y_test,feat)


if __name__ == '__main__':
    main()