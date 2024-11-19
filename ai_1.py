import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

DATA_FILE="data.csv"

SPECIAL_LABEL_DICT={
    0:"non-diabetic 非糖尿病", #非糖尿病
    1:"pre-diabetes 糖尿病前期",  #糖尿病前期
    2:"diabetes 糖尿病" #糖尿病
}

#所关联的特征
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

def main():
    #读取数据集
    diabetes_data=pd.read_csv(DATA_FILE,index_col='id')
    diabetes_data['Label']=diabetes_data['target']

    #获取数据和标签,x是数据,y表示结果
    X=diabetes_data[FEAT_COLS].values
    y=diabetes_data['Label'].values
    #划分数据集
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=10)

    model_dict={
        'KNN':(KNeighborsClassifier(),
        {
            'n_neighbors':[3,5,7],
            'p':[1,2]
        }),
        'Logistic Regression':(LogisticRegression(),
                               {'C':[1e-2,1,1e2]}),
        'SVM':(SVC(),
               {'C':[1e-2,1,1e2]})
    }

    #训练模型
    for model_name,(model,model_params) in model_dict.items():
        clf=GridSearchCV(estimator=model,param_grid=model_params,cv=5)
        clf.fit(X_train,y_train)
        best_model=clf.best_estimator_

        #验证
        acc=best_model.score(X_test,y_test)
        print('{}模型的预测准确率：{:.2f}%'.format(model_name,acc*100))
        print('{}模型的最优参数：{}'.format(model_name,clf.best_params_))


if __name__ == '__main__':
    main()