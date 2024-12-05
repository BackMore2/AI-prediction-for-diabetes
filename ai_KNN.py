import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

DATA_FILE="data.csv"

SPECIAL_LABEL_DICT={
    0:"non-diabetic 非糖尿病", #非糖尿病
    1:"pre-diabetes 糖尿病前期",  #糖尿病前期
    2:"diabetes 糖尿病" #糖尿病
}

#所关联的特征
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

def main():
    # 读取数据集0
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']
    # 获取数据和标签,x是数据,y表示结果
    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)
    # 声明KNN模型, 设置模型k值
    param_grid = {
        'n_neighbors': list(range(1, 7)),  # 尝试不同的k值
        'weights': ['uniform', 'distance'],  # 尝试两种权重方案
        'p': [1, 2]  # 尝试曼哈顿距离(p=1)和欧几里得距离(p=2)
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1)

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    # 输出最佳参数组合及对应的F1得分
    print("Best parameters found: ", grid_search.best_params_)
    print("Best F1 score: ", grid_search.best_score_)

    # 使用最佳参数重新训练模型并在测试集上评估
    best_knn = grid_search.best_estimator_
    predictions = best_knn.predict(X_test)
    final_f1_score = f1_score(y_test, predictions, average='macro')
    print("Final F1 score on the test set: ", final_f1_score)

if __name__ == '__main__':
    main()