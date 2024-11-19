import pandas as pd
from sklearn.model_selection import train_test_split
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

def investigate_knn(diabetes_data,k_val):
    # 获取数据和标签,x是数据,y表示结果
    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)
    # 声明KNN模型, 设置模型k值
    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    # 训练模型调教
    knn_model.fit(X_train, y_train)
    # 评价模型
    print("当前k值为：{}".format(k_val))
    accuracy = knn_model.score(X_test, y_test)
    print('预测准确率:{:.2f}%'.format(accuracy * 100))
    y_pred = knn_model.predict(X)
    print('评测分数', f1_score(y, y_pred, average='macro'))

def main():
    # 读取数据集0
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target'].map(SPECIAL_LABEL_DICT)

    k_vals=[2,3,4,5]
    for k_val in k_vals:
        investigate_knn(diabetes_data,k_val)

if __name__ == '__main__':
    main()