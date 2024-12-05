import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from ai_KNN import DATA_FILE
#所关联的特征
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values
    #Y的种类转换为整数
    Y=pd.Categorical(y).codes
    #数据标准化
    X=StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=10)

    X_t = StandardScaler().fit(X_train)
    X_train1=X_t.transform(X_train)
    X_test1=X_t.transform(X_test)

    #PCA降维
    k=0.98
    pca=PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train)  # 在训练集上拟合模型并进行降维
    X_test_pca = pca.transform(X_test)  # 将测试集降维
    print("主成分的数量：", pca.n_components_)

    log_reg = LogisticRegression(solver='newton-cg',multi_class='multinomial')

    log_reg.fit(X_train_pca, y_train)

    y_test_pca = log_reg.predict(X_test_pca)

    print("测试集分类准确率：\n", metrics.accuracy_score(y_test, y_test_pca))

    print(classification_report(y_test, y_test_pca))
if __name__ == '__main__':
    main()




