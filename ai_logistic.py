import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ai_KNN import DATA_FILE
#所关联的特征
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

    log_reg = LogisticRegression(solver='newton-cg',multi_class='multinomial')

    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print('评测分数', f1_score(y_test ,y_pred, average='macro'))

if __name__ == '__main__':
    main()