import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ai_KNN import DATA_FILE
FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits",
           "Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex",
           "Age","Education","Income"]

def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    # 获取数据和标签,x是数据,y表示结果
    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 设置LightGBM参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # 训练模型
    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[test_data],callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # 预测
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))

if __name__ == '__main__':
    main()