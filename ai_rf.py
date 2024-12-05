import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from ai_KNN import DATA_FILE

FEAT_COLS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
             "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
             "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

def plot_feature_importance(model, feat_names):
    # 获取特征重要性
    importance = model.feature_importances_

    # 创建DataFrame以方便绘图
    feature_importance_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    # 获取数据和标签,X是数据,y表示结果
    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Label'].values

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林分类器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 预测
    y_pred = rf_model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 绘制特征重要性图
    plot_feature_importance(rf_model, FEAT_COLS)

if __name__ == '__main__':
    main()