import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from ai_KNN import DATA_FILE

FEAT_COLS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack","PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost","GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]


def plot_feature_importance(bst, feat_names):
    # 获取特征重要性
    importance = bst.feature_importance(importance_type='gain')

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

    # 创建LightGBM数据集并指定特征名
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEAT_COLS)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=FEAT_COLS)

    # 设置LightGBM参数（假设是多分类问题）
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',  # 修改为多分类目标
        'num_class': 3,  # 指定类别数量
        'metric': 'multi_logloss',  # 修改为多分类评估指标
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # 训练模型
    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # 预测
    y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred = y_pred_prob.argmax(axis=1)  # 对于多分类问题，选择概率最高的类别作为预测值

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 绘制特征重要性图
    plot_feature_importance(bst, FEAT_COLS)


if __name__ == '__main__':
    main()