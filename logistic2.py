import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from ai_KNN import DATA_FILE

# 所关联的特征
FEAT_COLS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
             "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
             "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

SPECIAL_LABEL_DICT = {
    0: "non-diabetic",  # 非糖尿病
    1: "pre-diabetes",  # 糖尿病前期
    2: "diabetes"  # 糖尿病
}


def plot_fitting_line(classifier, X, y, feat):
    # 计算每个类别的概率响应
    x_min, x_max = X.min(), X.max()
    xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    probabilities = classifier.predict_proba(xx)

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(SPECIAL_LABEL_DICT.values()):
        yy = probabilities[:, i]
        plt.plot(xx, yy, label=f'P({class_label})', lw=2)

    # 绘制散点图
    scatter = plt.scatter(X, [SPECIAL_LABEL_DICT[yi] for yi in y], alpha=0.5, c=y, cmap='viridis')
    plt.title(feat)
    plt.xlabel(feat)
    plt.ylabel('Class Probability')
    plt.legend()
    plt.show()


def main():
    diabetes_data = pd.read_csv(DATA_FILE, index_col='id')
    diabetes_data['Label'] = diabetes_data['target']

    for feat in FEAT_COLS:
        X = diabetes_data[[feat]].values  # 注意这里保持二维数组
        y = diabetes_data['Label'].values

        # 将标签转换回整数形式以便于建模
        y = pd.Categorical(y).codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

        # 使用逻辑回归进行三分类
        classifier = LogisticRegression(max_iter=200, multi_class='ovr')  # 使用 One-vs-Rest 策略
        classifier.fit(X_train, y_train)

        # 预测测试集
        y_pred = classifier.predict(X_test)

        # 计算并打印评测分数
        print(f'{feat} 的 F1 分数:', f1_score(y_test, y_pred, average='macro'))

        # 绘制图表
        plot_fitting_line(classifier, X_test, y_test, feat)


if __name__ == '__main__':
    main()