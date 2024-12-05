# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置图形样式
sns.set(style="whitegrid")


def load_data(file_path):
    """加载CSV数据"""
    return pd.read_csv(file_path)


def basic_info(df):
    """显示数据的基本信息"""
    print("Data Overview:")
    print(df.info())
    print("\nDescription of numerical features:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())


def visualize_missing_values(df):
    """可视化缺失值情况"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()


def visualize_distribution(df, target_col):
    """绘制特征分布图"""
    features = [col for col in df.columns if col not in ['id', target_col]]

    # 直方图
    df[features].hist(bins=30, figsize=(20, 16), layout=(6, 4))
    plt.suptitle('Feature Distribution Histograms')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # 箱线图
    plt.figure(figsize=(20, 16))
    for i, col in enumerate(features, 1):
        plt.subplot(6, 4, i)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


def correlation_analysis(df, target_col):
    """计算并可视化相关性矩阵"""
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', mask=mask, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def feature_target_relationship(df, target_col):
    """分析每个特征与目标变量的关系"""
    features = [col for col in df.columns if col not in ['id', target_col]]

    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.title(f'{feature} vs {target_col}')
        plt.show()


def main():
    # 文件路径
    file_path = 'data.csv'

    # 加载数据
    df = load_data(file_path)

    # 数据概览
    basic_info(df)

    # 可视化缺失值
    visualize_missing_values(df)

    # 分布分析
    visualize_distribution(df, 'target')

    # 相关性分析
    correlation_analysis(df, 'target')

    # 特征与目标变量关系分析
    feature_target_relationship(df, 'target')


if __name__ == "__main__":
    main()