import os
import nni
import torch

def filter_samples(v, mu_k, Sigma_hat, epsilon):
    """
    过滤出满足多元正态分布概率密度函数值小于epsilon的样本点。

    参数:
    - v (torch.Tensor): 形状为 [num_samples, num_features] 的样本点集合
    - mu_k (torch.Tensor): 形状为 [num_features] 的均值
    - Sigma_hat (torch.Tensor): 形状为 [num_features, num_features] 的协方差矩阵
    - epsilon (float): 阈值

    返回:
    - filtered_samples (torch.Tensor): 被过滤出来的样本点
    """
    m = v.size(1)
    Sigma_hat_inv = torch.inverse(Sigma_hat)
    Sigma_hat_det = torch.det(Sigma_hat)
    
    # 计算样本与均值的差
    diff = v - mu_k
    
    # 计算多元正态分布的概率密度函数值
    normalization_constant = (2 * torch.pi) ** (m / 2) * torch.sqrt(Sigma_hat_det)
    exponent = -0.5 * torch.sum(diff.matmul(Sigma_hat_inv) * diff, dim=1)
    pdf_values = torch.exp(exponent) / normalization_constant
    
    # 过滤出满足条件的样本点
    mask = pdf_values < epsilon
    filtered_samples = v[mask]
    
    return filtered_samples

# 模拟数据
# v = torch.randn(200, 2048)
# mu_k = torch.randn(2048)
# Sigma_hat = torch.randn(2048, 2048)
# Sigma_hat = torch.mm(Sigma_hat, Sigma_hat.t())  # 使协方差矩阵为正定的
# epsilon = 0.01

# filtered = filter_samples(v, mu_k, Sigma_hat, epsilon)
# print(filtered.shape)  # 输出被过滤出来的样本点的形状


def compute_mean_cov_matrix(features):
    """
    计算特征的均值和协方差矩阵。

    参数:
    - features (torch.Tensor): 形状为 [num_samples, num_features, 1, 1] 的特征张量

    返回:
    - mean (torch.Tensor): 形状为 [num_features] 的均值张量
    - covariance_matrix (torch.Tensor): 形状为 [num_features, num_features] 的协方差矩阵张量
    """
    # 去除尺寸为1的维度
    features = features.squeeze(-1).squeeze(-1)
    
    # 计算均值
    mean = features.mean(dim=0)
    
    # 计算协方差矩阵
    diff = features - mean.unsqueeze(0)
    covariance_matrix = diff.t().matmul(diff) / features.size(0)
    
    return mean, covariance_matrix

# 生成模拟数据并测试函数
features = torch.rand((200, 2048, 1, 1))
mean, covariance_matrix = compute_mean_cov_matrix(features)

# print(mean.size())  # 应该输出 torch.Size([2048])
# print(covariance_matrix.size())  # 应该输出 torch.Size([2048, 2048])

# print(mean)
# print(covariance_matrix)

# compute_mean_cov_matrix(torch.randn((200, 2048, 1, 1)))

# import numpy as np
# import umap
# import matplotlib.pyplot as plt

# # 模拟一些数据
# data = np.random.randn(300, 5)  # 300个样本，每个样本5个特征
# labels = np.random.randint(2, size=(300, 3))  # 300个样本，3个标签的01矩阵

# mean_dict = {}
# cov_dict = {}

# # 对每个标签进行迭代
# for i in range(labels.shape[1]):
#     # 找出所有标记为当前标签的样本
#     relevant_data = data[labels[:, i] == 1]
    
#     # 计算均值和协方差
#     mean_dict[i] = np.mean(relevant_data, axis=0)
#     cov_dict[i] = np.cov(relevant_data, rowvar=False)

# # 打印结果
# # for i in range(labels.shape[1]):
# #     print(f"Label {i} - Mean: {mean_dict[i]}, Covariance: {cov_dict[i]}")

# # 将均值和协方差的对角线连接起来形成一个扩展的特征向量
# extended_features = []

# for i in range(labels.shape[1]):
#     extended_features.append(np.hstack([mean_dict[i], np.diag(cov_dict[i])]))

# extended_features = np.array(extended_features)
# print(extended_features)

# # 使用UMAP进行降维
# extended_features_dense = extended_features.toarray()
# reduced_data = umap.UMAP().fit_transform(extended_features_dense)
# # reduced_data = umap.UMAP().fit_transform(extended_features)

# # 可视化
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
# for i, point in enumerate(reduced_data):
#     plt.annotate(f"Label {i}", (point[0], point[1]))

# plt.xlabel("UMAP 1st Dimension")
# plt.ylabel("UMAP 2nd Dimension")
# plt.show()

import umap
import numpy as np
import matplotlib.pyplot as plt

# 假设 means 和 std_devs 分别是均值和标准差的列表，每个类别一个。
num_classes = 10
feature_dim = 3

# 生成均值：我们使用正态分布，其均值为0，标准差为5来生成均值向量
means = np.random.normal(0, 5, size=(num_classes, feature_dim))

# 生成标准差：我们使用一个均匀分布，范围从0.5到2.5，确保标准差为正值
std_devs = np.random.uniform(0.5, 2.5, size=(num_classes, feature_dim))

print("Generated means:\n", means)
print("\nGenerated standard deviations:\n", std_devs)

# 串联均值和标准差得到新的特征向量
combined_features = np.concatenate([means, std_devs], axis=1)

# 使用 UMAP 进行降维
reduced_data = umap.UMAP().fit_transform(combined_features)

# 可视化
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel("UMAP 1st Dimension")
plt.ylabel("UMAP 2nd Dimension")
plt.show()
