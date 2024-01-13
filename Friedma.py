# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:51 2020

@author: pc
"""

import numpy as np
import matplotlib.pyplot as plt
# import scikit_posthocs as sp
#plt.rc('font',family='Times New Roman')
plt.rc('font',family='STSong')


ACC1 = np.array([[88.06,89.42,88.65,89.18,89.81],
					[95.20,95.68,95.92,96.66,95.35],
					[96.36,96.97,96.55,96.68,96.6],
					[96.68, 96.49, 96.51, 96.57, 96.88],
					[95.22, 93.21, 88.13, 92.49, 93.98],
					[96.84, 96.68, 97.05, 97.12, 96.64],
					[96.05, 96.36, 96.11, 96.88, 96.40],
					[96.01, 96.22, 96.20, 96.23, 96.05],
					[95.81, 96.11, 95.55, 95.40, 95.79],
					[95.07, 95.51, 95.51, 95.24, 95.33],
					[93.88, 94.11, 94.08, 94.11, 94.39],
					[94.22, 94.43, 94.44, 94.54, 94.17],
					[93.98, 93.61, 93.69, 93.76, 93.84],
					[97.97, 98.10, 97.90, 97.86, 97.75],])

ACC2 = np.array([[92.7,93.2,92.3,93.0,93.6],
					[91.5,91.3,90.3,90.1,90.2],
					[89.5,90.5,90.1,89.9,90.3],
					[91.0,92.3,90.8,92.2,91.1],
					[89.8,92.2,90.7,90.1,91.7],
					[91.5,92.5,91.1,92.2,93.1],
					[90.4,92.7,90.6,91.9,91.3],
					[91.8,92.1,90.9,91.9,91.3],
					[91.8,92.6,91.2,92.5,90.2],
					[90.2,91.5,90.1,91.5,89.7],
					[93.8,92.9,94.0,92.9,93.0],
					[94.2,91.9,93.9,94.1,94.0],
					[94.4,93.2,93.7,93.1,93.8],
					[93.8,93.2,94.1,93.8,94.0],
                    ])
F1=np.array([[86.70, 87.87, 87.04, 87.63 ,88.07],
					 [94.72, 95.51, 95.49, 96.38, 94.69],
					 [95.99, 96.87, 96.35, 96.37, 96.48],
					 [96.33, 96.33, 96.39, 96.19, 96.58],
					 [94.59, 92.56, 86.93, 91.90, 94.02],
					 [97.06, 96.57, 96.85, 96.96, 96.49],
					 [95.76, 96.15, 95.93, 96.63, 96.18],
					 [95.78, 95.77, 95.13, 94.61, 95.29],
					 [95.08, 95.77, 95.13, 94.61, 95.29],
					 [94.74, 95.08, 94.94, 94.83, 95.04],
					 [92.80, 93.32, 92.78, 93.16, 93.29],
					 [93.31, 93.46, 93.45, 93.64, 92.95],
					 [92.93, 92.23, 92.56, 92.71, 92.77],
					 [97.80, 97.90, 97.69 ,97.59, 97.45],
                     ])

F11 = np.array([[94.71, 95.11, 94.40, 95.45, 95.07],
					 [93.94, 93.67, 93.42, 92.19, 92.87],
					 [92.49, 93.29, 92.82, 92.82, 93.11],
					 [93.46, 94.45, 93.25, 94.44, 93.59],
					 [92.67, 94.28, 93.18, 93.23, 94.02],
					 [94.10, 94.57, 93.49, 94.46, 95.08],
					 [93.04, 94.69, 93.14, 93.93, 94.49],
					 [93.98, 94.37, 93.36, 94.18, 93.76],
					 [94.10, 94.74, 93.68, 94.99, 93.10],
					 [92.94, 93.86, 92.80, 93.96, 92.61],
					 [92.75, 91.71, 92.94, 91.29, 91.63],
					 [93.20, 90.64, 92.84, 93.11, 93.74],
					 [93.37, 92.17, 92.68, 91.93, 92.62],
					 [95.53, 95.15, 95.82, 95.53, 95.74],
					 ])


name = list();
name.append("CNN")
name.append("LSTM")
name.append("BiLSTM")
name.append("AttBLSTM")
name.append("RCNN")
name.append("CNN_LSTM")
name.append("StackedLSTM")
name.append("BIGRU_CNN")
name.append("BiLSTM_CNN")
name.append("AC_LSTM")
name.append("BERT")
name.append("BERT_wwm_ext")
name.append("ERNIE")
name.append("SMC-DCRN-ATT")

data = np.array([[1,2,3],[1,2.5,2.5],[1,2,3],[1,2,3]]) #书上的数据

"""
    构造降序排序矩阵
"""
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    return matrix

"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""
def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    # print(sumr)
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result

"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))
    
matrix = np.array([[91.44, 95.55, 95.94, 89.07, 96.47, 96.07, 95.29, 94.22, 95.28, 96.29, 94.37, 94.70, 94.02, 98.28], 
                 [93.20, 91.00, 91.30, 92.60, 91.10, 93.00, 93.20, 91.70, 92.70, 92.10, 93.90, 94.00, 93.80, 94.10]]) 
matrix_2 = np.array([[90.87,95.45,95.60,87.72,96.13,95.86,95.19,93.71,94.47,95.96,93.36,93.76,93.01, 98.07], 
                     [95.17,93.61,93.79,94.72,93.60,94.99,95.18,94.11,94.85,94.39,95.61,95.74,92.62, 95.82]])
matrix_3 = np.row_stack((matrix,matrix_2))

matrix = np.array([[72.27,80.07,79.80,79.52,78.59,81.79,88.42,97.48],
                   [54.81,70.46,78.24,70.10,77.28,91.23,91.45,94.13],
                 [77.55,85.82,85.00,86.51,87.23,88.52,88.96,92.29]])

#matrix = matrix_3
#matrix = np.array([linear_scores, ridge_scores, lasso_scores, elasticNet_scores])
matrix_r = len(matrix[0]) + 1 - rank_matrix(matrix)
Friedman = friedman(len(matrix), len(matrix[0]), matrix_r)
CD = nemenyi(len(matrix), len(matrix[0]), 2.780)
print(CD)
##画CD图
rank_x = list(map(lambda x: np.mean(x), matrix_r.T))
print(rank_x)
name_y = ["Single Visual Model", "Single Textual Model", "Late Fusion", "Early Fusion", "CCR", "DMAF", "AMGN", "MIFN"]
min_ = [x for x in rank_x - CD/2]
max_ = [x for x in rank_x + CD/2]

fig = plt.figure()
font1 = {'family': 'Times New Roman',
		 'weight': 'normal',
		 }
fig.subplots_adjust(left=0.25, bottom=0.3, right=0.9, top=0.9, hspace=0., wspace=0.)
#plt.title("Friedman")
plt.scatter(rank_x, range(len(name_y)))
plt.hlines(range(len(name_y)), min_, max_)
plt.xlabel('Average rank value', font1)
plt.ylabel('Method', font1)
plt.yticks(range(len(name_y)), name_y, fontproperties='Times New Roman')
plt.xticks(range(-2, 12, 2), fontproperties='Times New Roman')
plt.savefig('13.pdf',dpi=1080, bbox_inches='tight')
plt.show()



import numpy as np
a = np.random.randn(3)
#
#sp.posthoc_conover_friedman(matrix.tolist())
#
#a = sp.posthoc_nemenyi_friedman(matrix)
##a = np.abs(a)
#b = sp.posthoc_wilcoxon(matrix.T.tolist())
#c = sp.posthoc_nemenyi(matrix.T.tolist())
#d = sp.posthoc_tukey(ACC1.tolist())
#e = sp.posthoc_ttest(ACC1)
#f = sp.posthoc_ttest(ACC2)
#fig = plt.figure()
##fig.add_subplot(121)
#fig.subplots_adjust(left=0.01, bottom=0.3, right=0.99, top=0.9, hspace=0., wspace=0.0)
#ax = plt.imshow(e, cmap='Blues')
#plt.yticks(range(len(name)), name)
#plt.xticks(range(len(name)), name, rotation=90)

#for i in range(e.shape[0]):
#    for j in range(e.shape[1]):
#        if j == i:
#            continue
#        plt.text(i-0.4, j+0.1, "{:.2f}".format(e.iloc[i,j]), fontsize=5.5)


#bar = plt.colorbar(ax)
##bar.set_ticks([])
##plt.show()
#plt.savefig(r'C:\Users\pc\Desktop\1.jpg',dpi=1000)
#
#
#fig = plt.figure()
#fig.subplots_adjust(left=0.01, bottom=0.3, right=0.99, top=0.9, hspace=0., wspace=0.0)
##fig.add_subplot(122)
#plt.imshow(f, cmap='Blues')
#plt.yticks(range(len(name)), name)
#plt.xticks(range(len(name)), name, rotation=90)
#
##for i in range(f.shape[0]):
##    for j in range(f.shape[1]):
##        if j == i:
##            continue
##        plt.text(i-0.4, j+0.1, "{:.2f}".format(f.iloc[i,j]), fontsize=5.5)
#
#
#bar = plt.colorbar(ax)
##bar.set_ticklabels(bar.get_ticks())
##bar.set_ticks([])
##plt.show()
#plt.savefig(r'C:\Users\pc\Desktop\2.jpg',dpi=1000)
