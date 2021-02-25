'''
2020.12.07
'''

from scipy.spatial import distance  #载入欧几里德计算机里模块或是套件

class ScrappyKNN():               #定义KNN对象(包含多个函式)
    def fit(self, X_train, y_train):  # 面向对象设计方法，self代表ScrappyKNN
        self.X_train = X_train        # 將X_train：150*4 鸢尾花训练数据，转存为ScrappyKNN类别的串列数组
        self.y_train = y_train        # 將y_train：150*1 鸢尾花类别代码训练数据，转存为ScrappyKNN类别的串列数组

    def predict(self, X_test):   # 预测X_test分类结果函式
        predictions = []         # 存放预测结果之串列数组
        for row in X_test:
            label = self.closest(row)  # 将测试资料代入closest函式，并将测试类别结果储存label
            predictions.append(label)  # 将测试结果新增一笔至prediction串列数组

        return predictions      

    def closest(self, row):      # 计算训练数据集及测试数据集的最近距离函式,
        best_dist = self.euc(row, self.X_train[0]) # 代入参数row数组测试资料，与训练及资料train[0]从第一笔数据，
                                                   # 利用尤拉公式(euc)计算最近距离。
                                                   # 把best_dist 视为第一笔记录的计算最近距离
        best_index = 0                             # 把 best_index 视为第一笔计算最近距离的索引值
        for i in range(len(X_train)):              # 将所有训练资料一笔一笔带进去计算最近距离
            dist = self.euc(row, self.X_train[i])  # 利用尤拉公式(euc)计算最近距离。
            if dist < best_dist:                   # 如果新的最近距离小于前一个最近距离，则将新的距离视为最近距离
                best_dist = dist                   # 则将新的距离视为最近距离(best_dist)
                best_index = i                     # 记录最新的最近距离索引值
        return self.y_train[best_index]

    def euc(self, a, b):     # 欧几里得公式计算最短距离
        return distance.euclidean(a, b)


from sklearn.datasets import load_iris

iris = load_iris()   # 载入鸢尾花数据库

X = iris.data        # 载入150*4 鸢尾花4种属性数据库

y = iris.target      # 载入150*1 鸢尾花3种类别代码数据库[0,1,2]

from sklearn.model_selection import train_test_split  # 载入交叉验证的函数

# train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
# test_size = 0.4 :训练资料占60%测试资料占40%
# X_train：150*4 鸢尾花训练数据; y_train: 150*1 鸢尾花3种类别代码训练数据
# X_test：150*4 鸢尾花测试数据; y_test: 150*1 鸢尾花3种类别测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9)  

my_classifier = ScrappyKNN()  # 继承复用KNN模块

# fit 函数接受两个参数，分别是训练集的特征和类别标签
my_classifier.fit(X_train, y_train) 

predictions = my_classifier.predict(X_test)   #预测X_test的分类结果

from sklearn.metrics import accuracy_score    #导入计算测试后的分类精确度函式

print("The classified accuracy is ",accuracy_score( y_test, predictions))    #计算测试后的分类精确度


