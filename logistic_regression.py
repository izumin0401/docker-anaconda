from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# アイリスデータをロードする
iris = load_iris()

# 学習用データ、予測用データ、学習用教師データ、予測用教師データを「学習：8、予測：2」の割合で割り当てる
(train_X, test_X, train_Y, test_Y) = train_test_split(iris.data, iris.target, test_size=0.2)

# ロジスティック回帰用インスタンス生成
model = LogisticRegression()

# 学習モデル生成
model.fit(train_X, iris.target_names[train_Y])

# ロジスティック回帰による予測を行う
pred = model.predict(test_X)

# 精度を計算する
score = accuracy_score(iris.target_names[test_Y], pred)

# 結果を出力する
print('score:%s' % score)
print(classification_report(iris.target_names[test_Y], pred))
print(confusion_matrix(iris.target_names[test_Y], pred))
