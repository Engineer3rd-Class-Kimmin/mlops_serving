import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 모델 학습
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'model.joblib')
print("Model saved!")