import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# โหลดข้อมูล
file_path = "cardio_train.csv"  # ตรวจสอบว่ามีไฟล์นี้อยู่
df = pd.read_csv(file_path, sep=";")

# ทำความสะอาดข้อมูล
df.drop(columns=["id"], inplace=True)
df["age"] = (df["age"] / 365).astype(int)
df_cleaned = df[
    (df["height"] >= 100) & (df["height"] <= 250) &
    (df["weight"] >= 30) & (df["weight"] <= 200) &
    (df["ap_hi"] >= 50) & (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 30) & (df["ap_lo"] <= 200)
]

# แยก Features และ Target
X = df_cleaned.drop(columns=["cardio"])
y = df_cleaned["cardio"]

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# สร้างโมเดล Voting Classifier
models = [
    ("LoR", LogisticRegression(max_iter=1000, random_state=42)),
    ("kNN", KNeighborsClassifier(n_neighbors=5)),
    ("DT", DecisionTreeClassifier(random_state=42)),
    ("GB", GradientBoostingClassifier(random_state=42)),
    ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Adaboost", AdaBoostClassifier(random_state=42))
]

voting_clf = VotingClassifier(estimators=models, voting="hard")
voting_clf.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(voting_clf, "voting_classifier.pkl")
print("บันทึกโมเดลสำเร็จ! 🎉")
