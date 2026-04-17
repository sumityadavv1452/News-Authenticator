import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

fake = pd.read_csv(r"C:\Users\SUMIT YADAV\Desktop\python\Fakedata.csv.csv")
real = pd.read_csv(r"C:\Users\SUMIT YADAV\Desktop\python\Truedata.csv.csv")

fake["label"] = 0   # fake = 0
real["label"] = 1   # real = 1

data = pd.concat([fake, real])

# input and output
X = data["text"]
y = data["label"]

# convert text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# train model
model = LogisticRegression()
model.fit(X, y)

# test your own news
news = input("Enter news: ")
news_vec = vectorizer.transform([news])

result = model.predict(news_vec)

if result[0] == 1:
    print("Real News ✅")
else:
    print("Fake News ❌")