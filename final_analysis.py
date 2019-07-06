#Sentiment Analysis on Movie Reviews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.stem import  WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,ComplementNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load the Train dataset
df=pd.read_csv("D:/DBA/project/movie_reviews/train.tsv", sep="\t")
pd.options.display.max_columns=50
df.head(10)
df.Sentiment.value_counts()
df.Phrase[0]
df.Phrase[1]
df.Phrase[2]
df[df.Sentiment==0]
df[df.Sentiment==1]
df[df.Sentiment==2]
df[df.Sentiment==3]
df[df.Sentiment==4]

#Null value checking
df.isnull().sum()


# Text preprocessing (cleaning the reviews, tokenize and lemmatize the data)
def pre_process(text):
    text=text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    tokens=nltk.word_tokenize(text)  # tokenize
    wnl = WordNetLemmatizer()
    L=[wnl.lemmatize(w) for w in tokens]
    text=" ".join(L)
    return text

# Split the data into train and test
X= df['Phrase'].apply(pre_process)
y=df['Sentiment']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

# build the model
model1=make_pipeline(CountVectorizer(binary=True,stop_words="english"),BernoulliNB())
model2=make_pipeline(CountVectorizer(binary=False,stop_words="english"),MultinomialNB())
model3=make_pipeline(TfidfVectorizer(stop_words="english"),MultinomialNB())


# Train the model
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)


# predict y of test set
y_pred_1 = model1.predict(X_test)
y_pred_2 = model2.predict(X_test)
y_pred_3 = model3.predict(X_test)


# Confusion matrix
mat= confusion_matrix(y_test,y_pred_1)
sns.heatmap(mat,annot=True,cbar=False,fmt='d')
plt.ylabel('True class')
plt.xlabel('Predicted class')

mat= confusion_matrix(y_test,y_pred_2)
sns.heatmap(mat,annot=True,cbar=False,fmt='d')
plt.ylabel('True class')
plt.xlabel('Predicted class')

mat= confusion_matrix(y_test,y_pred_3)
sns.heatmap(mat,annot=True,cbar=False,fmt='d')
plt.ylabel('True class')
plt.xlabel('Predicted class')


# overall accuracy
model1.score(X_test,y_test).round(3)
model2.score(X_test,y_test).round(3)
model3.score(X_test,y_test).round(3)


# Train the second model on the entire training data set 
# and use this model to predict unknown test data
model4=make_pipeline(CountVectorizer(binary=False,stop_words="english"),MultinomialNB())
model4.fit(X,y)

# load the test data
df1=pd.read_csv("D:/DBA/project/movie_reviews/test.tsv", sep="\t")
pd.options.display.max_columns=50
df1.head(10)

#Cleansing the test dataset
Xtest=df1.Phrase.apply(pre_process)

#Now make Prediction on Test dataset
predicted_sentiment=model4.predict(Xtest)
predicted_sentiment

#Now We create a DataFrame where we store the result as Sentiment analysis output
df2 = pd.DataFrame(predicted_sentiment,columns=['Predicted_Sentiment'])
df2.head()

#Take the "PhraseId" column from test dataset and put our prediction result according to it. 
final_result = df2.join(df1['PhraseId']).iloc[:,::-1]
final_result.head()

#Write the output to the csv file.
final_result.to_csv("D:/DBA/project/movie_reviews/final_submission.csv", index=False)
