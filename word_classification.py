#!/usr/bin/env python3

from collections import Counter
import urllib.request
from lxml import etree
from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import ShuffleSplit

alphabet="abcdefghijklmnopqrstuvwxyzäö-"
alphabet_set = set(alphabet)
Alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ä','ö']
# Returns a list of Finnish words
def load_finnish():
    finnish_url="https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename="src/kotus-sanalista_v1.xml"
    load_from_net=False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines=[]
            for line in data:
                lines.append(line.decode('utf-8'))
        doc="".join(lines)
    else:
        with open(filename, "rb") as data:
            doc=data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))

#Returns a list of English words
def load_english():
    with open("src/words", encoding="utf-8") as data:
        lines=map(lambda s: s.rstrip(), data.readlines())
    return lines

def get_features(a):
    """Y = np.zeros(shape=(1,29))
    for word in a:
        X=np.array(word)
        for i in range(0,28): 
            x=word.count(Alphabet[i])
            X=np.hstack((X,x))
        Y=np.vstack((Y,X))
    return Y"""
    #gets the feature matrix 
    array = []
    for i in a:
        l = []
        for j in alphabet:
            l.append(str(i).count(j))
        array.append(l)
 
    return np.array(array)
#filters out not-valid words
def contains_valid_chars(s):
    if all(elem in alphabet for elem in s):
        return True
    else:
        return False
    

def get_features_and_labels():
    """a=load_finnish()
    x=[]
    y=[]
    for word in a:
        word=word.lower()
        if contains_valid_chars(word):
            x.append(word)
    b=load_english()
    for eword in b:
        if eword.islower():
            if contains_valid_chars(eword):
                y.append(eword)
    



    c=get_features(x)
    d=get_features(y)
    y_en = np.ones(len(y))
    y_fin = np.zeros(len(x))
    Y = np.hstack((y_en,y_fin))
    return np.vstack((c,d)), Y"""
    a = [i.lower() for i in load_finnish()]
    fwrds = []
    for item in a:
        if contains_valid_chars(item):
            fwrds.append(item)
        else:
            continue 
 
    b = load_english()
    c = []
    for item in b:
        if item[0].islower():
            c.append(item)
        else:
            continue

    d = [i.lower() for i in c]
    ewrds = []
    for item in d:
        if contains_valid_chars(item):
            ewrds.append(item)
        else:
            continue            
    
    X_e = get_features(np.array(ewrds))
    X_f = get_features(np.array(fwrds))
    X = np.vstack((X_e,X_f))
    
    y_e = np.ones(len(ewrds))
    y_f = np.zeros(len(fwrds))
    y = np.hstack((y_e,y_f))   
    
    return X, y


def word_classification():
    X,y=get_features_and_labels()
    model=MultinomialNB()

    
    #best_svr = SVR(kernel='rbf')
    cv = KFold(n_splits=5,random_state=0,shuffle=True)

    #a,b,c,d,e=cross_val_score(model, X, y, cv=5)


 
    return cross_val_score(model, X, y, cv=cv)

def main():
    #a=load_english()
    print("Accuracy scores are:", word_classification())
    #int(get_features_and_labels())
if __name__ == "__main__":
    main()
