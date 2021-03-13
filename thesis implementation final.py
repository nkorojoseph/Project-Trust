#!/usr/bin/env python
# coding: utf-8

# In[1]:


#test data 
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import csv
import os
import ntpath
import networkx as nx
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections
from spotlight import annotate
from functools import partial
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity

F= nx.Graph()
fork_headers = ['user_id','projectid']
forks = pd.read_csv('/home/nkorojoseph/Documents/trialdata/fork.csv',header=None,skiprows=1, names=fork_headers)
fusers = forks.user_id.tolist()
fproj = forks.projectid.tolist()
#print(fusers)
#print(forks.head())
F.add_edges_from(forks.values)
#print(F.edges())

W= nx.Graph()
watch_headers = ['user_id','projectid']
watchers = pd.read_csv('/home/nkorojoseph/Documents/trialdata/watchers.csv',header=None,skiprows=1, names=watch_headers)
wusers = watchers.user_id.tolist()
wproj = watchers.projectid.tolist()
#print(wusers)
#print(watchers.head())
W.add_edges_from(watchers.values)
#print(W.edges())

P= nx.Graph()
pullrequest_headers = ['user_id','projectid']
pullrequest = pd.read_csv('/home/nkorojoseph/Documents/trialdata/pullrequest.csv',header=None,skiprows=1, names=pullrequest_headers)
pusers = pullrequest.user_id.tolist()
pproj = pullrequest.projectid.tolist()
#print(fusers)
#print(forks.head())
P.add_edges_from(pullrequest.values)

C= nx.Graph()
commit_headers = ['user_id','projectid']
commits = pd.read_csv('/home/nkorojoseph/Documents/trialdata/commits.csv',header=None,skiprows=1, names=commit_headers)
cusers = commits.user_id.tolist()
cproj = commits.projectid.tolist()
C.add_edges_from(commits.values)

PM= nx.Graph()
pm_headers = ['user_id','projectid']
pm = pd.read_csv('/home/nkorojoseph/Documents/trialdata/projectmembers.csv',header=None,skiprows=1, names=commit_headers)
pmusers = pm.user_id.tolist()
pmproj = pm.projectid.tolist()
PM.add_edges_from(pm.values)
print(PM.edges())
targetp = ntpath.basename('/home/nkorojoseph/Desktop/githubreadmefiles/p4')
totalusers = set(fusers + wusers + cusers + pusers+pmusers)

G = nx.DiGraph()
for i in totalusers:
    if (F.has_edge(i,targetp) and W.has_edge(i,targetp) and P.has_edge(i,targetp) and C.has_edge(i,targetp)) or PM.has_edge(i,targetp) :
        print('user %s found in (%s, %s)'% (i,i,targetp))
        G.add_edge(i,targetp)

occur_users = [u[0] for u in G.edges()]
occur_projects = [u[1] for u in G.edges()]

testfile = targetp
testusers = occur_users

print(occur_users)


# In[ ]:


import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import csv
import os
import ntpath
import networkx as nx
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections
from spotlight import annotate
from functools import partial
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
import numpy

#A function for Slicing a dictionary
'''n is the slice rate 
iterable -> the dictionary to be sliced. 
convert the dictionary to a list and use islice function to slice it
'''
def dicslice(n, iterable):
    return list(islice(iterable, n))

#nltk.download('punkt') # if necessary...
''' a stemmer which would be used to reduce 
each word to its root equivalence is built.
this will help reduce the noise in the text document.
this is also built alongside punctuation removal.
'''
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

'''Function that creates tokens to use.'''
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=STOPWORDS)

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


path = "/home/nkorojoseph/Desktop/githubreadmefiles/*"

oldProjectReadme = []
newProjectReadme = []
with open(r"/home/nkorojoseph/Desktop/githubtestreadme/"+testfile,'r') as newPorject:
    dataNew = newPorject.read()
    newProjectReadme.append(dataNew.replace("\n",""))

#print(len(newProjectReadme))
'''for both the new and old readme files, append the name of the files to a list containing the readme texts for each document'''
readmewithname = []
for fname in glob.glob(path):
    with open(fname, 'r') as infile:
        data = infile.read()
        oldProjectReadme.append(data.replace("\n",""))
        readmewithname.append((fname,data.replace("\n","")))
        
#print(readmewithname)
first_elts = [x[0] for x in readmewithname]
second_elts = [x[1] for x in readmewithname]
similarityvalues = []
for i in range(len(second_elts)):
    sim = cosine_sim(second_elts[i],newProjectReadme[0] )
    #sim = cosine_similarity(vectorizer.fit_transform(second_elts[i]),vectorizer.fit_transform(newProjectReadme[0])) 
    similarityvalues.append(sim)
    print("Similarity between old and new Readme %d = %f"%(i,sim))
maxi = 0    
for i in range(len(similarityvalues)):
    if similarityvalues[i] > maxi:
        maxi = similarityvalues[i]
        maxIndex = i

print('Maximum similarity value is %f index %d'%(maxi,maxIndex))
print('Similar project is ',ntpath.basename(first_elts[maxIndex]))

targetp = ntpath.basename(first_elts[maxIndex])

F= nx.Graph()
fork_headers = ['user_id','projectid']
forks = pd.read_csv('/home/nkorojoseph/Documents/trialdata/fork.csv',header=None,skiprows=1, names=fork_headers)
fusers = forks.user_id.tolist()
fproj = forks.projectid.tolist()
#print(fusers)
#print(forks.head())
F.add_edges_from(forks.values)
#print(F.edges())

W= nx.Graph()
watch_headers = ['user_id','projectid']
watchers = pd.read_csv('/home/nkorojoseph/Documents/trialdata/watchers.csv',header=None,skiprows=1, names=watch_headers)
wusers = watchers.user_id.tolist()
wproj = watchers.projectid.tolist()
#print(wusers)
#print(watchers.head())
W.add_edges_from(watchers.values)
#print(W.edges())

P= nx.Graph()
pullrequest_headers = ['user_id','projectid']
pullrequest = pd.read_csv('/home/nkorojoseph/Documents/trialdata/pullrequest.csv',header=None,skiprows=1, names=pullrequest_headers)
pusers = pullrequest.user_id.tolist()
pproj = pullrequest.projectid.tolist()
#print(fusers)
#print(forks.head())
P.add_edges_from(pullrequest.values)

C= nx.Graph()
commit_headers = ['user_id','projectid']
commits = pd.read_csv('/home/nkorojoseph/Documents/trialdata/commits.csv',header=None,skiprows=1, names=commit_headers)
cusers = commits.user_id.tolist()
cproj = commits.projectid.tolist()
#print(fusers)
#print(forks.head())
C.add_edges_from(commits.values)

totalusers = set(fusers + wusers + cusers + pusers)
totalprojec = set(fproj + wproj + cproj + pproj)
c = 0
G = nx.DiGraph()
for i in totalusers:
    for j in totalprojec:
        if F.has_edge(i,j) and W.has_edge(i,j) and P.has_edge(i,j) and C.has_edge(i,j):
            #print('user %s found in (%s, %s)'% (i,i,j))
            G.add_edge(i,j)
            c = c + 1

#print(c)
#print(G.edges())
nx.draw(W,with_labels=True,node_color='b')
plt.savefig(testfile+'graph')
plt.show()
occur_users = [u[0] for u in G.edges()]
occur_projects = [u[1] for u in G.edges()]

#max(occur_users, key=occur_users.count)
#nx.draw(F,with_labels=True,node_color='g')
#plt.show()
usercount = Counter(occur_users)

#nx.draw(G,with_labels = True,node_color='y')
#plt.show()

#print( 'Users with the four characteristics to a project ',G.edges() )

Recommended_users = [u[0] for u in G.in_edges(targetp)]

#print('Recommend users for the new project is project members of ',targetp,Recommended_users)

j = dict(usercount)
normUsercount = {}
for key,value in j.items():
    #print('Users Experience level =  {} -> {}'.format(key,value))
    normUsercount[key] = value/len(j)
#print('Normalized Experienced level',normUsercount)

dicOfQualifiedUsers = {}
for u in G.in_edges(targetp):
    dicOfQualifiedUsers[u[0]] = u[1]

QUsers = []   
for key,value in dicOfQualifiedUsers.items():
    QUsers.append(key)

#print('Qualified Users',QUsers)    
#getting the experience level all qualified users
QUserExp = {}
for user in QUsers:
    for key,value in normUsercount.items():
        if user == key:
            QUserExp[user] = value
            
#appending the concepts of the old readme file that selected users have worked with to the user,
#to generate concepts and profile for the user cum developer.
#getting project the Qusers were trusted partakers of 
#for x in occur_users,occur_projects:

QUserProj = {}

for user, project in zip(occur_users, occur_projects):
    if user in QUserExp:
        QUserProj[user] = project

prolang = []
with open(r"/home/nkorojoseph/Desktop/githubProjLang/languages.csv",'r') as Prolang:
    reader = csv.reader(Prolang)
    for row in reader:
        prolang.append(row)
#building a profile based on the programming languages for each user
#convert the programming languages into a dictionary
dicti = {}
for l2 in prolang:
    dicti['p'+l2[0]] = l2[1:]
#print(dicti)
#print(QUserProj)
QUsersProfile = {}
for key,value in QUserProj.items():
    for key1,value1 in dicti.items():
        if value==key1:
            QUsersProfile[key] = value1
#print('Users Programming language Profile',QUsersProfile)
QUsersProfileCount = {}
for key,value in QUsersProfile.items():
    QUsersProfileCount[key] = len(value)

#print('counter for programming languages',QUsersProfileCount)
#QUsersProfileCount = dict(Counter(QUsersProfileCount))

newprolang = []
with open("/home/nkorojoseph/Desktop/newprojectreadme/pnewlang.csv",'r') as NewProlang:
    reader = csv.reader(NewProlang)
    for row in reader:
        newprolang.append(row)
#building a profile based on the programming languages for each project
newdicti = {}
for l2 in newprolang:
    newdicti[l2[0]] = l2[1:]
#print('new dictionary',newdicti)
langsim = {}
for key,value in QUsersProfile.items():
    lang_sim = set(value).intersection(set(newdicti['pnew']))
    langsim[key] = len(lang_sim)/len(set(newdicti['pnew']))
print('Project relevance of each user \n langsim',langsim)
print('List of Qualified users with their Experience Level\n QUserExp',QUserExp)
totalRel = {}
for key,value in QUserExp.items():
    for k1,v2 in langsim.items():
        if key in langsim and k1 in QUserExp:
            totalRel[key] = QUserExp[key] + langsim[key]
print('total relevance level',totalRel)
sorted(totalRel,key=totalRel.get, reverse=True)
Recommendation_list = {key: rank for rank, key in enumerate(sorted(totalRel, key=totalRel.get, reverse=True), 1)}
print(Recommendation_list)
n_recomm = dicslice(2,Recommendation_list.items())
print('Top two recommended developers for the new project',n_recomm)
predicted_users = [x for x in Recommendation_list]
print(predicted_users)
count = 0
real_predicted = []
for user in predicted_users:
    if user in testusers:
        count = count + 1
        real_predicted.append(user)
print(real_predicted,'realpredicted')
acurracy = count/len(testusers)
print('Recommendation Acurracy = ',acurracy)


# In[37]:


keys = set(totalRel.keys()) | set(j.keys())
keys1 = list(QUserExp.keys() | j.keys())
print(j.keys(), QUserExp.keys())

print([QUserExp.get(x, 0) for x in keys1])
print([j.get(x, 0) for x in keys1])
x = [QUserExp.get(x, 0) for x in keys1]
y = [j.get(x, 0) for x in keys]
plt.scatter(x,y)
plt.savefig(testfile+'userexp')
plt.show()


print('Correlation between Trusted Users and Experience level = ',numpy.corrcoef(
    [QUserExp.get(x, 0) for x in keys1],
    [j.get(x, 0) for x in keys1])[0, 1])


# In[38]:



keys =  set(langsim.keys()) | set(j.keys())

print(j.keys(), langsim.keys())

print([langsim.get(x, 0) for x in keys])
print([j.get(x, 0) for x in keys])
x = [langsim.get(x, 0) for x in keys]
y = [j.get(x, 0) for x in keys]
plt.scatter(x,y)
plt.savefig(testfile+'langsim')
plt.show()

print('Correlation between programming language similarity and Trusted developers = ',numpy.corrcoef(
    [langsim.get(x, 0) for x in keys],
    [j.get(x, 0) for x in keys])[0, 1])


# In[ ]:




