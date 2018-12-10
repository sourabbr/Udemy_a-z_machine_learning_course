#do the data preprocessing before doing this template

#logistic regression fit		feature scaling is required
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#k-nn fit				feature scaling is required
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#svm fit				feature scaling is required
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)		#for linear svm
classifier=SVC(kernel='rbf',random_state=0)			#for gaussian kernel(non linear)
classifier.fit(X_train,y_train)

#naive bayes fit		feature scaling is required
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#decision tree fit		feature scaling is required
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#random forest fit		feature scaling is required
from sklearn.ensemble import RandomForestClassier
classifier=RandomForestClassier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#predict logistic regression,k-nn,svm,naive bayes,decision tree,random forest
y_pred=classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=comfusion_matrix(y_test,y_pred)

#plot logistic regression,k-nn,svm,naive bayes,decision tree,random forest
from matplotlib.colores import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01)
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X3.ravel()]).T).reshape(X1,shape),alpha=075,cmap=ListedColormap('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.uniquely(y_set)):
	plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
		c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic regression')
plt.Xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()



