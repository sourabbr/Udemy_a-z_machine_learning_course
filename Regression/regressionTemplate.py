#do the data preprocessing before this template

#linear fit
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#polynomial fit
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#svr fit	requires applying feature scaling manually
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#decision tree fit
from sklearn.tree import DecisionTreeRegressor
regressor=decisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#random forest fit
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#plot linear
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Number of graduates")
plt.xlabel('Year')
plt.ylabel("Number of graduates")
plt.show()

#plot polynomial
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Number of graduates")
plt.xlabel('Year')
plt.ylabel("Number of graduates")
plt.show()

#plot svr
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Number of graduates")
plt.xlabel('Year')
plt.ylabel("Number of graduates")
plt.show()

#plot decision tree
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or bluff")
plt.xlabel('position level')
plt.ylabel("level")
plt.show()

#plot random forest
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or bluff")
plt.xlabel('position level')
plt.ylabel("level")
plt.show()

#predict linear
lin_reg.predict(25)

#predict polynomial
lin_reg_2.predict(poly_reg.fit_transform(2019))

#predict svr
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[2018]]))))

#predict decision tree
y_pred=regressor.predict(6.5)

#predict random forest
y_pred=regressor.predict(6.5)

#selecting data using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=X,values=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#remove the respective column by not including the column number in line 103 whose p value is greater than significance level and then repeat line 104 and 105 till the p values of the columns are less than the significance level
