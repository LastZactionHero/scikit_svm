# Use a Support Vector Machine to fit a line to housing price data
#
#
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# Housing prices in Lafayette, CO, in dollars
# Period of 240 months, ending in April 2016
y = np.array([156700,156000,155100,154100,153600,153300,153100,153400,154200,155000,155800,156500,156800,157300,158500,159700,160900,162300,163900,165400,166800,168100,169200,169900,170100,170000,169900,170200,170600,171100,172000,173200,174600,176000,177100,179000,182100,184800,186500,188500,190800,193400,196100,198800,201300,203600,205500,207400,209700,212300,215400,218900,222000,224900,227400,229600,232000,234700,237000,239600,242300,244600,246800,249200,250900,251500,252300,252900,252800,252200,251500,250700,250400,251100,252300,253000,253100,253200,253200,253300,253300,253200,254100,254800,254100,252600,251300,250000,249200,248800,248700,248800,249200,250000,251000,251800,252700,253900,254600,254900,255300,255900,256000,256000,256200,256800,257300,257700,258300,258400,258300,258300,258800,259500,260100,260400,260500,260700,261300,261700,261800,261500,261000,260900,261300,261600,261400,261300,261200,261000,260100,259100,259100,259800,260700,261900,262800,263300,263700,264000,264600,265500,266000,265700,265000,265000,265600,265800,265000,263600,262800,263000,262200,260200,258200,257500,257600,257900,258000,258200,258600,259000,259400,259400,259200,259400,259700,260300,260900,260700,260000,258800,257900,257000,256300,256100,256700,257200,257900,258600,258100,256500,255000,253600,252700,252600,253200,253900,255000,256600,258400,260300,262300,264200,266400,268300,269800,271400,273100,274600,276000,278100,280200,281300,282300,284200,287000,290400,294400,298800,302300,304300,305400,306400,307800,309700,311400,312700,314200,315600,316900,318400,320700,324000,328000,332600,338200,344800,351900,358900,364700,370100,376000,380800,384400,388400,393200,397400,399800,400900])
y =  y / 1000

# Months, adding polynomaial features
x_range = np.array(range(240))
X = np.array([
    np.power(x_range,1),
    np.power(x_range,2),
    np.power(x_range,3),
    np.power(x_range,4),
    np.power(x_range,5),
]).transpose()

# Bias
X = preprocessing.scale(X)
ones = np.ones([len(X), 1])
X = np.concatenate((ones, X), axis=1)

# Split into training, cross validation, and test data
X_train = np.array([])
X_cross = np.array([])
X_test  = np.array([])

Y_train = np.array([])
Y_cross = np.array([])
Y_test  = np.array([])

X_test_range = np.array([])

for i in x_range:
    x = X[i]
    if (i % 10) in [0,1,2,3,4,5]:
        X_train = np.append(X_train, x)
        Y_train = np.append(Y_train, y[i])
    elif (i % 10) in [6,7]:
        X_cross = np.append(X_cross, x)
        Y_cross = np.append(Y_cross, y[i])
    else:
        X_test = np.append(X_test, x)
        Y_test = np.append(Y_test, y[i])
        X_test_range = np.append(X_test_range, i)


X_train = X_train.reshape(-1, X.shape[1])
X_cross = X_cross.reshape(-1, X.shape[1])
X_test  = X_test.reshape(-1, X.shape[1])

# Train a SVM
min_cost = None     # minimum cost found so far
best_c = 0          # best penalty parameter found

for c in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]:
    # Fit to the training set
    svr = SVR(kernel='linear', C=c, gamma='auto')
    fit = svr.fit(X_train, Y_train)

    # Find error in cross validation set
    y_pred = fit.predict(X_cross)
    error = mean_squared_error(y_pred, Y_cross)
    print "%d: %f" % (c, error)

    if((min_cost == None) or (error < min_cost)):
        min_cost = error
        best_c = c

print "Lowest error: %d: %f" % (best_c, min_cost)

# Predict training set
svr = SVR(kernel='linear', C=best_c, gamma='auto')
fit = svr.fit(X_cross, Y_cross)
y_pred = fit.predict(X_test)

# Plot all data
plt.scatter(x_range, y, c='aqua', label='Actual')
plt.hold('on')

# Plot test predictions
plt.plot(X_test_range, y_pred, c='red', label='Prediction', linewidth=3)
plt.xlabel('Months')
plt.ylabel('Housing Price, in Thousands $')

plt.legend()
plt.show()