# import tensorflow as tf
# import numpy
# import matplotlib.pyplot as plt
# from IPython import embed
# 
# rng = numpy.random
# 
# # learning_rate = 0.01
# training_epochs = 200
# display_step = 50
# 
# 
# x1 = numpy.array(range(240))
# x1_norm = x1
# # 
# # x2 = 0 * numpy.power(numpy.array(range(240)), 2)
# # x2_norm = (x2 - x2.mean()) / x2.max()
# # 
# # x3 = 0 * numpy.power(numpy.array(range(240)), 3)
# # x3_norm = (x3 - x3.mean()) / x3.max()
# 
# train_X = x1_norm #numpy.array([x1_norm, x2_norm, x3_norm]).transpose()
# train_Y = numpy.asarray([156700,156000,155100,154100,153600,153300,153100,153400,154200,155000,155800,156500,156800,157300,158500,159700,160900,162300,163900,165400,166800,168100,169200,169900,170100,170000,169900,170200,170600,171100,172000,173200,174600,176000,177100,179000,182100,184800,186500,188500,190800,193400,196100,198800,201300,203600,205500,207400,209700,212300,215400,218900,222000,224900,227400,229600,232000,234700,237000,239600,242300,244600,246800,249200,250900,251500,252300,252900,252800,252200,251500,250700,250400,251100,252300,253000,253100,253200,253200,253300,253300,253200,254100,254800,254100,252600,251300,250000,249200,248800,248700,248800,249200,250000,251000,251800,252700,253900,254600,254900,255300,255900,256000,256000,256200,256800,257300,257700,258300,258400,258300,258300,258800,259500,260100,260400,260500,260700,261300,261700,261800,261500,261000,260900,261300,261600,261400,261300,261200,261000,260100,259100,259100,259800,260700,261900,262800,263300,263700,264000,264600,265500,266000,265700,265000,265000,265600,265800,265000,263600,262800,263000,262200,260200,258200,257500,257600,257900,258000,258200,258600,259000,259400,259400,259200,259400,259700,260300,260900,260700,260000,258800,257900,257000,256300,256100,256700,257200,257900,258600,258100,256500,255000,253600,252700,252600,253200,253900,255000,256600,258400,260300,262300,264200,266400,268300,269800,271400,273100,274600,276000,278100,280200,281300,282300,284200,287000,290400,294400,298800,302300,304300,305400,306400,307800,309700,311400,312700,314200,315600,316900,318400,320700,324000,328000,332600,338200,344800,351900,358900,364700,370100,376000,380800,384400,388400,393200,397400,399800,400900]) / 1000.0
# 
# 
# # X = tf.placeholder("float", shape=[None, 3])
# X = tf.placeholder("float")
# Y = tf.placeholder("float")
# 
# # W = tf.Variable(tf.random_normal([3,1]), name="weight")
# W = tf.Variable(rng.randn(), name="weight")
# b = tf.Variable(rng.randn(), name="bias")
# 
# activation = tf.add(tf.mul(X,W), b)
# cost = tf.reduce_sum(tf.pow(activation-Y, 2)) / (2*240)
# 
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 
# init = tf.initialize_all_variables()
# 
# with tf.Session() as sess:
#     sess.run(init)
# 
#     for epoch in range(training_epochs):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
# 
#         print "Epoch:", '%04d' % epoch
#         print "Cost: ", '%f' % sess.run(cost, feed_dict={X: train_X, Y: train_Y})
# 
#     sess.run(cost, feed_dict={X: train_X, Y: train_Y})
# 
#     v = sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
#     print sess.run(W)
# 
#     plt.plot(range(240), train_Y, 'ro', label='Original Data')
#     plt.plot(range(240), sess.run(activation, feed_dict={X: train_X}), label='Fitted Line')
# 
#     # embed()
# 
#     # train_Y[239]
#     # txdemo = numpy.reshape(train_X[239], (1,3))
#     # sess.run(activation, feed_dict={X: txdemo})
#     
#     plt.savefig('lafayette_prices.png')
#     
#     

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from IPython import embed
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 200
display_step = 50

# Training Data
train_X = numpy.array([range(240), range(240)]).transpose() * 1.0

train_Y = numpy.asarray([156700,156000,155100,154100,153600,153300,153100,153400,154200,155000,155800,156500,156800,157300,158500,159700,160900,162300,163900,165400,166800,168100,169200,169900,170100,170000,169900,170200,170600,171100,172000,173200,174600,176000,177100,179000,182100,184800,186500,188500,190800,193400,196100,198800,201300,203600,205500,207400,209700,212300,215400,218900,222000,224900,227400,229600,232000,234700,237000,239600,242300,244600,246800,249200,250900,251500,252300,252900,252800,252200,251500,250700,250400,251100,252300,253000,253100,253200,253200,253300,253300,253200,254100,254800,254100,252600,251300,250000,249200,248800,248700,248800,249200,250000,251000,251800,252700,253900,254600,254900,255300,255900,256000,256000,256200,256800,257300,257700,258300,258400,258300,258300,258800,259500,260100,260400,260500,260700,261300,261700,261800,261500,261000,260900,261300,261600,261400,261300,261200,261000,260100,259100,259100,259800,260700,261900,262800,263300,263700,264000,264600,265500,266000,265700,265000,265000,265600,265800,265000,263600,262800,263000,262200,260200,258200,257500,257600,257900,258000,258200,258600,259000,259400,259400,259200,259400,259700,260300,260900,260700,260000,258800,257900,257000,256300,256100,256700,257200,257900,258600,258100,256500,255000,253600,252700,252600,253200,253900,255000,256600,258400,260300,262300,264200,266400,268300,269800,271400,273100,274600,276000,278100,280200,281300,282300,284200,287000,290400,294400,298800,302300,304300,305400,306400,307800,309700,311400,312700,314200,315600,316900,318400,320700,324000,328000,332600,338200,344800,351900,358900,364700,370100,376000,380800,384400,388400,393200,397400,399800,400900]) / 1000.0
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(tf.zeros([2,1]), name="weight")
b = tf.Variable(1.0, name="bias")

# Construct a linear model
activation = tf.add(tf.matmul(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            embed()
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'



    #Graphic display
    plt.plot(range(240), train_Y, 'ro', label='Original data')
    plt.plot(range(240), sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig('lafayette_prices.png')