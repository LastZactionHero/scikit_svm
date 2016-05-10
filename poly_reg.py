# http://www.voidcn.com/blog/helei001/article/p-5781037.html
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import embed

# %% Let's create some toy data
# plt.ion()
n_observations = 240
# fig, ax = plt.subplots(1, 1)
# xs = np.linspace(-3, 3, n_observations)
xs = np.array(range(240))
xs = (xs) / float(xs.max())
# ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

xs_feat = np.array([
    np.power(xs,0),
    np.power(xs,1),
    np.power(xs,2),
    np.power(xs,3),
    np.power(xs,4),
]).transpose()

ys = np.asarray([156700,156000,155100,154100,153600,153300,153100,153400,154200,155000,155800,156500,156800,157300,158500,159700,160900,162300,163900,165400,166800,168100,169200,169900,170100,170000,169900,170200,170600,171100,172000,173200,174600,176000,177100,179000,182100,184800,186500,188500,190800,193400,196100,198800,201300,203600,205500,207400,209700,212300,215400,218900,222000,224900,227400,229600,232000,234700,237000,239600,242300,244600,246800,249200,250900,251500,252300,252900,252800,252200,251500,250700,250400,251100,252300,253000,253100,253200,253200,253300,253300,253200,254100,254800,254100,252600,251300,250000,249200,248800,248700,248800,249200,250000,251000,251800,252700,253900,254600,254900,255300,255900,256000,256000,256200,256800,257300,257700,258300,258400,258300,258300,258800,259500,260100,260400,260500,260700,261300,261700,261800,261500,261000,260900,261300,261600,261400,261300,261200,261000,260100,259100,259100,259800,260700,261900,262800,263300,263700,264000,264600,265500,266000,265700,265000,265000,265600,265800,265000,263600,262800,263000,262200,260200,258200,257500,257600,257900,258000,258200,258600,259000,259400,259400,259200,259400,259700,260300,260900,260700,260000,258800,257900,257000,256300,256100,256700,257200,257900,258600,258100,256500,255000,253600,252700,252600,253200,253900,255000,256600,258400,260300,262300,264200,266400,268300,269800,271400,273100,274600,276000,278100,280200,281300,282300,284200,287000,290400,294400,298800,302300,304300,305400,306400,307800,309700,311400,312700,314200,315600,316900,318400,320700,324000,328000,332600,338200,344800,351900,358900,364700,370100,376000,380800,384400,388400,393200,397400,399800,400900]) / 100000.0

# %% tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# %% Instead of a single factor and a bias, we'll create a polynomial function
# of different polynomial degrees.  We will then learn the influence that each
# degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
W = tf.Variable(tf.random_normal([5,1]))
Y_pred = tf.matmul(X,W)
# Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
# for pow_i in range(1, 5):
#     W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
#     Y_pred = tf.add(tf.mul(tf.pow(X, pow_i), W), Y_pred)

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# %% if we wanted to add regularization, we could add other terms to the cost,
# e.g. ridge regression has a parameter controlling the amount of shrinkage
# over the norm of activations. the larger the shrinkage, the more robust
# to collinearity.
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# %% We create a session to use the graph
n_epochs = 2000
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        print epoch_i
        for (x, y) in zip(xs_feat, ys):
            x = np.matrix(x)
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(cost, feed_dict={X: np.matrix(xs_feat), Y: np.matrix(ys)})
        print(training_cost)

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

    plt.plot(xs, ys, 'ro', label='Original data')
    plt.plot(xs, sess.run(Y_pred, feed_dict={X: np.matrix(xs_feat)}), label="Predicted")
    plt.legend()
    plt.savefig('lafayette_prices.png')