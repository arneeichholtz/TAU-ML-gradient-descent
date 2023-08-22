import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

###
### The runtime is about 9 seconds per epoch, so 100*9sec for all epochs. This is not very good although I used pd.apply and not for loops. I think using
### numpy would have been better for vectorization, but I thought pd.apply was also an optimized vectorization method. 
###

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == "__main__":

    # Input and target
    X = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    df = pd.DataFrame(X)
    target = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    # Variables
    dim = X.shape[1]  # dimensionality of input
    N = X.shape[0]
    M = 3  # number of cells in hidden layer
    K = 1  # number of cells in output layer
    learning_rate = 2

    epochs = 10
    max_iter = 2000

    begin = time.time()
    ep_errors = np.zeros(shape=(epochs, max_iter))

    print("Gradient descent epochs: ")
    for curr_ep in tqdm(range(epochs)):
        iter_errors = np.zeros(max_iter) # per run, the errors for all iterations

        # Initialize weight vectors
        w_hidden = np.random.normal(loc=0, scale=1, size=(dim, M+1))  # M + 1 for bias term
        w_out = np.random.normal(loc=0, scale=1, size=M+1)

        b_hid = w_hidden[:, 0]  # bias for hidden layer, shape (M,)
        w_hid = w_hidden[:, 1:]  # weights for hidden layer, shape (dim, M)

        b_out = w_out[0]  # bias for output layer, shape (K,)
        w_out = w_out[1:]  # weights for output layer, shape (M, K)

        for curr_iter in range(max_iter):
            # Forward propagation
            a_j = df.apply(lambda x: b_hid + np.matmul(w_hid, x), axis=1, raw=True)  # scalar for each hidden cell, shape (N, M)
            Z = sigmoid(a_j)

            a_k = Z.apply(lambda x: b_out + np.matmul(w_out, x), axis=1, raw=True)  # scalar for each output cell, shape (N, K)
            y = sigmoid(a_k)

            error = 1/8 * np.sum((y - target) ** 2)
            iter_errors[curr_iter] = error

            # Back propagation
            # Updating according to:
            #   slope (gradient) is dE/dx, where x is b_hid, w_hid, b_out or w_out
            #   steps = learning_rate * slope
            #   w^(t+1) = w^(t) - steps

            """
            Updating b_hid:
                slope (gradient) dE/db_hid = (y - target) * deriv_sigmoid(a_k) * w_out * deriv_sigmoid(a_j) * 1
            """
            delta = (y - target) * deriv_sigmoid(a_k)  # shape (N, K)
            prod = delta.apply(lambda x: x * w_out)  # product of delta and w_out
            prod = np.vstack(prod)  # make 2d array of shape (N, K, M)
            db_hid_n = prod * deriv_sigmoid(a_j) * 1  # shape (N, M)
            db_hid = np.array(np.sum(db_hid_n, axis=0)) # gradient shape (M)
            steps = learning_rate * db_hid
            b_hid = b_hid - steps

            """
            Updating w_hid:
                slope (gradient) dE/dw_hid = (y - target) * deriv_sigmoid(a_k) * w_out * deriv_sigmoid(a_j) * X
                                        = dE/db_hid * X                       (ie, previous gradient times X)
            """
            X_exp = np.expand_dims(X, 1) # add another dimension to X so output is correct shape
            db_hid_df = pd.DataFrame(db_hid_n)
            dw_hid_df = db_hid_df.apply(lambda m: np.matmul(np.array(m).reshape(3, -1), X_exp[m.name]), axis=1)
            dw_hid_n = np.stack(np.array(dw_hid_df)).transpose(0, 2, 1)
            dw_hid = np.sum(dw_hid_n, axis=0)
            steps = learning_rate * dw_hid
            w_hid = w_hid - steps

            """
            Updating b_out:
                slope (gradient) dE/db_out = (y - target) * deriv_sigmoid(a_k) * 1
                                        = delta * 1
            """
            db_out_n = delta * 1 # shape (N, K)
            db_out = np.sum(db_out_n, axis=0) # gradient shape (K)
            steps = learning_rate * db_out
            b_out = b_out - steps

            """
            Updating w_out:
                slope (gradient) dE/dw_out = (y - target) * deriv_sigmoid(a_k) * Z
                                        = delta * Z
            """
            delta_df = pd.DataFrame(delta)  # shape (N)
            Z_arr = np.array(Z)  # shape (N, M)
            dw_out_df = delta_df.apply(lambda x: x.values * Z_arr[x.name], axis=1)
            dw_out_n = np.stack(np.array(dw_out_df))
            dw_out = np.sum(dw_out_n, axis=0)  # gradient shape (M, K)
            steps = learning_rate * dw_out
            w_out = w_out - steps
            curr_iter = curr_iter + 1

        ep_errors[curr_ep] = iter_errors
        curr_ep = curr_ep + 1

    end = time.time() - begin
    print(f"\nRuntime: {end / 60:.0f} minutes and {end % 60:.2f} seconds")

    print(ep_errors.shape)

    # Getting data
    errors = np.array(ep_errors)
    err_means = np.mean(errors, axis=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2000)
    ax.plot(x, err_means)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel("Mean Squared Error", fontsize=15)
    ax.set_title("M = 3 hidden cells | MSE as function of iteration index", fontsize=15)
    plt.show()