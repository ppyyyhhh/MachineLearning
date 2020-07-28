import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

def init_w_with_bias(x):
    return np.ones((x.shape[1], 1)) # naive initialization


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def predict_logistic(x, w):
    return sigmoid(np.matmul(x,w)) #sigmoid function


def classify(x, w):
    return np.round(predict_logistic(x, w))


def plot_result_logistic_regression(x, y, w):
    clear_output(wait=True)
    #     print(x, y, w)
    plt.plot(x[:, 0], y, 'o')
    plt.plot(x[:, 0], sigmoid(np.matmul(x, w)))
    plt.plot(x[:, 0], classify(x, w))

    plt.ylim((-0.1, 1.1))

    plt.show()


def plot_loss_metric(iterations, loss, test_loss):
    import matplotlib.pyplot as plt
    clear_output(wait=True)
#     plt.plot(iterations, loss, label='train')
#     plt.plot(iterations, test_loss, label='test')
    plt.ylim((-0.1, 0.5))

    train, = plt.plot(iterations, loss, label='train')
    test, = plt.plot(iterations, test_loss, label='test')
    plt.legend(handles=[train, test])
    plt.show()


lambda_v_default = 1e-4


def logloss(x, y, w):
    y_hat = predict_logistic(x, w)
    first_term = y * np.log(y_hat)
    second_term = (1 - y) * np.log(1 - y_hat)
    L = -np.average(first_term + second_term)  # minus cross-entropy (log loss)
    return L


def l2loss(w, lambda_v=lambda_v_default):
    return 1 / 2 * lambda_v * np.matmul(w.T, w)[0][0]

def compute_gradients_logistic(x, y, w):
    grad=  np.matmul(x.T, (predict_logistic(x,w) - y))/x.shape[0]
    return grad

def compute_gradients_with_l2(x, y, w, lamba_v=lambda_v_default):
    return compute_gradients_logistic(x, y, w) + 2*lamba_v*w


def train_with_gradient_logistic_regression(x, y, w, iterations, learning_rate=0.01, reg=False):
    previous_loss = 100000000000
    for i in range(iterations):
        current_loss = logloss(x, y, w) + (0 if not reg else l2loss(w))
        if abs(previous_loss - current_loss) < 10 ** -7:
            break

        if i % 1000 == 0:
            plot_result_logistic_regression(x, y, w)
            accuracy = sum(classify(x, w) == y) / len(y)

            print(f"Iteration: {i} Loss:{current_loss} Accuracy: {accuracy}")
        weight_gradient = compute_gradients_logistic(x, y, w) if not reg else compute_gradients_with_l2(x, y, w)

        w -= learning_rate * weight_gradient
        previous_loss = current_loss
    plot_result_logistic_regression(x, y, w)
    accuracy = sum(classify(x, w) == y) / len(y)

    print(f"Stopped at iteration {i} with loss {current_loss} and accuracy {accuracy[0]}")
    return w


def train_with_gradient_logistic_regression_train_test(x, y, x_test, y_test, w, iterations, learning_rate=0.1,
                                                       reg=False):
    previous_loss = 100000000000
    loss = []
    metric = []
    test_loss = []
    test_metric = []
    iterations_count = []
    for i in range(iterations):
        current_loss = logloss(x, y, w) + (0 if not reg else l2loss(w))
        if abs(previous_loss - current_loss) < 10 ** -10:
            break

        if i % 10000 == 0:
            loss.append(current_loss)
            accuracy = sum(classify(x, w) == y) / len(y)
            test_accuracy = sum(classify(x_test, w) == y_test) / len(y_test)

            current_test_loss = logloss(x_test, y_test, w) + (0 if not reg else l2loss(w))
            test_loss.append(current_test_loss)
            iterations_count.append(i)
            #             print(len(loss), len(test_loss), len(iterations_count))
            plot_loss_metric(iterations_count, loss, test_loss)
            print(current_test_loss / current_loss)

            print(f"Iteration: {i} Loss:{current_loss}")
        weight_gradient = compute_gradients_logistic(x, y, w) if not reg else compute_gradients_with_l2(x, y, w)
        w -= learning_rate * weight_gradient
        previous_loss = current_loss
    #     plot_result_logistic_regression(x_test, y_test , w)
    print(f"Stopped at iteration {i} with loss {current_loss}")
    return w