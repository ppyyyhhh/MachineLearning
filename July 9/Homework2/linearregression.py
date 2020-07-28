from IPython.display import clear_output
import numpy as np

def plot_result(x, y, m=1, b=0, is_classif=False):
    clear_output(wait=True)

    plt.plot(x, y, 'o')
    plt.plot(x, m * x + b)
    if is_classif:
        plt.ylim((-0.1, 1.1))
    else:
        plt.ylim((0, 140000))

    plt.show()

# linear regression functions
def predict(x, w, b=0):  # also known as y_hat. w weights, b bias, y^
    return x * w + b

def init_w(x):
    return np.ones(x.shape) # naive initialization, 1

def init_b(x):
    return 0 # naive initialization, 0

def compute_error(x, w, y, b=0):
    return predict(x,w, b) - y  #find each error between y^ and y

def compute_mse(x, w, y, b=0):
    return np.average(compute_error(x, w, y, b) ** 2) # MSE = 1/n (y-y^)^^2


# train the model with only weight

def train(x, y, w, iterations, learning_rate=2):
    for i in range(iterations):
        current_loss = compute_mse(x, w, y)

        if i % 50 == 0:
            plot_result(x, y, w)
            print(f"Iteration: {i} Loss:{current_loss}")

        if compute_mse(x, w + learning_rate, y) < current_loss:
            w += learning_rate  # adding learning rate
        elif compute_mse(x, w - learning_rate, y) < current_loss:
            w -= learning_rate  # minus the learning rate on w
        else:
            return w


# train with bias and weight
def train_with_bias(x, y, w, b, iterations, learning_rate=5):
    for i in range(iterations):
        current_loss = compute_mse(x, w, y, b)

        if i % 50 == 0:
            plot_result(x, y, w, b)
            print(f"Iteration: {i} Loss:{current_loss}")

        if compute_mse(x, w, y, b + learning_rate) < current_loss:
            b += learning_rate
        elif compute_mse(x, w, y, b - learning_rate) < current_loss:
            b -= learning_rate
        elif compute_mse(x, w + learning_rate, y, b) < current_loss:
            w += learning_rate
        elif compute_mse(x, w - learning_rate, y, b) < current_loss:
            w -= learning_rate
        else:
            plot_result(x, y, w, b)

            print(f"Stopped at iteration {i} with loss {current_loss}")
            return w, b

#define gradient weight
def compute_weights_gradient(x, y, w, b):
    return 2*np.average(x * (predict(x,w,b) -y))

def compute_bias_gradient(x,y,w,b):
    return 2*np.average(predict(x,w,b) -y)


def train_with_gradient(x, y, w, b, iterations, learning_rate=0.005, is_classif=False):
    previous_loss = 100000000000
    errors = []
    all_ws = []
    for i in range(iterations):
        current_loss = compute_mse(x, w, y, b)
        if abs(previous_loss - current_loss) < 1:
            break
        if i % 10 == 0:
            plot_result(x, y, w, b, is_classif)
            print(f"Iteration: {i} Loss:{current_loss}")
            errors.append(compute_error(x, w, y, b))
            all_ws.append((w, b))

        weight_gradient = compute_weights_gradient(x, y, w, b)
        bias_gradient = compute_bias_gradient(x, y, w, b)
        w -= weight_gradient * learning_rate
        b -= bias_gradient * learning_rate
        previous_loss = current_loss
    plot_result(x, y, w, b, is_classif)
    print(f"Stopped at iteration {i} with loss {current_loss}")
    return w, b
