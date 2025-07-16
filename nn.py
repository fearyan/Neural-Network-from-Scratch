"""
A simple neural network for MNIST digit classification using only numpy and math.
No TensorFlow or PyTorch required!
"""
import numpy as np
import math
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# activation functions (these are like the brain's vibes)
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation for output layer."""
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # keeps things chill for big numbers
    return exps / np.sum(exps, axis=1, keepdims=True)

# loss function (how much we messed up)
def cross_entropy(predictions, labels):
    """Cross-entropy loss."""
    n = predictions.shape[0]
    log_likelihood = -np.log(predictions[range(n), labels])
    return np.sum(log_likelihood) / n

def cross_entropy_derivative(predictions, labels):
    """Derivative of cross-entropy loss."""
    n = predictions.shape[0]
    grad = predictions.copy()
    grad[range(n), labels] -= 1
    return grad / n

# loading the mnist data (handwritten digits, super classic)
def load_data():
    """Load and preprocess MNIST data."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32) / 255.0  # normalize so it's all between 0 and 1
    y = mnist['target'].astype(np.int64)
    # split into train and test (first 60k for training, rest for testing)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test

# making the weights (random start, but not too wild)
def initialize_parameters(input_size, hidden_size, output_size):
    """Initialize weights and biases."""
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * math.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * math.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# training time! (where the magic happens)
def train(X, y, hidden_size=128, output_size=10, epochs=10, lr=0.01, batch_size=64):
    """Train the neural network."""
    input_size = X.shape[1]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            # forward pass (let's see what the network thinks)
            Z1 = np.dot(X_batch, W1) + b1
            A1 = relu(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = softmax(Z2)
            # loss (how off were we?)
            loss = cross_entropy(A2, y_batch)
            # backprop (fixing our mistakes)
            dZ2 = cross_entropy_derivative(A2, y_batch)
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * relu_derivative(Z1)
            dW1 = np.dot(X_batch.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            # update weights (level up!)
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        # after each epoch, let's see how we're doing
        preds = predict(X, W1, b1, W2, b2)
        acc = np.mean(preds == y)
        loss_history.append(loss)
        acc_history.append(acc)
        print(f"epoch {epoch+1}/{epochs} - loss: {loss:.4f} - accuracy: {acc*100:.2f}%")
    return W1, b1, W2, b2, loss_history, acc_history

# making predictions (let's guess some digits)
def predict(X, W1, b1, W2, b2):
    """Predict class labels for samples in X."""
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return np.argmax(A2, axis=1)

# plot how the training went (vibes check)
def plot_training(loss_history, acc_history):
    """Plot training loss and accuracy."""
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(loss_history, marker='o')
    plt.title('training loss per epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(np.array(acc_history)*100, marker='o', color='orange')
    plt.title('training accuracy per epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# confusion matrix (where did we mess up? lol)
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('confusion matrix (test set)')
    plt.show()

# show off some sample predictions (flexing the results)
def show_sample_predictions(X_test, y_test, test_preds):
    """Show random sample predictions from the test set."""
    plt.figure(figsize=(10,6))
    for i in range(12):
        idx = np.random.randint(0, X_test.shape[0])
        img = X_test[idx].reshape(28,28)
        plt.subplot(3,4,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"pred: {test_preds[idx]}, true: {y_test[idx]}")
        plt.axis('off')
    plt.suptitle('sample predictions on test set')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# save the model weights (so you can flex later)
def save_model(W1, b1, W2, b2, filename='simple_nn_weights.npz'):
    """Save model weights to a file."""
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"model weights saved to {filename}")

# let's run the whole thing!
def main():
    X_train, y_train, X_test, y_test = load_data()
    W1, b1, W2, b2, loss_history, acc_history = train(X_train, y_train, hidden_size=128, epochs=10, lr=0.01)
    plot_training(loss_history, acc_history)
    print("\nevaluating on test set...")
    test_preds = predict(X_test, W1, b1, W2, b2)
    test_acc = np.mean(test_preds == y_test)
    print(f"test accuracy: {test_acc * 100:.2f}%")
    plot_confusion_matrix(y_test, test_preds)
    show_sample_predictions(X_test, y_test, test_preds)
    save_model(W1, b1, W2, b2)

if __name__ == "__main__":
    main()
