from customers import generate_customers, generate_dataframe, prep_data
from tf_model import LoanPredictionModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    loan_predictor = init_model()
    plot_history(loan_predictor.history)

def init_model():
    customer_data = generate_customers(20)
    data = generate_dataframe(customer_data)
    features, targets, _ = prep_data(data)

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    loan_predictor = LoanPredictionModel()
    loan_predictor.train(x_train_scaled, y_train)

    return loan_predictor
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
