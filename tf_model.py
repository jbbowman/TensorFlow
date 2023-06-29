import tensorflow as tf

class LoanPredictionModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=10, validation_split=0.2):
        self.history = self.model.fit(X, y, epochs=epochs, validation_split=validation_split, verbose=1)
