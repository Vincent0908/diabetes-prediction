import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras


np.random.seed(42)
tf.random.set_seed(42)


def train_logistic_regression(X_train, y_train, model_path: str = None) -> LogisticRegression:
    print("\n[LOGISTIC REGRESSION]")

    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train, y_train)

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"  [SAVE] -> {model_path}")

    return model


def build_neural_network(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="DiabetesNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_neural_network(
    X_train, y_train, X_val, y_val,
    model_path: str = None,
    epochs: int = 150,
    batch_size: int = 32
):
    print("\n[NEURAL NETWORK]")

    model = build_neural_network(input_dim=X_train.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Stopped at epoch {final_epoch}")
    print(f"  Val accuracy: {history.history['val_accuracy'][-1]:.4f}")

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"  [SAVE] -> {model_path}")

    return model, history
