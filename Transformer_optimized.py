from datetime import datetime
from time import time
import json
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import Callback

from kerastuner.tuners import RandomSearch

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from utils import rmse, coeff_determination, smape


class ConsTransformer(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self):
        """ Initialization of the object
        """

        with open("parameters.json") as f:
            parameters = json.load(f)


        # Get model hyperparameters
        self.look_back = parameters["look_back"]
        self.n_features = parameters["n_features"]
        self.horizon = parameters["horizon"]

        # Get directories name
        self.log_dir = parameters["log_dir"]
        self.checkpoint_dir = parameters["checkpoint_dir"]

        self.head_size=256
        self.num_heads=4
        self.ff_dim=4
        self.num_transformer_blocks=4
        self.mlp_units=[128]
        self.mlp_dropout=0.4
        self.dropout=0.25
        self.n_units_init = parameters["n_units_init"]
        self.n_units_end = parameters["n_units_end"]

    def transformer_encoder(self,
        inputs):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
        key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


    def build(self, hp):
        """ Build the model architecture
        """

        inputs = keras.Input(shape=(self.look_back, self.n_features))
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in [hp.Int(f"mlp_units", self.n_units_init, self.n_units_end, step=16)]:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        # output layer
        outputs = layers.Dense(self.horizon, activation="relu")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                            loss = ['mse'],
                            metrics=[rmse, 'mae', smape, coeff_determination],
                            )
        return model


    def restore(self,
        filepath):
        """ Restore a previously trained model
        """

        # Load the architecture
        self.best_model = load_model(filepath, custom_objects={'smape': smape,
                                                         #'mape': mape,
                                                         'rmse' : rmse,
                                                         'coeff_determination' : coeff_determination})

        ## added cause with TF 2.4, custom metrics are not recognize custom metrics with only load-model
        self.best_model.compile(
            optimizer='adam',
            loss = ['mse'],
            metrics=[rmse, 'mae', smape, coeff_determination])

    def plot_model_rmse_and_loss(self,history):
    
        # Evaluate train and validation accuracies and losses
        
        train_rmse = history.history['rmse']
        val_rmse = history.history['val_rmse']
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Visualize epochs vs. train and validation accuracies and losses
        
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.plot(train_rmse, label='Training RMSE')
        plt.plot(val_rmse, label='Validation RMSE')
        plt.legend()
        plt.title('Epochs vs. Training and Validation RMSE')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Epochs vs. Training and Validation Loss')
        
        plt.show()
    
    def train_advance(self,
        X_train,
        y_train,
        epochs=600,
        batch_size=64):
        """ Training the network
        :param X_train: training feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_train: training target vectors
        :type 2-D Numpy array of float values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """
        tuner = RandomSearch(
            self.build,
            objective = 'val_loss',
            max_trials = 3, #5
            executions_per_trial = 2, #3
            directory='ktuner2',
            project_name='kerastuner_bayesian_tfm',
            overwrite=True,
            )
        tuner.search(X_train, y_train, epochs=5, validation_split=0.2)
        print(tuner.search_space_summary())

        self.best_model = tuner.get_best_models()[0]
        print(self.best_model.summary())


        # self.model = self.build()
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        #                    loss = ['mse'],
        #                    metrics=[rmse, 'mae', smape, coeff_determination],
        #                    )
        # print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        filepath = self.checkpoint_dir+"/Transformer.best"+datetime.now().strftime('%d%m%Y_%H%M%S')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor, checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])

        drawplot = self.plot_model_rmse_and_loss(callback_history)

    def evaluate(self,
        X_test,
        y_test):
        """ Evaluating the network
        :param X_test: test feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_test: test target vectors
        :type 2-D Numpy array of int values
        :return  Evaluation losses
        :rtype 5 Float tuple
        :raise -
        """

        y_pred = self.best_model.predict(X_test)

        # Print accuracy if ground truth is provided
        """
        if y_test is not None:
            loss_ = session.run(
                self.loss,
                feed_dict=feed_dict)
        """

        _, rmse_result, mae_result, smape_result, _ = self.best_model.evaluate(X_test, y_test)

        r2_result = r2_score(y_test.flatten(),y_pred.flatten())
        mymodel = self.best_model
        return mymodel, rmse_result, mae_result, smape_result, r2_result
