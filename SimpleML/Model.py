from .Activation import ActivationSoftmaxLossCategoricalCrossentropy as SoftmaxCCEloss
from .Loss import LossCategoricalCrossentropy as LossCCE
from .Activation import ActivationSoftmax
from .Normalize import BatchNorm1D
from .Layer import LayerInput
import numpy as np
import pickle
import copy


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
        # the above is used only when both softmax and CCELoss are used

    # Add objects to the model
    def add(self, Layer):
        self.layers.append(Layer)

    # Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        self.input_layer = LayerInput()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(
            self.loss, LossCCE
        ):
            self.softmax_classifier_output = SoftmaxCCEloss()

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):
        """
        Iterate over the parameters & layers &
        Update each layer with each set of parameters
        """
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)  # unpack weights and biases

    # Saves the parameters to a file
    def save_parameters(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    # Load the weights & updates a model instance with them
    def load_parameters(self, path):
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, "wb") as f:
            pickle.dump(model, f)

    # Load the model by creating a static method which can be
    # called without initializing the model class at all.
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    # Perform the forward pass
    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    # Performs backward pass
    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Evaluate the model based on the data passes on-demand
    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(
            f"validation, "
            + f"acc: {validation_accuracy:.3f}, "
            + f"loss: {validation_loss:.3f}"
        )

    # Train the model
    def train(
        self, X, y, *, epochs=1, batch_size=None, print_every=100, validation_data=None
    ):
        # Initialize accuracy object
        self.accuracy.init(y)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):
            print(f"epoch: {epoch}")
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterate over steps
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regularization=True
                )
                loss = data_loss + regularization_loss

                # Get predictions and accuracy from the last saved layer
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.preUpdateParams()
                for layer in self.trainable_layers:
                    self.optimizer.updateParams(layer)
                self.optimizer.postUpdateParams()

                # print a summary
                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step: {step}, "
                        + f"acc: {accuracy:.3f}, "
                        + f"loss: {loss:.3f} ("
                        + f"data_loss: {data_loss:.3f}, "
                        + f"reg_loss: {regularization_loss:.3f}), "
                        + f"lr: {self.optimizer.current_learning_rate}"
                    )

            # Get and print epoch loss and accuracy
            (
                epoch_data_loss,
                epoch_regularization_loss,
            ) = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(
                f"training, "
                + f"acc: {epoch_accuracy:.3f}, "
                + f"loss: {epoch_loss:.3f} ("
                + f"data_loss: {epoch_data_loss:.3f}, "
                + f"reg_loss: {epoch_regularization_loss:.3f}), "
                + f"lr: {self.optimizer.current_learning_rate}"
            )

            # If there is a validation data
            if validation_data is not None:
                # Evaluate the model
                self.evaluate(*validation_data, batch_size=batch_size)

            # TODO: clear all the outputs and print only the last value
            # TODO: obtained after training.

    # Predict on the samples
    def predict(self, X, *, batch_size=None):
        # Default value if batch size in None itself
        prediction_steps = 1

        # Calculate the number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        # Stack and return results as a np.array
        # with each prediction present in the array.
        return np.vstack(output)
