from typing import Type, List

import numpy as np


class BaseActivation:
    def forward(self, layer: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, layer: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearActivation(BaseActivation):
    def forward(self, layer: np.ndarray) -> np.ndarray:
        return layer

    def backward(self, layer: np.ndarray) -> np.ndarray:
        return np.ones(layer.shape)


class ReluActivation(BaseActivation):
    def forward(self, layer: np.ndarray) -> np.ndarray:
        return np.maximum(layer, 0)

    def backward(self, layer: np.ndarray) -> np.ndarray:
        return np.where(layer > 0, 1, 0)


class LeakyReluActivation(BaseActivation):
    def __init__(self, coeff=1.0/5.5):
        self.coeff = coeff

    def forward(self, layer: np.ndarray) -> np.ndarray:
        return np.where(layer > 0, layer, layer * self.coeff)

    def backward(self, layer: np.ndarray) -> np.ndarray:
        return np.where(layer > 0, 1, self.coeff)


class SigmoidActivation(BaseActivation):
    def forward(self, layer: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-layer))

    def backward(self, layer: np.ndarray) -> np.ndarray:
        f = self.forward(layer)
        return f * (1 - f)


class BaseLossFunction:
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquareError(BaseLossFunction):
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return ((y_pred - y_true) ** 2) / 2

    def compute_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true)


class DenseNeuralNetwork:
    activation_klass: Type[BaseActivation] = LeakyReluActivation
    loss_function_klass: Type[BaseLossFunction] = MeanSquareError

    def __init__(self, blocks=(), bias=True):
        self.blocks: List[int] = blocks
        self.bias: bool = bias
        self.layers: List[np.ndarray] = self.initialize_layers()
        self.activation_fn: BaseActivation = self.activation_klass()
        self.loss_fn: BaseLossFunction = self.loss_function_klass()

    def initialize_layers(self) -> List[np.ndarray]:
        layers = []
        for prev_block, next_block in zip(self.blocks[:-1], self.blocks[1:]):
            # Random initializer
            layers.append(
                np.random.normal(
                    size=(prev_block + self.bias, next_block),
                    scale=2/(prev_block+next_block),
                )
            )
        return layers

    def forward(self, x_data: np.ndarray) -> List[np.ndarray]:
        assert x_data.shape[0] == self.blocks[0]

        nodes = []
        nodes.append(np.copy(x_data))
        node = np.concatenate([x_data] + [np.array([1])] * self.bias, axis=0)

        for layer in self.layers:
            node = np.dot(node, layer)
            nodes.append(np.copy(node))
            node = self.activation_fn.forward(np.concatenate([node] + [np.array([1])] * self.bias, axis=0))

        return nodes

    def backward(self, nodes: List[np.ndarray], labels: np.ndarray, learning_rate=0.001):
        # Compute gradient of loss over linear combination in last nodes
        y_true = labels
        y_pred = self.activation_fn.forward(nodes[-1])
        backprop_gradients = []

        loss_gradient = self.loss_fn.compute_gradient(y_pred=y_pred, y_true=y_true)
        last_nodes_gradient = self.activation_fn.backward(nodes[-1]) * loss_gradient

        _pre_last_nodes = np.concatenate(
            [self.activation_fn.forward(nodes[-2])] + [np.array([1.0])] * self.bias,
            axis=0,
        )
        backprop_gradients.append(
            np.expand_dims(_pre_last_nodes, axis=1) *
            np.expand_dims(last_nodes_gradient, axis=0)
        )

        for layer_idx in range(len(self.layers)-1, -1, -1):
            _node = nodes[layer_idx]
            _last_node_gradient = np.zeros(_node.shape)
            _activation_grad = self.activation_fn.backward(_node)
            for j in range(_last_node_gradient.shape[0]):
                s = 0
                for k in range(nodes[layer_idx+1].shape[0]):
                    s += self.layers[layer_idx][j,k] * last_nodes_gradient[k] * _activation_grad[j]
                _last_node_gradient[j] = s
            last_nodes_gradient = _last_node_gradient.copy()

            """
            _activation_gradient = self.activation_fn.backward(_node)
            _expanded_activation_gradient = np.concatenate(
                [_activation_gradient] + [np.array([1.0])] * self.bias,
                axis=0,
            )
            _repeated_grad = np.repeat(
                np.expand_dims(_expanded_activation_gradient, axis=1),
                repeats=self.layers[layer_idx].shape[1],
                axis=1,
            ) * self.layers[layer_idx]
            last_nodes_gradient = np.dot(_repeated_grad[self.bias:,:], last_nodes_gradient)
            """

            if layer_idx != 0:
                _pre_last_nodes = np.concatenate(
                    [self.activation_fn.forward(nodes[layer_idx-1])] + [np.array([1.0])] * self.bias,
                    axis=0,
                )
                backprop_gradients.append(
                    np.expand_dims(_pre_last_nodes, axis=1) *
                    np.expand_dims(last_nodes_gradient, axis=0)
                )

        # Backpropagation
        for layer_idx in range(len(backprop_gradients)):
            self.layers[len(self.layers) - 1 - layer_idx] -= learning_rate * backprop_gradients[layer_idx]


if __name__ == "__main__":
    neural_network = DenseNeuralNetwork(blocks=(2, 3, 3, 1), bias=True)

    class XORDataset:
        def __init__(self):
            self.data = [
                (np.array([0.0, 0.0]), np.array([0.0])),
                (np.array([1.0, 0.0]), np.array([1.0])),
                (np.array([0.0, 1.0]), np.array([1.0])),
                (np.array([1.0, 1.0]), np.array([0.0])),
            ]
            self.cur_idx = 0

        def __iter__(self):
            self.cur_idx = 0
            return self

        def __next__(self):
            if self.cur_idx >= 4:
                raise StopIteration

            val = self.data[self.cur_idx]
            self.cur_idx += 1
            return val

    dataset = XORDataset()

    for epoch in range(3000):
        print("EPOCH: ", epoch)
        for x_data, y_true in dataset:
            nodes = neural_network.forward(x_data)
            loss = neural_network.loss_fn.compute_loss(y_pred=nodes[-1], y_true=y_true)
            print(np.abs(loss).sum())
            neural_network.backward(nodes, y_true, learning_rate=0.005)

    print(neural_network.forward(np.array([0, 0]))[-1])
    print(neural_network.forward(np.array([0, 1]))[-1])
    print(neural_network.forward(np.array([1, 0]))[-1])
    print(neural_network.forward(np.array([1, 1]))[-1])
