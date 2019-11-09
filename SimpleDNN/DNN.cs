using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	public class DNN {
		private Node[][] nodes;
		private Weight[][][] weights;
		private double learningRate = 0.1;

		public DNN(int inputNumber, int outputNumber, int[] hiddenNumbers) {
			// Initialise node layers, one input layer, however many hidden layers and one output layer
			nodes = new Node[2+hiddenNumbers.Length][];

			// Initialise the nodes in input layer
			nodes[0] = new Node[inputNumber+1];
			for (int inputCount = 0; inputCount < inputNumber; inputCount++) {
				// Create each node as an input node
				nodes[0][inputCount] = new Node(Node.ActivationType.Input);
			}
			// Creates a Bias Node
			nodes[0][inputNumber] = new Node(Node.ActivationType.Bias);

			// Initialise the nodes for each hidden layer
			for (int hiddenLayerCount = 0; hiddenLayerCount < hiddenNumbers.Length; hiddenLayerCount++) {
				// Initialise the nodes for the current layer including the bias node
				nodes[hiddenLayerCount + 1] = new Node[hiddenNumbers[hiddenLayerCount] + 1];
				for (int hiddenNodeCount = 0; hiddenNodeCount < hiddenNumbers[hiddenLayerCount]; hiddenNodeCount++) {
					// Create each node in this layer with the Sigmoid activation type
					nodes[hiddenLayerCount + 1][hiddenNodeCount] = new Node(Node.ActivationType.Sigmoid);
				}
				// Create a bias node
				nodes[hiddenLayerCount + 1][hiddenNumbers[hiddenLayerCount]] = new Node(Node.ActivationType.Bias);
			}

			// Initialise the output layer
			nodes[nodes.Length - 1] = new Node[outputNumber];
			for (int outputCount = 0; outputCount < outputNumber; outputCount++) {
				nodes[nodes.Length - 1][outputCount] = new Node(Node.ActivationType.Sigmoid);
			}

			Random rand = new Random();

			// Initialise the weights from each node in one layer to the nodes in the next layer
			weights = new Weight[nodes.Length-1][][];
			// For each layer
			for (int layer = 0; layer < weights.Length; layer++) {
				// Initialise the weights for the nodes that are passing their value forward
				weights[layer] = new Weight[nodes[layer].Length][];
				for (int inputIndex = 0; inputIndex < weights[layer].Length; inputIndex++) {
					// Initilise the weights for the nodes that are recieving values from the previous layer
					weights[layer][inputIndex] = new Weight[nodes[layer+1].Length];
					for (int outputIndex = 0; outputIndex < weights[layer][inputIndex].Length; outputIndex++) {
						// Give the weight a random value
						weights[layer][inputIndex][outputIndex] = new Weight(rand.NextDouble() * 2 - 1);
					}
				}
			}
		}

		public DNN(int inputNumber, int outputNumber) : this(inputNumber, outputNumber, new int[0]) {
			
		}

		public double[] Guess(double[] inputs) {
			for (int inputIndex = 0; inputIndex < nodes[0].Length - 1; inputIndex++) {
				nodes[0][inputIndex].Activate(inputs[inputIndex]);
			}

			for (int layer = 1; layer < nodes.Length; layer++) {
				for (int outputNode = 0; outputNode < nodes[layer].Length; outputNode++) {
					double nodeValue = 0;
					for (int inputNode = 0; inputNode < nodes[layer-1].Length; inputNode++) {
						nodeValue += nodes[layer - 1][inputNode].output * weights[layer - 1][inputNode][outputNode].value;
					}
					nodes[layer][outputNode].Activate(nodeValue);
				}
			}


			double[] ret = new double[nodes[nodes.Length-1].Length];
			for (int outputIndex = 0; outputIndex < ret.Length; outputIndex++) {
				ret[outputIndex] = nodes[nodes.Length - 1][outputIndex].output;
			}

			return ret;
		}

		public void Train(double[] inputs, double[] expectedOutputs, bool execute) {
			double[] guess = Guess(inputs);

			for (int outputIndex = 0; outputIndex < nodes[nodes.Length-1].Length; outputIndex++) {
				nodes[nodes.Length - 1][outputIndex].expected = expectedOutputs[outputIndex];
			}

			for (int layer = nodes.Length - 1; layer > 0; layer--) {
				// Get adjustments for all nodes except the Bias nodes, which is not present in the last layer
				for (int outputNode = 0; outputNode < nodes[layer].Length - (layer == nodes.Length - 1 ? 0 : 1); outputNode++) {
					double error = nodes[layer][outputNode].expected - nodes[layer][outputNode].output;
					double gradient = nodes[layer][outputNode].Derivative() * error * learningRate;

					for (int inputNode = 0; inputNode < nodes[layer-1].Length; inputNode++) {
						// If just starting on this layer, reset the expectation of the nodes of the previous layer
						if (outputNode == 0) {
							nodes[layer - 1][inputNode].expected = 0;
						}

						weights[layer - 1][inputNode][outputNode].Adjust(gradient * nodes[layer - 1][inputNode].output);
						nodes[layer - 1][inputNode].expected += weights[layer - 1][inputNode][outputNode].value * error;

						if (execute) {
							weights[layer - 1][inputNode][outputNode].Train();
						}
					}
				}
			}
		}

		public void Train(double[] inputs, double[] outputs) {
			Train(inputs, outputs, true);
		}
	}

	class Node {
		private double value;
		private double _output = 0;
		public double output {
			get { return activator == ActivationType.Bias ? 1 : _output; }
		}
		public double expected = 0;
		public ActivationType activator = ActivationType.Sigmoid;

		public enum ActivationType {
			Sigmoid,
			Tanh,
			Input,
			Bias
		}

		public Node(ActivationType activatorType) {
			activator = activatorType;
		}

		public void Activate(double inputTotal) {
			value = inputTotal;
			_output = activator == ActivationType.Sigmoid ? Sigmoid(value) : activator == ActivationType.Tanh ? Tanh(value) : value;
		}

		public double Derivative() {
			return activator == ActivationType.Sigmoid ? SigmoidDerivative(value) : TanhDerivative(value);
		}

		private static double Sigmoid(double value) {
			return 1 / (1 + Math.Exp(-1 * value));
		}

		private static double SigmoidDerivative(double value) {
			return Sigmoid(value) * (1 - Sigmoid(value));
		}

		private static double Tanh(double value) {
			return Math.Tanh(value);
		}

		private static double TanhDerivative(double value) {
			return 1 - Math.Pow(Math.Tanh(value), 2);
		}
	}

	class Weight {
		public double value;
		private double adjustment = 0;

		public Weight(double value) {
			this.value = value;
		}

		public void Train() {
			value += adjustment;
			adjustment = 0;
		}

		public void Adjust(double amount) {
			adjustment += amount;
		}
	}
}
