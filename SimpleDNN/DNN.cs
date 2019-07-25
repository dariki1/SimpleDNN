using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class DNN {
		private Node[][] nodes;
		private Weight[][][] weights;
		public DNN(int inputNumber, int outputNumber, int[] hiddenNumbers) {
			nodes = new Node[inputNumber+outputNumber+hiddenNumbers.Length][];

			nodes[0] = new Node[inputNumber];
			for (int inputCount = 0; inputCount < inputNumber; inputCount++) {
				nodes[0][inputCount] = new Node(Node.ActivationType.Input);
			}

			for (int hiddenLayerCount = 0; hiddenLayerCount < hiddenNumbers.Length; hiddenLayerCount++) {
				nodes[hiddenLayerCount + 1] = new Node[hiddenNumbers[hiddenLayerCount]];
				for (int hiddenNodeCount = 0; hiddenNodeCount < hiddenNumbers[hiddenLayerCount]; hiddenNodeCount++) {
					nodes[hiddenLayerCount + 1][hiddenNodeCount] = new Node(Node.ActivationType.Sigmoid);
				}
			}

			nodes[nodes.Length - 1] = new Node[outputNumber];
			for (int outputCount = 0; outputCount < outputNumber; outputCount++) {
				nodes[nodes.Length - 1][outputCount] = new Node(Node.ActivationType.Sigmoid);
			}

			Random rand = new Random();

			weights = new Weight[nodes.Length][][];
			for (int layer = 0; layer < this.weights.Length - 1; layer++) {
				weights[layer] = new Weight[nodes[layer].Length][];
				for (int inputIndex = 0; inputIndex < weights[layer].Length; inputIndex++) {
					weights[layer][inputIndex] = new Weight[nodes[layer+1].Length];
					for (int outputIndex = 0; outputIndex < weights[layer][inputIndex].Length; outputIndex++) {
						weights[layer][inputIndex][outputIndex] = new Weight(rand.NextDouble() * 2 - 1);						
					}
				}
			}
		}

		public DNN(int inputNumber, int outputNumber) : this(inputNumber, outputNumber, new int[0]) {
			
		}

		public double[] guess(double[] inputs) {
			for (int inputIndex = 0; inputIndex < nodes[0].Length; inputIndex++) {
				nodes[0][inputIndex].activate(inputs[inputIndex]);
			}

			for (int layer = 1; layer < nodes.Length; layer++) {
				for (int outputNode = 0; outputNode < nodes[layer].Length; outputNode++) {
					double nodeValue = 0;
					for (int inputNode = 0; inputNode < nodes[layer-1].Length; inputNode++) {
						nodeValue += nodes[layer - 1][inputNode].output * weights[layer][inputNode][outputNode].value;
					}
					nodes[layer][outputNode].activate(nodeValue);
				}
			}


			double[] ret = new double[nodes[nodes.Length-1].Length];
			for (int outputIndex = 0; outputIndex < ret.Length; outputIndex++) {
				ret[outputIndex] = nodes[nodes.Length - 1][outputIndex].output;
			}

			return ret;
		}
	}

	class Node {
		private double bias = 1;
		private double value;
		public double output;
		private ActivationType activator = ActivationType.Sigmoid;

		public enum ActivationType {
			Sigmoid,
			Tanh,
			Input
		}

		public Node(ActivationType activatorType) {
			activator = activatorType;
		}

		public void activate(double inputTotal) {
			value = inputTotal + bias;
			output = activator == ActivationType.Sigmoid ? sigmoid(value) : activator == ActivationType.Tanh ? tanh(value) : value;
		}

		private static double sigmoid(double value) {
			return 1 / (1 + Math.Exp(-1 * value));
		}

		private static double sigmoidDerivative(double value) {
			return sigmoid(value) * (1 - sigmoid(value));
		}

		private static double tanh(double value) {
			return Math.Tanh(value);
		}

		private static double tanhDerivative(double value) {
			return 1 - Math.Pow(Math.Tanh(value), 2);
		}
	}

	class Weight {
		public double value;
		public double adjustment;

		public Weight(double value) {
			this.value = value;
		}
	}
}
