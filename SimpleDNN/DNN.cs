using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleDNN {
	public class DNN {
		private Node[][] nodes;
		private Weight[][][] weights;
		public double learningRate = 0.1;

		public DNN(int inputNumber, int outputNumber, int[] hiddenNumbers) {
			// Initialise node layers, one input layer, however many hidden layers and one output layer
			nodes = new Node[2+hiddenNumbers.Length][];

			// Initialise the nodes in input layer, including a bias node
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

		public DNN(int inputNumber, int outputNumber) : this(inputNumber, outputNumber, new int[0]) {}

		public void SaveToFile(string path) {
			string sWeights = "";
			for (int layer = 0; layer < weights.Length; layer++) {
				sWeights += "{";
				for (int input = 0; input < weights[layer].Length; input++) {
					sWeights += "{";
					for (int output = 0; output < weights[layer][input].Length; output++) {
						sWeights += "{" + weights[layer][input][output].value + "}";
					}
					sWeights += "}";
				}
				sWeights += "}";
			}
			File.WriteAllText(path, sWeights);
		}

		/**
		TODO: Make this work
		**/
		public void LoadFromFile(string path) {
			string sWeights = File.ReadAllText(path);
		}

		public double[] Guess(double[] inputs) {
			// Set the values for the input nodes
			for (int inputIndex = 0; inputIndex < nodes[0].Length - 1; inputIndex++) {
				nodes[0][inputIndex].Activate(inputs[inputIndex]);
			}

			// Pull the values from the previous layer forward to the current layer
			for (int layer = 1; layer < nodes.Length; layer++) {
				int nodeCount = nodes[layer].Length;
				using (ManualResetEvent resetEvent = new ManualResetEvent(false)) {
					for (int outputNode = 0; outputNode < nodes[layer].Length; outputNode++) {
						// For each node in the current layer, make its value equal to the sum of the output of each node in the previous layer multiplied by the appropriate weight
						ThreadPool.QueueUserWorkItem(new WaitCallback(input => {
							Tuple<int, int> val = (Tuple<int, int>)input;
							pullValue(val.Item1, val.Item2);
							if (Interlocked.Decrement(ref nodeCount) == 0) {
								resetEvent.Set();
							}
						}), new Tuple<int, int>(layer, outputNode));
					}
					// Wait for all the nodes to finish updating
					resetEvent.WaitOne();
				}
			}

			
			// Get the return values from the final layer
			double[] ret = new double[nodes[nodes.Length-1].Length];
			for (int outputIndex = 0; outputIndex < ret.Length; outputIndex++) {
				ret[outputIndex] = nodes[nodes.Length - 1][outputIndex].output;
			}

			// Return the return values
			return ret;
		}

		private void pullValue(int layer, int outputNode) {
			double nodeValue = 0;
			// Sum the outputs of each node in the previous layer through the appropriate weights
			for (int inputNode = 0; inputNode < nodes[layer - 1].Length; inputNode++) {
				nodeValue += nodes[layer - 1][inputNode].output * weights[layer - 1][inputNode][outputNode].value;
			}
			// Set the given node to it's new value
			nodes[layer][outputNode].Activate(nodeValue);
		}


		public double BulkTest(double[][] inputs, double[][] outputs) {
			// The number of guesses that were correct
			int correct = 0;
			for (int input = 0; input < inputs.Length; input++) {
				// Make a guess on the current input set
				double[] guess = Guess(inputs[input]);
				// Set to true if there is a value mismatch
				bool wrong = false;
				for (int i = 1; i < guess.Length; i++) {
					if (Math.Round(guess[i]) != guess[i]) {
						wrong = true;
						break;
					}
				}
				// If every value was correct, increment correct counter
				if (!wrong) {
					correct++;
				}
			}
			// Return the number of correct guesses as a percentage (0.0 to 1.0)
			return ((double)correct)/((double)outputs.Length);
		}

		public void BulkTrain(double[][] inputs, double[][] expectedOutputs, int interval) {
			// For each input, train on it, execute the changes every <interval> inputs
			for (int input = 0; input < inputs.Length-1; input++) {
				Train(inputs[input], expectedOutputs[input], input % interval == 0);
			}
			// Train on the last input and always execute
			Train(inputs[inputs.Length-1], expectedOutputs[inputs.Length-1], true);
		}

		public void Train(double[] inputs, double[] expectedOutputs, bool execute) {
			// Make a guess
			double[] guess = Guess(inputs);

			// Set the expected outputs for the output nodes
			for (int outputIndex = 0; outputIndex < nodes[nodes.Length-1].Length; outputIndex++) {
				nodes[nodes.Length - 1][outputIndex].expected = expectedOutputs[outputIndex];
			}

			// Go through each layer backwards
			for (int layer = nodes.Length - 1; layer > 0; layer--) {
				// Get adjustments for all nodes except the Bias nodes, which is not present in the last layer
				for (int outputNode = 0; outputNode < nodes[layer].Length - (layer == nodes.Length - 1 ? 0 : 1); outputNode++) {
					// Calculate the error of the node outputs
					double error = nodes[layer][outputNode].expected - nodes[layer][outputNode].output;
					// Calculate the amount of change needed for the exact value, multiply it by learningRate
					double gradient = nodes[layer][outputNode].Derivative() * error * learningRate;

					// For each input node
					for (int inputNode = 0; inputNode < nodes[layer-1].Length; inputNode++) {
						// If just starting on this layer, reset the expectation of the nodes of the previous layer
						if (outputNode == 0) {
							nodes[layer - 1][inputNode].expected = 0;
						}

						// Set the amount the weight needs to change, but don't change it yet
						weights[layer - 1][inputNode][outputNode].Adjust(gradient * nodes[layer - 1][inputNode].output);
						// Change the expected output of the input node
						nodes[layer - 1][inputNode].expected += weights[layer - 1][inputNode][outputNode].value * error;

						if (execute) {
							// Change the value of the weight
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
