using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class fDNN {	
		double[][,] weights;
		Func<double, double> activation;
		Func<double, double> activationDeriver;

		public fDNN(int inputs, int outputs) : this(inputs, outputs, new int[] { }, sigmoid, sigmoidDerivative) { }

		public fDNN(int inputs, int outputs, int[] hidden) : this(inputs, outputs, hidden, sigmoid, sigmoidDerivative) { }

		public fDNN(int inputs, int outputs, int[] hidden, Func<double, double> activationFunction, Func<double, double> activationFunctionDerivative) {
			activation = activationFunction;
			activationDeriver = activationFunctionDerivative;

			Random rdm = new Random();
			weights = new double[1+hidden.Length][,];

			weights[0] = new double[inputs, hidden.Length > 0 ? hidden[0] : outputs];
			for (int layer = 1; layer < hidden.Length; layer++) {
				weights[layer] = new double[weights[layer - 1].GetLength(1), hidden[layer]];
			}
			if (hidden.Length > 0) {
				weights[weights.Length - 1] = new double[weights[weights.Length - 2].GetLength(1), outputs];
			}

			for (int layer = 0; layer < weights.Length; layer++) {
				for (int input = 0; input < weights[layer].GetLength(0); input++) {
					for (int output = 0; output < weights[layer].GetLength(1); output++) {
						weights[layer][input, output] = rdm.NextDouble();
					}
				}
			}
		}

		public double[] guess(double[] inputs) {
			return runNet(inputs, this.weights, this.activation);
		}

		public void train(double[][] inputs, double[][] outputs, int changeWeightFrequency) {
			if (inputs.Length != outputs.Length) {
				throw new Exception("Number of inputs and number of outputs should be the same");
			}
			if (changeWeightFrequency < 1) {
				throw new Exception("changeWeightFrequency must be at least 1");
			}
			double[][][,] wChange = new double[changeWeightFrequency][][,];
			int top = (inputs.Length - 1) / changeWeightFrequency;
			for (int batch = 0; batch < top; batch++) {
				for (uint item = 0; item < wChange.Length; item++) {
					wChange[item] = trainWeights(inputs[batch*changeWeightFrequency+item], this.weights, outputs[batch*changeWeightFrequency+item], this.activation, this.activationDeriver);
				}
				
				for (int item = 0; item < wChange.Length; item++) {
					for (int layer = 0; layer < this.weights.Length; layer++) {
						for (int input = 0; input < this.weights[layer].GetLength(0); input++) {
							for (int output = 0; output < this.weights[layer].GetLength(1); output++) {
								this.weights[layer][input, output] += wChange[item][layer][input, output];
							}
						}
					}
				}
			}

			// Trains the left over items, this ensures the final training items actually change the weights
			wChange = new double[inputs.Length % changeWeightFrequency][][,];
			for (int item = 0; item < inputs.Length%changeWeightFrequency; item++) {
				wChange[item] = trainWeights(inputs[top * changeWeightFrequency + item], this.weights, outputs[top * changeWeightFrequency + item], this.activation, this.activationDeriver);
			}

			for (int item = 0; item < wChange.Length; item++) {
				for (int layer = 0; layer < this.weights.Length; layer++) {
					for (int input = 0; input < this.weights[layer].GetLength(0); input++) {
						for (int output = 0; output < this.weights[layer].GetLength(1); output++) {
							this.weights[layer][input, output] += wChange[item][layer][input, output];
						}
					}
				}
			}
		}

		public static double[] runNet(double[] input, double[][,] weights, Func<double, double> activationFunction) {
			double[] nodes = feedForward(input, weights[0], activationFunction);
			for (int layer = 1; layer < weights.Length; layer++) {
				nodes = feedForward(nodes, weights[layer], activationFunction);
			}
			return activate(activationFunction, nodes);
		}

		public static double[][,] trainWeights(double[] input, double[][,] weights, double[] expectedOutput, Func<double, double> activationFunction, Func<double, double> activationDerivative) {
			double[][,] ret = new double[weights.Length][,];			
			double[] thisExpectation = expectedOutput;

			double[][] nodes = new double[weights.Length+1][];
			nodes[0] = input;
			for (int layer = 1; layer < nodes.Length; layer++) {
				nodes[layer] = feedForward(nodes[layer-1], weights[layer-1], activationFunction);
			}
			
			for (int layer = weights.Length - 1; layer >= 0; layer--) {
				double[] err = error(activate(activationFunction, nodes[layer+1]), thisExpectation);
				double[] grad = gradient(activate(activationDerivative, nodes[layer + 1]), err, 0.1);

				thisExpectation = nodeAdjustment(weights[layer], err);

				ret[layer] = weightAdjustment(activate(activationFunction, nodes[layer]), grad);				
			}

			/*Console.WriteLine("v*********************************************v");
			for (int layer = 0; layer < nodes.Length; layer++) {
				for (int node = 0; node < nodes[layer].Length; node++) {
					Console.Write(nodes[layer][node]+",");
				}
				Console.WriteLine();
			}
			Console.WriteLine("^*********************************************^");*/

			return ret;
		}

		public static double sigmoid(double value) {
			return 1 / (1 + Math.Exp(-value));
		}

		public static double deriveSigmoid(double value) {
			return value * (1 - value);
		}

		public static double sigmoidDerivative(double value) {
			// Equivalent to return Sigmoid(value) * (1 - Sigmoid(value));
			return (1 / (1 + Math.Exp(-value))) * (1 - (1 / (1 + Math.Exp(-value))));			
		}

		/**
		 * Takes the inputs, weights and desired activation function and calculates the output
		 */
		private static double[] feedForward(double[] inputNodes, double[,] weights, Func<double, double> activationFunction) {
			if (inputNodes.Length != weights.GetLength(0)) {
				throw new Exception("inputNodes and first dimension of weights must be the same length");
			}
			double[] ret = new double[weights.GetLength(1)];
			for (int input = 0; input < inputNodes.Length; input++) {
				for (int output = 0; output < weights.GetLength(1); output++) {
					ret[output] += activationFunction(inputNodes[input]) * weights[input,output];
				}
			}
			return ret;
		}

		/**
		 * 
		 */
		private static double[] backProp(double[] outputNodes, double[,] weights) {
			if (outputNodes.Length != weights.GetLength(1)) {
				throw new Exception("outputNodes and second dimension of weights must be the same length");
			}
			double[] ret = new double[weights.GetLength(0)];
			for (int output = 0; output < outputNodes.Length; output++) {
				for (int input = 0; input < weights.GetLength(0); input++) {
					ret[input] += outputNodes[output] * weights[input,output];
				}
			}
			return ret;
		}

		/**
		 * Calculates the error between the output values and the expected outptus
		 */
		private static double[] error(double[] outputValues, double[] expectedOutputs) {
			if (outputValues.Length != expectedOutputs.Length) {
				throw new Exception("outputValues and expectedOutputs must be the same length");
			}
			double[] ret = new double[outputValues.Length];
			for (int i = 0; i < outputValues.Length; i++) {
				ret[i] = expectedOutputs[i] - outputValues[i];
			}
			return ret;
		}

		/**
		 * Applies the activation function to the values of the nodes
		 */
		private static double[] activate(Func<double, double> activationFunction, double[] nodes) {
			double[] ret = new double[nodes.Length];

			for (int i = 0; i < nodes.Length; i++) {
				ret[i] = activationFunction(nodes[i]);
			}

			return ret;
		}

		/**
		 * Applies the derived activation function to the values of the nodes
		 */
		private static double[] derive(Func<double, double> derivationFunction, double[] nodes) {
			double[] ret = new double[nodes.Length];
			for (int i = 0; i < nodes.Length; i++) {
				ret[i] = derivationFunction(nodes[i]);
			}
			return ret;
		}

		/**
		 * Calculates the gradient of the error
		 */
		private static double[] gradient(double[] derivedOutput, double[] error, double learningRate) {
			if (derivedOutput.Length != error.Length) {
				throw new Exception("derivedOutput and error should have the same length");
			}
			double[] ret = new double[derivedOutput.Length];
			for (int i = 0; i < derivedOutput.Length; i++) {
				ret[i] = derivedOutput[i] * error[i] * learningRate;
			}
			return ret;
		}

		/**
		 * Calculates the gradient of the error
		 */
		private static double[] gradient(double[] output, double[] error, double learningRate, Func<double, double> activationDerivative) {
			return gradient(derive(activationDerivative, output), error, learningRate);
		}

		/**
		 * Adjusts weights to provide better accuracy
		 */
		private static double[,] weightAdjustment(double[] activatedInputs, double[] gradients) {			
			double[,] ret = new double[activatedInputs.Length, gradients.Length];

			for (int input = 0; input < activatedInputs.Length; input++) {
				for (int output = 0; output < gradients.Length; output++) {
					ret[input, output] = gradients[output] * activatedInputs[input];
				}
			}

			return ret;
		}

		/**
		 * Returns the expected value of the input nodes
		 */
		private static double[] nodeAdjustment(double[,] weights, double[] error) {
			if (weights.GetLength(1) != error.Length) {
				throw new Exception("error and second dimension of weights should be equal");
			}
			double[] ret = new double[weights.GetLength(0)];

			for (int input = 0; input < weights.GetLength(0); input++) {
				for (int output = 0; output < weights.GetLength(1); output++) {
					ret[input] += weights[input, output] * error[output];
				}
			}

			return ret;
		}

	}
}
