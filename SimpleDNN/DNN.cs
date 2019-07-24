using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class DNN {
		private Node[][] nodes;
		private Weight[] weights;
		public DNN(int inputNumber, int outputNumber, int[] hiddenNumber) : this(inputNumber, outputNumber) {
			
		}

		public DNN(int inputNumber, int outputNumber) {
			
		}
	}

	class Node {
		private double bias;
		private double value;
		private double output;
		private ActivationType activator = ActivationType.Sigmoid;

		public enum ActivationType {
			Sigmoid,
			Tanh
		}

		public Node(ActivationType activatorType) {
			bias = 1;
			value = 0;
			output = 0;
			activator = activatorType;
		}

		public double Activate(int inputTotal) {
			value = inputTotal;
			output = activator == ActivationType.Sigmoid ? sigmoid(value) : tanh(value);
			return 0.0;
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
		public int inputNodeIndex { get; }
		public int outputNodeIndex { get; }
		public Weight(int inputIndex, int outputIndex) {
			inputNodeIndex = inputIndex;
			outputNodeIndex = outputIndex;
		}
	}
}
