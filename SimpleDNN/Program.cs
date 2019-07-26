using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class Program {
		static Random rand = new Random();
		static void Main(string[] args) {
			DNN net = new DNN(784, 10, new int[] { 20, 20 });
			MNistReader data = new MNistReader();

			for (int i = 0; i < 1; i++) {
				train(net, data.training);
				test(net, data.testing);
			}			

			Console.ReadKey();
		}

		public static void train(DNN net, double[][][] data) {
			double[] expectedResults = new double[10];
			int index = 0;
			for (int image = 0; image < data.Length; image++) {
				expectedResults[index] = 0;
				index = (int)data[image][0][0];
				expectedResults[index] = 1;

				net.Train(data[image][1], expectedResults, index % 10 == 9);

				if (image % 1000 == 999) {
					Console.WriteLine("Completed " + image + " training sets");
				}
			}
		}

		public static void test(DNN net, double[][][] data) {
			int correct = 0;
			for (int image = 0; image < data.Length; image++) {
				double[] guess = net.Guess(data[image][1]);
				int hIndex = 0;
				for (int i = 1; i < guess.Length; i++) {
					if (guess[i] > guess[hIndex]) {
						hIndex = i;
					}
				}
				if (hIndex == (int)data[image][0][0]) {
					++correct;
				}

				if (image % 1000 == 999) {
					Console.WriteLine("Completed " + image + " testing sets");
				}
			}

			Console.WriteLine("Completed testing with " + (100*correct/data.Length) + "% accuracy");
		}
	}
}
