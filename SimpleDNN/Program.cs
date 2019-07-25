using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class Program {
		static Random rand = new Random();
		static void Main(string[] args) {
			DNN temp = new DNN(2, 1, new int[] { 4 });
			double[][] data = getData();

			for (int i = 0; i < 50000; i++) {
				data = getData();
				data[0][0] = Math.Round(data[0][0]);
				data[0][1] = Math.Round(data[0][1]);
				temp.Train(data[0], data[1], i % 10 == 9);
			}

			int accurate = 0;
			double precise = 0;
			const int testNum = 1000;

			for (int i = 0; i < testNum; i++) {
				data = getData();
				double[] guess = temp.Guess(data[0]);
				if (Math.Round(guess[0]) == data[1][0]) {
					++accurate;
					precise += Math.Abs(data[1][0] - guess[0]);
				}
			}

			Console.WriteLine(100.0*accurate/testNum + "% accurate, " + 100*precise/accurate + "% precise");

			int[,] outputs = new int[20,20];
			for (double x = 0; x < outputs.GetLength(0); x++) {
				for (double y = 0; y < outputs.GetLength(1); y++) {
					outputs[(int)x, (int)y] = (int)Math.Round(temp.Guess(new double[] { x / outputs.GetLength(0), y / outputs.GetLength(1) })[0]);
					Console.Write(outputs[(int)x,(int)y] + ",");
				}
				Console.WriteLine();
			}			


			/*for (int e = 0; e < 10; e++) {
				data = getData();

				Console.WriteLine("Input; " + data[0][0] + ", " + data[0][1] + ", Expected Output; " + data[1][0]);

				Console.WriteLine(temp.Guess(data[0])[0]);

				for (int i = 0; i < 100; i++) {
					temp.Train(data[0], data[1], true);
				}

				Console.WriteLine(temp.Guess(data[0])[0]);
			}*/

			Console.ReadKey();
		}

		static double[][] getData() {
			double[] input = new double[] { rand.NextDouble(), rand.NextDouble() };
			double[] output = new double[] { Math.Round(input[0]) == Math.Round(input[1]) ? 0 : 1};
			return new double[][] { input, output };
		}
	}
}
