using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleDNN {
	class Program {
		static Random rand = new Random();
		static DNN net = new DNN(784, 10, new int[] { 20, 5 });
		static MNist data = new MNist();
		static OutputForm form = new OutputForm();
		static bool drawing = false;
		static PictureBox tttInput = new PictureBox();

		static void Main(string[] args) {
			ThreadStart formRef = new ThreadStart(startForm);
			Thread formThread = new Thread(formRef);
			formThread.Start();

			for (int i = 0; i < 10; i++) {
				mNistTrain(net, data.training);
				mNistTest(net, data.testing);
			}			
		}

		static void tTrain(string play, char winner) {

		}

		static double[] tGuess(double[] boardState) {
			double[] guess = net.Guess(boardState);
			Array.Sort(guess);
			return guess;
		}

		static void initMNistInput(PictureBox pB) {
			pB.SizeMode = PictureBoxSizeMode.Zoom;
			pB.Dock = DockStyle.Fill;

			Bitmap b = new Bitmap(28, 28);

			pB.MouseDown += (object o, MouseEventArgs m) => {
				if (m.Button == MouseButtons.Right) {
					b = new Bitmap(28, 28);
					pB.Image = b;
				} else {
					drawing = true;
				}
			};

			pB.MouseMove += (object o, MouseEventArgs m) => {
				if (!drawing || m.X < 0 || m.X > pB.Width || m.Y < 0 || m.Y > pB.Height) {
					drawing = false;
					return;
				}
				b.SetPixel(28 * (m.X - 1) / pB.Width, 28 * (m.Y - 1) / pB.Height, Color.Black);

				pB.Image = b;
			};

			pB.MouseUp += (object o, MouseEventArgs m) => {
				if (drawing) {
					drawing = false;
					MNistGuess(new Bitmap(pB.Image));
				}
			};
		}

		static void MNistGuess(Bitmap b) {
			drawing = false;

			double[] input = new double[28 * 28];

			for (int y = 0; y < 28; y++) {
				for (int x = 0; x < 28; x++) {
					input[x * 28 + y] = (b.GetPixel(y, x).A) / 255.0;
				}
			}
			double[] guess = net.Guess(input);
			int hIndex = 0;
			for (int i = 1; i < guess.Length; i++) {
				if (guess[i] > guess[hIndex]) {
					hIndex = i;
				}
			}
			Console.WriteLine(hIndex);
		}


		public static void startForm() {
			PictureBox mnistInput = new PictureBox();
			initMNistInput(mnistInput);
			form.Controls.Add(mnistInput);
			form.Resize += (object send, EventArgs e) => {
				mnistInput.Size = form.Size;
			};

			/*TicTacToe tTT = new TicTacToe(tttInput);

			tttInput.SizeMode = PictureBoxSizeMode.Zoom;
			tttInput.Dock = DockStyle.Fill;

			form.Controls.Add(tttInput);*/

			form.ShowDialog();
		}

		public static void mNistTrain(DNN net, double[][][] data) {
			double[] expectedResults = new double[10];
			int index = 0;			
			for (int image = 0; image < data.Length; image++) {
				expectedResults[index] = 0;
				index = (int)data[image][0][0];
				expectedResults[index] = 1;

				net.Train(data[image][1], expectedResults, index % 1 == 0);

				if (image % 1000 == 999) {
					Console.WriteLine("Completed " + image + " training sets");
				}
			}
		}

		public static void mNistTest(DNN net, double[][][] data) {
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
