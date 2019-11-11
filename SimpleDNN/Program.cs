using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleDNN {
	class Program {
		static Random rand;
		static DNN net;
		static MNist data;
		static OutputForm form;
		static bool drawing;

		static void Main(string[] args) {
			rand = new Random();
			net = new DNN(784, 10, new int[] {20, 20});
			data = new MNist();
			form = new OutputForm();
			drawing = false;

			/*ThreadStart formRef = new ThreadStart(startForm);
			Thread formThread = new Thread(formRef);
			formThread.Start();*/

			Stopwatch s = Stopwatch.StartNew();			
			for (int i = 0; i < 1; i++) {
				mNistTrain(net, data.trainingData, data.trainingLabels);
				mNistTest(net, data.testingData, data.testingLabels);
			}
			s.Stop();
			Console.WriteLine("Complete in " + s.ElapsedMilliseconds);
			//net.SaveToFile(@"C:\Projects\Visual Studio\C#\SimpleDNN\" + "Weights.txt");
			Console.ReadKey();
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

			form.ShowDialog();
		}

		public static void mNistTrain(DNN net, double[][] data, double[][] labels) {
			for (int image = 0; image < data.Length - 1; image++) {
				net.Train(data[image], labels[image], image % 1 == 0);
			}
			net.Train(data[data.Length-1], labels[data.Length-1], true);
		}

		public static void mNistTest(DNN net, double[][] data, double[][] labels) {
			int correct = 0;
			for (int image = 0; image < data.Length; image++) {
				double[] guess = net.Guess(data[image]);
				int hIndex = 0;
				for (int i = 1; i < guess.Length; i++) {
					if (guess[i] > guess[hIndex]) {
						hIndex = i;
					}
				}
				if ((int)labels[image][hIndex] == 1) {
					correct++;
				}

				if (image % 1000 == 999) {
					Console.WriteLine("Completed " + image + " testing sets");
				}
			}

			Console.WriteLine("Completed testing with " + (100*correct/data.Length) + "% accuracy");
		}
	}
}
