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
		static bool doTimer;

		static void Main(string[] args) {
			rand = new Random();
			/*net = new DNN(784, 10, new int[] {20, 20});
			*/
			data = new MNist();
			/*
			form = new OutputForm();
			drawing = false;
			doTimer = true;

			while (true) {
				runCommand(Console.ReadLine());
			}*/

			/*
			 * Input: 1
			 * Hidden: 0
			 * Output: 2
			 */

			
			fDNN f = new fDNN(784,10, new int[] { 20, 20 });

			Console.WriteLine("Training");
			f.train(data.trainingData, data.trainingLabels,10);
			Console.WriteLine("Testing");
			
			int correct = 0;
			for (int i = 0; i < data.testingData.Length; i++) {
				double[] g = f.guess(data.testingData[i]);
				int hInd = 0;
				for (int e = 1; e < g.Length; e++) {
					if (g[e] > g[hInd]) {
						hInd = e;
					}
				}
				if (data.testingLabels[i][hInd] == 1) {
					correct++;
				}
			}
			Console.WriteLine("Complete with " + (((double)correct/(double)data.testingLabels.Length) * 100) + "% accuracy");			

			Console.ReadKey();
		}

		static void runCommand(string command) {
			string[] split = command.ToLower().Split();
			Stopwatch s = new Stopwatch();
			switch (split[0]) {
				case ("help"):
					Console.WriteLine("Commands\n\thelp: shows this information.\n\ttrain <num>: Trains the network for <num> iterations.\n\ttest: Tests the network and outputs accuracy.\n\tsave <file>: saves the current network as <file>\n\tinput: Adds a form for custom data input\n\ttimer <state=!timer>: Sets timer use to state, with a default value of the opposite of what it currently is.\n\tlearnrate <rate>: Sets the DNN learning rate to rate");
					break;
				case ("train"):
					Console.WriteLine("Starting training");
					int end = 1;
					if (split.Length > 1) {
						Int32.TryParse(split[1], out end);
					}
					for (int i = 0; i < end; i++) {
						Console.WriteLine("Training iteration " + (i+1));
						if (doTimer) {
							s = Stopwatch.StartNew();
						}
						net.BulkTrain(data.trainingData, data.trainingLabels, 10);
						if (doTimer) {
							s.Stop();
							Console.WriteLine("Completed training in " + s.ElapsedMilliseconds);
						} else {
							Console.WriteLine("Completed training");
						}
					}
					break;
				case ("test"):
					Console.WriteLine("Starting testing");
					if (doTimer) {
						s = Stopwatch.StartNew();
					}
					Console.WriteLine(net.BulkTest(data.testingData, data.testingLabels, 0.1) * 100 + "% accurate");
					if (doTimer) {
						s.Stop();
						Console.WriteLine("Completed testing in " + s.ElapsedMilliseconds);
					} else {
						Console.WriteLine("Completed testing");
					}
					
					break;
				case ("save"):
					if (split.Length > 1) {
						net.SaveToFile(@"C:\Projects\Visual Studio\C#\SimpleDNN\" + split[1]);
						Console.WriteLine("File saved");
					} else {
						Console.WriteLine("Please define file name");
					}
					break;
				case ("input"):
					ThreadStart formRef = new ThreadStart(startForm);
					Thread formThread = new Thread(formRef);
					formThread.Start();
					break;
				case ("timer"):
					if (split.Length > 1) {
						Boolean.TryParse(split[1], out doTimer);
					} else {
						doTimer = !doTimer;
					}
					Console.WriteLine("Timer is now " + doTimer);
					break;
				case ("learnrate"):
					if (split.Length > 0) {
						Double.TryParse(split[1], out net.learningRate);
						Console.WriteLine("Learning rate set to " + split[1]);
					} else {
						Console.WriteLine("Please set a value");
					}
					break;
				default:
					Console.WriteLine("Unknown command, try 'help'");
					break;
			}
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
	}
}
