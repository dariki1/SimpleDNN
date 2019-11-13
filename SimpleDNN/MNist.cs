using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	public class MNist {
		const int labelOffset = 8;
		const int dataOffset = 16;
		const int imageWidth = 28;
		const int imageHeight = 28;

		public double[][] trainingData;
		public double[][] trainingLabels;
		public double[][] testingData;
		public double[][] testingLabels;

		public MNist() {
			byte[] rawTrainLabels = Properties.Resources.train_labels;
			byte[] rawTrainData = Properties.Resources.train_images;
			trainingData = new double[rawTrainLabels.Length - labelOffset][];			
			trainingLabels = new double[rawTrainLabels.Length - labelOffset][];

			for (int image = 0; image < trainingLabels.Length; image++) {
				trainingData[image] = new double[imageWidth*imageHeight];

				double[] label = new double[10];
				label[rawTrainLabels[image + labelOffset]] = 1;
				trainingLabels[image] = label;

				for (int x = 0; x < imageWidth; x++) {
					for (int y = 0; y < imageHeight; y++) {
						trainingData[image][x * imageWidth + y] = Math.Round(rawTrainData[image * 28 * 28 + x * 28 + y + dataOffset] / 255.0);
					}
				}
			}

			byte[] rawTestLabels = Properties.Resources.t10k_labels;
			byte[] rawTestData = Properties.Resources.t10k_images;
			testingData = new double[rawTestLabels.Length - labelOffset][];
			testingLabels = new double[rawTestLabels.Length - labelOffset][];

			for (int image = 0; image < testingData.Length; image++) {
				testingData[image] = new double[imageWidth * imageHeight];
				double[] label = new double[10];
				label[rawTestLabels[image + labelOffset]] = 1;
				testingLabels[image] = label;
				
				for (int x = 0; x < imageWidth; x++) {
					for (int y = 0; y < imageHeight; y++) {
						testingData[image][x*imageWidth+y] = Math.Round(rawTestData[image * 28 * 28 + x * 28 + y + dataOffset] / 255.0);
					}
				}
			}
		}

	}
}
