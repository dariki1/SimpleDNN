using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleDNN {
	class MNistReader {
		const int labelOffset = 8;
		const int dataOffset = 16;
		const int imageWidth = 28;
		const int imageHeight = 28;

		public double[][][] training;
		public double[][][] testing;

		public MNistReader() {

			byte[] rawTrainLabels = Properties.Resources.train_labels;
			byte[] rawTrainData = Properties.Resources.train_images;
			training = new double[rawTrainLabels.Length - labelOffset][][];

			for (int image = 0; image < training.Length; image++) {
				training[image] = new double[2][];
				training[image][0] = new double[] { rawTrainLabels[image + labelOffset] };
				training[image][1] = new double[imageWidth * imageHeight];
				for (int x = 0; x < imageWidth; x++) {
					for (int y = 0; y < imageHeight; y++) {
						training[image][1][x * imageWidth + y] = rawTrainData[image * 28 * 28 + x * 28 + y + dataOffset] / 255.0;
					}
				}
			}

			byte[] rawTestLabels = Properties.Resources.t10k_labels;
			byte[] rawTestData = Properties.Resources.t10k_images;
			testing = new double[rawTestLabels.Length - labelOffset][][];

			for (int image = 0; image < testing.Length; image++) {
				testing[image] = new double[2][];
				testing[image][0] = new double[] { rawTestLabels[image + labelOffset] };
				testing[image][1] = new double[imageWidth * imageHeight];
				for (int x = 0; x < imageWidth; x++) {
					for (int y = 0; y < imageHeight; y++) {
						testing[image][1][x*imageWidth+y] = rawTestData[image * 28 * 28 + x * 28 + y + dataOffset] / 255.0;
					}
				}
			}
		}

	}
}
