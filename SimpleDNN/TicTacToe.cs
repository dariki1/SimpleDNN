using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleDNN {
	class TicTacToe {

		public char[,] board = new char[3,3];
		bool xTurn = true;

		public TicTacToe(PictureBox showBox): this() {

			Bitmap b = new Bitmap(99,99);

			using (Graphics g = Graphics.FromImage(b)) {
				for (int x = 0; x < 3; x++) {
					for (int y = 0; y < 3; y++) {
						g.DrawRectangle(new Pen(Color.Black, 1), x*33, y*33, 33, 33);
					}
				}
			}				

			showBox.Image = b;

			showBox.MouseDown += (object o, MouseEventArgs m) => {
				int x = 3 * m.X / showBox.Width;
				int y = 3 * m.Y / showBox.Height;
				if (play(x, y)) {
					MessageBox.Show((xTurn ? "Y" : "X") + " wins!");
				}
				using (Graphics g = Graphics.FromImage(b)) {
					g.DrawString(""+board[x,y], new Font("Arial", 20), Brushes.Black, x*33, y*33);
				}

				showBox.Image = b;
			};
		}

		public TicTacToe() {
			for (int x = 0; x < board.GetLength(0); x++) {
				for (int y = 0; y < board.GetLength(1); y++) {
					board[x, y] = ' ';
				}
			}
		}

		public double[] getLinearOutput() {
			double[] ret = new double[board.GetLength(0) * board.GetLength(1)];
			for (int x = 0; x < board.GetLength(0); x++) {
				for (int y = 0; y < board.GetLength(1); y++) {
					ret[x / 3 + y] = board[x,y] == 'X' ? 1 : board[x,y] == 'Y' ? -1 : 0;
				}
			}
			return ret;
		}

		public bool play(int x, int y) {
			if (board[x, y] != ' ') {
				return false;
			}
			board[x, y] = xTurn ? 'X' : 'O';
			xTurn = !xTurn;
			return getWinner() != ' ';
		}

		public char getWinner() {
			bool winnerFound = false;

			// Check for verticle lines
			for (int x = 0; x < board.GetLength(0); x++) {
				winnerFound = true;
				for (int y = 1; y < board.GetLength(1); y++) {
					if (board[x, y] != board[x, 0]) {
						winnerFound = false;
						break;
					}
				}
				if (winnerFound && board[x, 0] != ' ') {
					return board[x, 0];
				}
			}

			// Check for horizontal lines
			for (int y = 0; y < board.GetLength(0); y++) {
				winnerFound = true;
				for (int x = 1; x < board.GetLength(1); x++) {
					if (board[x, y] != board[0, y]) {
						winnerFound = false;
						break;
					}
				}
				if (winnerFound && board[0, y] != ' ') {
					return board[0, y];
				}
			}

			// Check for diagonal lines
			if (board[0,0] != ' ' && board[0,0] == board[1,1] && board[0,0] == board[2,2] ) {
				return board[0, 0];
			}
			if (board[2, 0] != ' ' && board[2, 0] == board[1, 1] && board[2, 0] == board[0, 2]) {
				return board[2, 0];
			}
			return ' ';
		}
	}
}
