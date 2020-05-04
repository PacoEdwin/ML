// std include
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <sstream>

// project includes
#include "logisticregression.h"

using namespace std;

int main()
{
	LogisticRegression lr;

	{
		ifstream fin("train.txt");
		if (!fin.is_open())
		{
			cout << "Couldn't open file." << endl;
			return 1;
		}

		double val;
		vector<double> d(3, 1);
		while (fin >> val >> d[1] >> d[2])
		{
			TrainingExample tr(d, val);
			lr.addExample(tr);
		}

		fin.close();
	}

	lr.setThreshold(0.005);
	lr.setLearningRate(0.000001);

	// train model
	auto theta = lr.train();

	// Test
	{
		ifstream fin("test2.txt");
		if (!fin.is_open())
		{
			cout << "Couldn't open file." << endl;
			return 1;
		}

		double ans;
		int correct = 0, size = 0;
		vector<double> d(3, 1);
		while (fin)
		{
			string s;
			if (!getline(fin, s))
				break;

			istringstream ss(s);
			vector<string> record;

			while (ss)
			{
				string s;
				if (!getline(ss, s, ','))
					break;

				record.push_back(s);
			}

			d[1] = stoi(record[0]);
			d[2] = stoi(record[1]);
			ans = stoi(record[2]);

			TestExample t(d, ans);
			if (lr.test(t))
				correct++;

			size++;
		}

		fin.close();

		cout << "correct: " << correct << " size: " << size << endl;
		cout << "accuracy: " << (double)100 * correct / size << endl;
		cout << endl;
	}

	// Test
	{
		ifstream fin("test1.txt");
		if (!fin.is_open())
		{
			cout << "Couldn't open file." << endl;
			return 1;
		}

		double val;
		vector<double> d(3, 1);
		
		int correct = 0;
		int size = 0;

		while (fin >> val >> d[1] >> d[2])
		{
			TestExample t(d, val);
			if (lr.test(t))
				correct++;

			size++;
		}
		fin.close();

		cout << "correct: " << correct << " size: " << size << endl;
		cout << "accuracy: " << (double)100 * correct / size << endl;
	}

	system("pause");
}
