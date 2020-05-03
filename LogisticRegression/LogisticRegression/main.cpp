// std include
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

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

	//lr.setThreshold(0.00139);
	//lr.setLearningRate(0.000002);

	auto theta = lr.train();

	{
		ifstream fin("test.txt");
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
