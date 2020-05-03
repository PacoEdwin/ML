#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "gradient.h"
#include <math.h>
#include <algorithm>

using namespace std;

void log(const vector<double>& theta, const vector<double>& newTheta, const vector<double>& minDelta)
{
	{
		cout << "old theta: ";
		for (int i = 0; i < theta.size(); i++)
			cout << theta[i] << " ";
		cout << endl;
	}

	{
		cout << "new theta: ";
		for (int i = 0; i < theta.size(); i++)
			cout << newTheta[i] << " ";
		cout << endl;
	}

	{
		cout << "delta: ";
		for (int i = 0; i < theta.size(); i++)
			cout << abs(theta[i] - newTheta[i]) << " ";
		cout << endl;
	}


	{
		cout << "Min delta: ";
		for (int i = 0; i < theta.size(); i++)
			cout << minDelta[i] << " ";
		cout << endl;
	}
}

double h(const vector<double>& theta, const vector<double>& trainData)
{
	double powerValue = 0;
	for (int i = 0; i < theta.size(); i++)
		powerValue -= theta[i] * trainData[i];

	double result = 1 / (1 + exp(powerValue));

	return result;
}

double derivativeValue(int j, const vector<double>& theta, const vector<double>& y, const vector<vector<double>>& trainData)
{
	double value = 0;

	for (int i = 0; i < trainData.size(); i++)
		value += (h(theta, trainData[i]) - y[i])*trainData[i][j];

	return value;
}

vector<double> train(const double& learningRate, const vector<double>& theta, const vector<double>& y, const vector<vector<double>>& trainData)
{
	auto newTheta = theta;

	for (int i = 0; i < theta.size(); i++)
		newTheta[i] -= learningRate * derivativeValue(i, theta, y, trainData);

	return newTheta;
}

int main()
{
	vector<vector<double>> trainData;
	vector<double> theta(3, 0);
	vector<double> y;

	{
		ifstream fin("train.txt");
		if (!fin.is_open())
		{
			cout << "Couldn't open file." << endl;
			return 1;
		}

		vector<double> d(3, 1);
		double val;
		while (fin >> val >> d[1] >> d[2])
		{
			trainData.push_back(d);
			y.push_back(val);
		}
		fin.close();
	}

	double threshhold = 0.00139;
	double learningRate = 0.000002;

	int count = 0;
	int showLogIf = 300;
	int minDeltaAbort = 4;
	bool converged = false;

	vector<double> minDelta(3, numeric_limits<double>::max());

	while (!converged)
	{
		auto newTheta = train(learningRate, theta, y, trainData);
		
		bool check = true;
		for (int j = 0; j < theta.size(); j++)
		{
			minDelta[j] = std::min(minDelta[j], abs(newTheta[j] - theta[j]));

			if (abs(newTheta[j] - theta[j]) >= threshhold)
			{
				check = false;
				break;
			}
		}

		if (showLogIf == count)
		{
			log(theta, newTheta, minDelta);
			count = 0;
		}

		theta = newTheta;

		if (check == true)
		{
			converged = true;
			break;
		}

		count++;
	}

	cout << "theta: ";
	for (auto th : theta)
		cout << th << " ";
	cout << endl;

	vector<vector<double>> testData;

	{
		ifstream fin("train.txt");
		if (!fin.is_open())
		{
			cout << "Couldn't open file." << endl;
			return 1;
		}

		vector<double> d(3, 1);
		double val;
		int correct = 0;
		int size = 0;
		while (fin >> val >> d[1] >> d[2])
		{
			size++;
			testData.push_back(d);
			int ans = h(theta, d) >= 0.5 ? 1 : 0;
			//if (ans)
			//	cout << "Actual: " << val << " " << 1 << endl;
			//else
			//	cout << "Actual: " << val << " " << 0 << endl;

			if (ans == val)
				correct++;
		}
		cout << "correct: " << correct << " size: " << size << endl;
		cout << "accuracy: " << (double)100 * correct / size << endl;
		fin.close();
	}

	system("pause");
}
