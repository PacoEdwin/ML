#include "logisticregression.h"

#include <iostream>
#include <math.h>
#include <algorithm>

LogisticRegression::LogisticRegression() :
	m_examples(0),
	m_features(0),
	m_learningRate(0.000002),
	m_threshold(0.00139)
{}

LogisticRegression::LogisticRegression(const std::vector<TrainingExample>& trainingData) :
	m_trainingData(trainingData)
{
	m_examples = m_trainingData.size();
	if (m_examples > 0)
		m_features = m_trainingData[0].getFeatures().size();

	m_theta = std::vector<double>(m_features, 0/*30.0*/);
}

void LogisticRegression::addExample(const TrainingExample& value)
{
	m_trainingData.push_back(value);

	m_examples++;
	if (m_examples == 1)
	{
		m_features = m_trainingData[0].getFeatures().size();
		m_theta = std::vector<double>(m_features, 0);
	}
}

void LogisticRegression::setThreshold(const double& value)
{
	m_threshold = value;
}

void LogisticRegression::setLearningRate(const double& value)
{
	m_learningRate = value;
}

void LogisticRegression::log(const std::vector<double>& theta, const std::vector<double>& newTheta, const std::vector<double>& minDelta)
{
	{
		std::cout << "old theta: ";
		for (int i = 0; i < theta.size(); i++)
			std::cout << theta[i] << " ";
		std::cout << std::endl;
	}

	{
		std::cout << "new theta: ";
		for (int i = 0; i < theta.size(); i++)
			std::cout << newTheta[i] << " ";
		std::cout << std::endl;
	}

	{
		std::cout << "delta: ";
		for (int i = 0; i < theta.size(); i++)
			std::cout << abs(theta[i] - newTheta[i]) << " ";
		std::cout << std::endl;
	}


	{
		std::cout << "Min delta: ";
		for (int i = 0; i < theta.size(); i++)
			std::cout << minDelta[i] << " ";
		std::cout << std::endl;
	}
}

std::vector<double> LogisticRegression::theta() const
{
	return m_theta;
}

double LogisticRegression::h(const Example& trainData) const
{
	double powerValue = 0;

	for (int i = 0; i < m_features; i++)
		powerValue -= m_theta[i] * trainData.m_features[i];

	double result = 1 / (1 + exp(powerValue));

	return result;
}

double LogisticRegression::derivativeValue(int j) const
{
	double value = 0;

	for (int i = 0; i < m_examples; i++)
		value += (h(m_trainingData[i]) - m_trainingData[i].m_target)*m_trainingData[i].m_features[j];

	return value;
}

std::vector<double> LogisticRegression::train()
{
	int count = 0;
	int showLogIf = 300;
	int minDeltaAbort = 4;
	bool converged = false;

	std::vector<double> minDelta(m_features, std::numeric_limits<double>::max());

	while (!converged)
	{
		auto newTheta = train();

		bool check = true;
		for (int j = 0; j < m_features; j++)
		{
			minDelta[j] = std::min(minDelta[j], abs(newTheta[j] - m_theta[j]));

			if (abs(newTheta[j] - m_theta[j]) >= m_threshold)
			{
				check = false;
				break;
			}
		}

		if (showLogIf == count)
		{
			log(m_theta, newTheta, minDelta);
			count = 0;
		}

		m_theta = newTheta;

		if (check == true)
		{
			converged = true;
			break;
		}

		count++;
	}

	return m_theta;
}

bool LogisticRegression::test(const TestExample& value) const
{
	int ans = h(value) >= 0.5? 1 : 0;
	
	return ans == value.m_target;
}