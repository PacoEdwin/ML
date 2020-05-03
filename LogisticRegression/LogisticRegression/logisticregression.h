#pragma once
#pragma once
#ifndef GRADIENT_H
#define GRADIENT_H

// std includes
#include <vector>

// prtojects includes
#include "example.h"

class LogisticRegression
{

public:
	LogisticRegression();
	LogisticRegression(const std::vector<TrainingExample>& trainingData);

	void addExample(const TrainingExample& value);

	void setThreshold(const double& value);
	void setLearningRate(const double& value);

	std::vector<double> theta() const;

	std::vector<double> train();
	bool test(const TestExample&) const;

private:
	void log(const std::vector<double>& theta, const std::vector<double>& newTheta, const std::vector<double>& minDelta);

	double h(const Example& trainData) const;
	double derivativeValue(int j) const;

	unsigned m_examples;
	unsigned m_features;

	double m_threshold;
	double m_learningRate;

	std::vector<double> m_theta;
	std::vector<TrainingExample> m_trainingData;
}

#endif