#pragma once

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

	/// Returns theta for dataset
	std::vector<double> theta() const;

	/// Train model
	std::vector<double> train();
	/// Returns true if model gave correct answer
	bool test(const TestExample&) const;

private:
	void log(const std::vector<double>& theta, const std::vector<double>& newTheta, const std::vector<double>& minDelta);

	/// Value of sigmoid function 
	double h(const Example& trainData) const;
	/// Value of cost function derivative
	double derivativeValue(int j) const;
	/// Return updated thetas values
	std::vector<double> calculate() const;

	unsigned m_examples;
	unsigned m_features;

	double m_threshold;
	double m_learningRate;

	std::vector<double> m_theta;
	std::vector<TrainingExample> m_trainingData;
};