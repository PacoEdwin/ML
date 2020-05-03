#pragma once

#include <vector>

class LogisticRegression;

class Example
{
	friend LogisticRegression;

public:
	Example() = default;
	Example(const std::vector<double>& features, int target);

	virtual double getFeature(int i) = 0;
	virtual std::vector<double>& getFeatures() = 0;
	virtual int getTarget() = 0;


protected:
	int m_target;
	std::vector<double> m_features;
};

class TrainingExample :
	public Example
{
public:
	TrainingExample(const std::vector<double>& features, int target);

	virtual double getFeature(int i) override;
	virtual std::vector<double>& getFeatures() override;
	virtual int getTarget() override;
};

class TestExample :
	public Example
{
public:
	TestExample(const std::vector<double>& features, int target);

	virtual double getFeature(int i) override;
	virtual std::vector<double>& getFeatures() override;
	virtual int getTarget() override;
};