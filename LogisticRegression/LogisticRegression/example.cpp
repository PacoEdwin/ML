// projects includes
#include "example.h"

Example::Example(const std::vector<double>& features, int target):
	m_features(features),
	m_target(target) {}	


TrainingExample::TrainingExample(const std::vector<double>& features, int target) :
	Example(features, target) {}


double TrainingExample::getFeature(int i) 
{ 
	return m_features.at(i); 
}

std::vector<double>& TrainingExample::getFeatures()
{ 
	return m_features; 
}

int TrainingExample::getTarget() 
{
	return m_target; 
}


TestExample::TestExample(const std::vector<double>& features, int target) :
	Example(features, target) {}

double TestExample::getFeature(int i) 
{
	return m_features.at(i); 
}

std::vector<double>& TestExample::getFeatures() 
{
	return m_features; 
}

int TestExample::getTarget() 
{
	return m_target; 
}