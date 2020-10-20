//#include "CImg.h"
//using namespace cimg_library;

#include <iostream>
#include <random>

using namespace std;

double divide_q()
{
	double DIVIDE_PROB = 0.0416;
	default_random_engine generator;
	binomial_distribution<int> binomialdivide(1, DIVIDE_PROB);

	double value = binomialdivide(generator);
	return value;
}

double symmetric_divide_q()
{
	double SYMMETRIC_DIVIDE_PROB = 0.3;
	default_random_engine generator;
	binomial_distribution<int> binomialsymdivide(1, SYMMETRIC_DIVIDE_PROB);

	double value = binomialsymdivide(generator);
	return value;
}

double die_q()
{
	double DEATH_PROB = 0.01;
	default_random_engine generator;
	binomial_distribution<int> binomialdeath(1, DEATH_PROB);

	double value = binomialdeath(generator);
	return value;
}

int main()
{
	for (int i = 0; i < 1000; i++)
	{
		cout << divide_q() << "\t" << symmetric_divide_q() << "\t" << die_q() << endl;
	}
	cin.get();
	return 0;
}