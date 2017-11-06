// ConsoleApplication2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;

struct TrainContext
{

  const int nClass0;
  const int nClass1;
  const int nClass2;


  float class0[10];
  float class1[10];
  float class2[10];

  TrainContext(int nClass0, int nClass1, int nClass2) : nClass0(nClass0), nClass1(nClass1), nClass2(nClass2)
  {
    for (int i = 0; i < 10; i++)
    {
      class0[i] = 0;
      class1[i] = 0;
      class2[i] = 0;
    }
  }

};



struct Person
{
  float data[10];

  float evidence0;
  float evidence1;
  float evidence2;

  int predicted_class = -1;

  void build(const uniform_real_distribution<>& dist, mt19937& gen)
  {
    for (int i = 0; i < 10; ++i) {
      data[i] = float(dist(gen));
    }
  }


  // Class 0, 1, 2

  int classify()
  {
    int d = 13;

    bool
      increasing0 = data[1] > data[0];

    bool
      increasing1 = data[2] > data[1];

    bool
      increasing2 = data[3] > data[2];

    bool
      mid_boost = data[4] >= 0.8f;

    bool
      extra_boost = increasing0 && increasing1 && increasing2 && data[5] >= 0.91f;

    bool
      low_high0 = data[6] < 0.2f;

    bool
      low_high1 = data[7] < data[6];

    float
      fak = 0.7f;

    if (increasing0)
    {
      fak += 0.1f;
    }

    if (increasing0 && increasing1)
    {
      fak += 0.15f;
    }
    else
    {
      return 0;
    }

    if (increasing0 && increasing1 && increasing2)
    {
      fak += 0.2f;
    }

    if (mid_boost)
    {
      fak *= 1.3f;
    }

    if (extra_boost)
    {
      fak *= 3.1f;
    }

    if (low_high0 && increasing2)
    {
      fak *= 1.1f;
    }

    if (low_high1 && extra_boost)
    {
      fak *= 1.7f;
    }
    else
    {
      return 1;
    }

    int
      eval = int(fak * d);

    if (eval > 49)
    {
      return 2;
    }
    else
    {
      return 1;
    }

  }

  void calculate_evidence(const TrainContext& context)
  {

    evidence0 = 0.f;
    evidence1 = 0.f;
    evidence2 = 0.f;

    for (int i = 0; i < 10; i++)
    {
      float
        x = data[i];

      float
        delta0 = 1.f - abs(x - context.class0[i]), // 0..1
        delta1 = 1.f - abs(x - context.class1[i]),
        delta2 = 1.f - abs(x - context.class2[i]);

      delta0 -= .5f;  // -.5 .. .5
      delta1 -= .5f;
      delta2 -= .5f;

      delta0 *= 2.f; // -1 .. 1
      delta1 *= 2.f;
      delta2 *= 2.f;


      evidence0 += delta0;
      evidence1 += delta1;
      evidence2 += delta2;

    }

    // Softmax;

    float sum = exp(evidence0) + exp(evidence1) + exp(evidence2);

    evidence0 = exp(evidence0) / sum;
    evidence1 = exp(evidence1) / sum;
    evidence2 = exp(evidence2) / sum;
  
    AI_NOP;


  }

};







void train(const float(&data)[10], const int label, TrainContext& context)
{

  // Input: Data point is a 10 vector of float [0,1]
  // Label is 0,1 or 2.

  float* w = nullptr;

  int N = -1;

  if (label == 0)
  {
    w = context.class0;
    N = context.nClass0;
  }
  else if (label == 1)
  {
    w = context.class1;
    N = context.nClass1;
  }
  else
  {
    w = context.class2;
    N = context.nClass2;
  }

  assert(w != nullptr);
  assert(N != -1);

  for (int iDimension = 0; iDimension < 10; iDimension++)
  {
    float
      x = data[iDimension];

    w[iDimension] += (x / N);
  }
 
}

int measure_quality(Person p [], const int N, const TrainContext& c)
{
  const float
    N_FULL = 10000.f;

  int
    nCorrect = 0,
    nError = 0;

  for (int iPerson = 0; iPerson < N; iPerson++)
  {
    p[iPerson].calculate_evidence(c);
  }

  int
    nClass2 = int(1.f * c.nClass2 * N / N_FULL);

  std::sort(p, p + N, [](Person const & a, Person const & b) -> bool
  {
    return a.evidence2 > b.evidence2;
  });

  for (int iPerson = 0; iPerson < nClass2; iPerson++)
  {
    p[iPerson].predicted_class = 2;
  }

  int
    nClass1 = int(1.f * c.nClass1 * N / N_FULL);

  std::sort(p + nClass2, p + N, [](Person const & a, Person const & b) -> bool
  {
    return a.evidence1 > b.evidence1;
  });

  for (int iTestPerson = nClass2; iTestPerson < nClass1 + nClass2; iTestPerson++)
  {
    p[iTestPerson].predicted_class = 1;
  }

  for (int iTestPerson = nClass1 + nClass2; iTestPerson < N; iTestPerson++)
  {
    p[iTestPerson].predicted_class = 0;
  }


  for (int iTestPerson = 0; iTestPerson < N; iTestPerson++)
  {

    int
      nEvaluateSize = p[iTestPerson].classify();


    const float(&data)[10] = p[iTestPerson].data;

    int
      nPredictedSize = p[iTestPerson].predicted_class;

    if (nPredictedSize == nEvaluateSize)
    {
      nCorrect++;
    }
    else
    {
      nError++;
    }
  }

  return nError;
}



int main()
{
  random_device rd;   // non-deterministic generator  
  mt19937 gen(rd());  // to seed mersenne twister.  
                      // replace the call to rd() with a  
                      // constant value to get repeatable  
                      // results.  


  uniform_real_distribution<> dist(0, 1); // distribute results between 0 and 1

  const int
    nPerson = 10000;

  Person p[nPerson];

  int
    nClass[3] = { 0 };

  for (int iPerson = 0; iPerson < nPerson; iPerson++)
  {
    p[iPerson].build(dist, gen);

    int
      result = p[iPerson].classify();

    nClass[result]++;

  }

  TrainContext
    c(nClass[0], nClass[1], nClass[2]);

  for (int iPerson = 0; iPerson < nPerson; iPerson++)
  {
    int
      result = p[iPerson].classify();

    const float(&data)[10] = p[iPerson].data;

    train(data, result, c);
  }

  int
    nErrorTrainingSet = measure_quality(p, 10000, c);


  // Evaluate quality


  const int
    nTestPerson = 1000;

  Person t[nTestPerson];

  for (int iTestPerson = 0; iTestPerson < nTestPerson; iTestPerson++)
  {
    t[iTestPerson].build(dist, gen);
  }

  int
    nError = measure_quality(t, 1000, c);

  return 0;

}

