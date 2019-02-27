

#include <iostream>
#include <windows.h>
#include <random>
#include <ctime>
#include <cstdlib>
#include <omp.h>

using namespace std;
//���������� ��������� ����� double
double getRandomNumber(int min, int max)
{
	return (double)min + (double)rand() / (double)RAND_MAX;
}

//��������� �������������� ������, ������� ��� ������, ���������� �������
void randomizeSmth(double* vector, int size) {
	for (int i = 0; i < size; i++) {
		vector[i] = getRandomNumber(1, 10);
	}
}

//������� ������ ������ - ������� ��� ������. ��� �������� ���������� ������� ���������� � ��������� �������� ������� �����������: size*size
double* createNullable(int size) {
	double* matrix = new double[size];
	for (int i = 0; i < size; i++)
		matrix[i] = 0;
	return matrix;
}

//����� ������� �� �����
void printMatrix(double* matrix, int size) {
	printf_s("Matrix\n");
	for (int i = 0; i < size * size; i++)
	{
		printf_s("%f ", matrix[i]);
		if (i % size == size - 1)
			printf_s("\n");
	}
}
//����� ������� �� �����
void printVector(double* vector, int size) {
	printf_s("vector\n");
	for (int i = 0; i < size; i++) {
		printf_s("%f", vector[i]);
		printf_s(" ");
	}
	printf_s("\n");
}
// ��������� ������� � ������� ������ ������
void matrixVectorMultiplication(double* pMatrix, double* pVector, double* pResult, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			pResult[i] += pVector[j] * pMatrix[i*size + j];//�������� ������ �� �������
		}
	}
}
// ��������� ������������-������������ ������������ ������� � �������������� ��������� ���������
double* randomSynchronousPositiveDefiniteMatrix(int size) {
	double* matrix = new double[size*size];
	int k = 0;
	for (int i = 0; i < size; i++) {
		k = 0;
		for (int j = 0; j < size; j++) {
			if (i == j) {
				matrix[(size + 1)*i] = getRandomNumber(size, (size + 1));
			}
			else if (j > i) {
				k++;
				matrix[(size + 1)*i + k] = getRandomNumber((size - k), (size - k + 1));
				matrix[(size + 1)*j - k] = matrix[(size + 1)*i + k];
			}
		}

	}
	return matrix;
}


// ������������� �����
void initializeMatrix(double* &pMatrix, double* &pVector, double* &pResult) {
	pMatrix = new double[4]{ 3, -1, -1, 3 };
	pVector = new double[2]{ 3, 7 };
	pResult = new double[2]{ 0,0 };
}

// ������ ��� ��������� - ���������� ��������� �� ������� ���� (currentG)
void compute_gradient(double* pMatrix, double* pVector, double* previousX, double* &currentG, int size) {
#pragma omp parallel for num_threads(4)
	for (int i = 0; i<size; i++) {
		currentG[i] = -pVector[i];
		for (int j = 0; j<size; j++)
			currentG[i] += pMatrix[i*size + j] * previousX[j];
	}
}

// ������ ��� ��������� - ���������� ������� ����������� (currentD)
void compute_direction(double* currentG, double* previousG, double* previousD, double* &currentD, int size) {
	double IP1 = 0, IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2) num_threads(4)
	for (int i = 0; i<size; i++) {
		IP1 += currentG[i] * currentG[i];
		IP2 += previousG[i] * previousG[i];
	}
#pragma omp parallel for num_threads(4)
	for (int i = 0; i<size; i++) {
		currentD[i] = -currentG[i] + previousD[i] * IP1 / IP2;
	}
}

// ������ ��� ��������� - ���������� �������� �������� �� ���������� ����������� (step)
void compute_scalyar_step(double* currentD, double* currentG, double* pMatrix, double &step, double* denom, int size) {
	double IP1 = 0, IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2) num_threads(4)
	for (int i = 0; i<size; i++) {
		denom[i] = 0;
		for (int j = 0; j<size; j++)
			denom[i] += pMatrix[i*size + j] * currentD[j];
		IP1 += currentD[i] * currentG[i];
		IP2 += currentD[i] * denom[i];
	}
	step = -IP1 / IP2;
}

// ��������� ��� ��������� - ���������� ������ �����������, �������������� ���������� (currentX)
void compute_x(double* previousX, double step, double* currentD, double* &currentX, int size) {
#pragma omp parallel for num_threads(4)
	for (int i = 0; i<size; i++) {
		currentX[i] = previousX[i] + step * currentD[i];
	}
}

/* ���������� ��� ��������� �������� �� ���� ���������� ������� ���������. ��� ����� ����������� ������������ �������� ����� ������������ ��������
�� ������ � ����� ������� �� ����� ���������� �������. � ����� ���������� ������������� �����������.*/
bool checkStopCondition(double* previousX, double* currentX, int size, float accuracy) {
	double sum = 0;
	double max = 0;
	for (int i = 0; i < size; i++) {
		if (max < fabs(currentX[i] - previousX[i])) {
			max = fabs(currentX[i] - previousX[i]);
		}
		sum += currentX[i] * currentX[i];
	}
	sum = sqrt(sum);
	return (max / sum) > accuracy;
}

/* ���������� ������������� ����������� ����� ��������� ���������*/
double relativeError(double* previousX, double* currentX, int size) {
	double sum = 0;
	double max = 0;
	for (int i = 0; i < size; i++) {
		if (max < fabs(currentX[i] - previousX[i])) {
			max = fabs(currentX[i] - previousX[i]);
		}
		sum += currentX[i] * currentX[i];
	}
	sum = sqrt(sum);
	return (max / sum);
}

//���������� ���������� ����������� ����� Vector1 � Vector2 � ���� �����
double absoluteError(double* Vector1, double* Vector2, int size) {
	double max = 0;
	for (int i = 0; i < size; i++) {
		if (max < fabs(Vector1[i] - Vector2[i]))
		{
			max = fabs(Vector1[i] - Vector2[i]);
		}
	}
	return max;
}

// ������ ���������� ��������� ������ ����������� ����������
void algorithmCalculation(double* pMatrix, double* pVector, double* pResult, int size, float accuracy) {
	double *currentX, *previousX;//x
	double *currentG, *previousG;//g
	double *currentD, *previousD;//d
	double *denom;//�����������?
	double step; // ��������
				 // ������ �������������� �������� �������� � �� ����������� �� ��������� �� �� ����� ������� - �������� ������� ������� � �����
	int numberOfIterations = 1, maxNumberOfIterations = size + 1;

	//�������������� ������� ��� ���������
	currentX = createNullable(size);
	previousX = createNullable(size);//=0
	currentG = createNullable(size);
	previousG = createNullable(size);
	currentD = createNullable(size);//=0
	previousD = createNullable(size);
	denom = createNullable(size);

	//������ ����������� �������� ��������
	for (int i = 0; i<size; i++) {
		previousG[i] = -pVector[i];
	}

	do {
		// ����� ������ ��������� (����� ������) ���������� �������������� �������. �� �������, ��� ���� �������� ������ �����������.
		if (numberOfIterations > 1) {
			for (int i = 0; i < size; i++) {
				previousX[i] = currentX[i];
				previousG[i] = currentG[i];
				previousD[i] = currentD[i];
			}
		}
		// ���������������� ���������� ������ ����� ��������� � ������� ������������� ����������
		compute_gradient(pMatrix, pVector, previousX, currentG, size);//���������� ��������
																	  //printVector(currentG, size);
		compute_direction(currentG, previousG, previousD, currentD, size);
		//printVector(currentD, size);
		compute_scalyar_step(currentD, currentG, pMatrix, step, denom, size);
		//printf_s("%f", step);
		compute_x(previousX, step, currentD, currentX, size);
		//printVector(currentX,size);
		numberOfIterations++;
	} while (checkStopCondition(previousX, currentX, size, accuracy));

///	printf_s("Quantity of iteratins: ");
///	printf_s("%d\n", numberOfIterations);
	// ������ � �������������� ������
	for (int i = 0; i<size; i++)
		pResult[i] = currentX[i];
	//DeleteVectors TODO  - ���������� ������������ ������������ ������, ���������� ��� ������� � ������� ����� ����������. �.�. ��� ����������� ���������� �� ���������� ������������ ��� ����� ��������� ������������ ���������� ������.
}


//�������� ������ ������ ����� ������������ ������� �� ���������
void checkTotalResultAbsolutely(double* pMatrix, double* pVector, double* pResult, int size) {
	double* newVector;
	newVector = createNullable(size);
	//printMatrix(pMatrix, size);
	///printVector(pVector, size);
	//printVector(pResult, size);
	matrixVectorMultiplication(pMatrix, pResult, newVector, size);
	//printVector(newVector, size);
	double error1 = absoluteError(pVector, newVector, size);
	printf_s("Pogreshnost absolunnaya: ");
	printf_s("%.15f\n", error1);
	
	double error2 = relativeError(pVector, newVector, size);
	printf_s("Pogreshnost otnositelnaya: ");
	printf_s("%.15f\n", error2);
}


// ������� ����� 
int main(int argc, char **argv) {
	srand(static_cast<unsigned int>(time(0))); //��� �������

	double* pMatrix; //�������� �������
	double* pVector; //������
	double* pResult; //� ������ ��� ����������

	int size = 1000;//������ �������. ������ �������� ����������

	int dimensions[9] = { 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000};
	int iterations[9] = { 100, 100, 100, 100, 80, 50, 50, 50, 25};

	float accuracy = 0.01f; // ����������� ���������� - �������. ������ �������� �����������
	const unsigned __int64 frequency = 2000000000;

	for (int iterator = 0; iterator < 9; iterator++) {

		pMatrix = randomSynchronousPositiveDefiniteMatrix(dimensions[iterator]);
		pVector = createNullable(dimensions[iterator]);
		randomizeSmth(pVector, dimensions[iterator]);
		pResult = createNullable(dimensions[iterator]);

		unsigned __int64 registrStartTime, registrEndTime;
		
		DWORD windowsStartTime = GetTickCount();
		registrStartTime = __rdtsc();

		for (int i = 0; i < iterations[iterator]; i++) {
			algorithmCalculation(pMatrix, pVector, pResult, dimensions[iterator], accuracy);
		}

		registrEndTime = __rdtsc();
		DWORD windowsEndTime = GetTickCount();

		printf_s("Dimension: %d\n", dimensions[iterator]);
		printf_s("Iterations: %d\n", iterations[iterator]);

		printf_s("Windows, %f milliseconds\n", (double)(windowsEndTime - windowsStartTime) / (double)iterations[iterator]);
		printf_s("Register  , %f milliseconds\n", ((registrEndTime - registrStartTime) / (double)frequency) * 1000 / (double)iterations[iterator]);

		checkTotalResultAbsolutely(pMatrix, pVector, pResult, dimensions[iterator]);
	}
	scanf_s("%d");
	return 0;
}
