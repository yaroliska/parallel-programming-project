// Ali_Yara.cpp: определяет точку входа для консольного приложения.
//
#include "stdafx.h"
#include <iostream>
#include <random>
#include <ctime>
#include <cstdlib>

using namespace std;
//возвращает рандомное число double
double getRandomNumber(int min, int max)
{
	static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
	// равномерно распределяем рандомное число в нашем диапазоне
	return static_cast<double>(rand() * fraction * (max - min + 1) + min);
}
//заполняет матрицу рандомными числами
void randomizeMatrix(double* matrix, int size) {
	for (int i = 0; i < size * size; i++)
		matrix[i] = getRandomNumber(1,10);
}
//заполняет вектор рандомными числами
void randomizeVector(double* vector, int size) {
	for (int i = 0; i < size; i++) {
		vector[i] = getRandomNumber(1, 10);
	}
}
//создает пустую матрицу
double* createMatrix(int width, int height) {
	double* matrix = new double[width * height];
	for (int i = 0; i < width * height; i++)
		matrix[i] = 0;
	return matrix;
}
//создает пустой вектор
double* createVector(int size) {
	double* vector = new double[size];
	for (int i = 0; i < size; i++) {
		vector[i] = 0;
	}
	return vector;
}
//вывод матрицы на экран
void printMatrix(double* matrix, int size) {
	printf_s("Matrix\n");
	for (int i = 0; i < size * size; i++)
	{
		printf_s("%f ", matrix[i]);
		if (i % size == size - 1)
			printf_s("\n");
	}
}
//вывод вектора на экран
void printVector(double* vector, int size) {
	printf_s("vector\n");
	for (int i = 0; i < size; i++) {
		printf_s("%f", vector[i]);
		printf_s(" ");
	}
	printf_s("\n");
}
//умножение вектора и матрицы вектор слева
void vectorMatrixMultiplication(double* pMatrix, double* pVector, double* pResult, int size) {
		for (int i=0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				pResult[i] += pVector[j] * pMatrix[j*size+i];//умножаем строку на столбец
			}
		}
}
//умножение вектора и матрицы вектор справа
void matrixVectorMultiplication(double* pMatrix, double* pVector, double* pResult, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			pResult[i] += pVector[j] * pMatrix[i*size + j];//умножаем строку на столбец
		}
	}
}

//определяет размер и элементы для матрицы A и вектора b, создает пустой вектор для результата.
void ProcessInitialization(double* &pMatrix, double* &pVector, double* &pResult, int &size) {
	do {
		printf("\nEnter size: ");
		scanf_s("%d", &size);
		if (size <= 0) {
			printf("Size of the objects must be greater than 0! \n ");
		}
	} while (size <= 0);
	pMatrix = createMatrix(size, size);
	randomizeMatrix(pMatrix,size);
	pVector = createVector(size);
	randomizeVector(pVector,size);
	pResult = createVector(size);
}
void initializeMatrix(double* &pMatrix, double* &pVector, double* &pResult) {
	pMatrix= new double[4]{ 3, -1, -1, 3 };
	pVector = new double[2] {3, 7};
	pResult = new double[2]{ 0,0 };
}

void compute_gradient(double* pMatrix, double* pVector, double* previousX, double* &currentG, int size) {
	printf("\ncompute_gradient\n");
	for (int i = 0; i<size; i++) {
		currentG[i] = -pVector[i];
		for (int j = 0; j<size; j++)
			currentG[i] += pMatrix[i*size + j] * previousX[j];
	}
}
void compute_direction(double* currentG, double* previousG, double* previousD, double* &currentD, int size) {
	printf("\ncompute_direction\n");
	double IP1 = 0, IP2 = 0;
	for (int i = 0; i<size; i++) {
		IP1 += currentG[i] * currentG[i];
		IP2 += previousG[i] * previousG[i];
	}
	for (int i = 0; i<size; i++) {
		currentD[i] = -currentG[i] + previousD[i] * IP1 / IP2;
	}
}
void compute_size_step(double* currentD, double* currentG, double* pMatrix, double &step, double* denom, int size) {
	double IP1 = 0, IP2 = 0;
	for (int i = 0; i<size; i++) {
		denom[i] = 0;
		for (int j = 0; j<size; j++)
			denom[i] += pMatrix[i*size + j] * currentD[j];
		IP1 += currentD[i] * currentG[i];
		IP2 += currentD[i] * denom[i];
	}
	step = -IP1 / IP2;
}
void compute_x(double* previousX, double step, double* currentD, double* &currentX, int size) {
	for (int i = 0; i<size; i++) {
		currentX[i] = previousX[i] + step * currentD[i];
	}
}
float compare_x(double* previousX, double* currentX, int size) {
	float f;
	return f;
}
void parallelResultCalculation(double* pMatrix, double* pVector, double* pResult, int size) {
	double *currentX, *previousX;//x
	double *currentG, *previousG;//g
	double *currentD, *previousD;//d
	double *denom;//знаменатель?
	double step;//s
	int Iter = 1, MaxIter = size + 1;
	float Accuracy = 0.0001f;
	//инициализируем вектора для алгоритма
	currentX = createVector(size);
	previousX = createVector(size);//=0
	currentG = createVector(size);
	previousG = createVector(size);
	currentD = createVector(size);//=0
	previousD = createVector(size);
	denom = createVector(size);

	//задаем изначальные значения векторам
	for (int i = 0; i<size; i++) {
		//PreviousApproximation[i] = 0;
		//PreviousDirection[i] = 0;
		previousG[i] = -pVector[i];
	}
	do {
		if (Iter > 1) {//поменять местами указатели
			//SwapPointers(PreviousApproximation, CurrentApproximation);
			//SwapPointers(PreviousGradient, CurrentGradient);
			//SwapPointers(PreviousDirection, CurrentDirection);
		}
		compute_gradient(pMatrix, pVector, previousX, currentG, size);//определили градиент
		printVector(currentG, size);
		compute_direction(currentG, previousG, previousD, currentD, size);
		printVector(currentD, size);
		compute_size_step(currentD, currentG, pMatrix, step, denom, size);
		printf_s("%f", step);
		compute_x(previousX, step, currentD, currentX, size);
		printVector(currentX,size);
		Iter++;
	} while
		((compare_x(previousX, currentX, size) > Accuracy)
			&& (Iter < MaxIter));
	for (int i = 0; i<size; i++)
		pResult[i] = currentX[i];
	//DeleteVectors(CurrentApproximation, PreviousApproximation, CurrentGradient,PreviousGradient, CurrentDirection, PreviousDirection, Denom);
}
int main(int argc, char **argv) {
	srand(static_cast<unsigned int>(time(0))); //для рандома
	double* pMatrix; //объявили матрицу
	double* pVector; //вектор
	double* pResult; //и вектор для результата
	int size =2;//объявили размер матрицы
	/*//создаем матрицу и вектора и заполняем матрицу и вектор рандомными числами
	ProcessInitialization(pMatrix, pVector, pResult, size);
	printMatrix(pMatrix, size);
	printVector(pVector, size);
	printVector(pResult, size);
	//умножение вектора на матрицу (вектор слева)
	matrixVectorMultiplication(pMatrix, pVector, pResult, size);
	printMatrix(pMatrix, size);
	printVector(pVector, size);
	printVector(pResult, size);
	//умножение матрицы на вектор (вектор справа)
	vectorMatrixMultiplication(pMatrix, pVector, pResult, size);
	printMatrix(pMatrix, size);
	printVector(pVector, size);
	printVector(pResult, size);
	//скалярное произведение векторов*/
	initializeMatrix(pMatrix, pVector, pResult);
	printMatrix(pMatrix, size);
	printVector(pVector, size);

	parallelResultCalculation(pMatrix, pVector, pResult, size);

	return 0;
}
