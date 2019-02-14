// Ali_Yara.cpp: определяет точку входа для консольного приложения.
//
#include "stdafx.h"
#include <iostream>
#include <random>
#include <ctime>
#include <cstdlib>

using namespace std;

//Здесь определяются значения для матрицы A и вектора b
double getRandomNumber(int min, int max)
{
	static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
	// равномерно распределяем рандомное число в нашем диапазоне
	return static_cast<double>(rand() * fraction * (max - min + 1) + min);
}
//Эта функция определяет размер и элементы для матрицы A и вектора b.
void ProcessInitialization(double * pMatrix, double* &pVector, double* &pResult, int Size) {//передаем штуку типа указатель со значением взятым через &)
	do {
		printf("\nEnter size of the initial objects: ");
		scanf_s("%d", &Size);
		if (Size <= 0) {
			printf("Size of the objects must be greater than 0! \n ");
		}
	} while (Size <= 0);
	pMatrix = new double[Size*Size];
	pVector = new double[Size];
	pResult = new double[Size];
	//заполняем массивы рандомными данными
	for (int i = 0; i < Size; i++) {
		pResult[i] = 0;
	}
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			//getRandomNumber(1, 10);
		}
	}
	for (int i = 0; i < Size; i++) {
		pVector[i] = getRandomNumber(1, 10);
	}
	//RandomDataInitialization(pMatrix, pVector, Size);
}
void PrintMatrix(double *pMatrix, int &Size) {
	cout << "Matrix" << endl;
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++)
		{
			cout << pMatrix[i, j] << "\t"; //вывод очередного элемента матрицы
		}
		cout << endl;
	}
}
double** createMatrix(int size) {
	double** matrix = new double*[size];

	for (int i = 0; i < size; i++)
		matrix[i] = new double[size];

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			matrix[i][j] = 0;
	return matrix;
}

double* createVector(int size) {
	double* vector = new double[size];
	for (int i = 0; i < size; i++) {
		// Заполнение массива и вывод значений его элементов
		vector[i] = 0;
	}
	return vector;
}

void randomizeMatrix(int size, double** matrix) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			matrix[i][j] = (double)(rand()) / RAND_MAX * 10;
}

void randomizeVector(int size, double* vector) {
	for (int i = 0; i < size; i++)
			vector[i] = (double)(rand()) / RAND_MAX * 10;
}

void printMatrix(int size, double** matrix) {
	printf_s("Matrix\n");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			printf_s("%f ", matrix[i][j]);
		printf_s("\n");
	}
}

int main(int argc, char **argv) {
	srand(static_cast<unsigned int>(time(0))); //для рандома
	int size;
	cout << "Enter size of matrix: ";
	cin >> size;
	//создаем матрицу и вектор
	double** AMatrix = createMatrix(size);
	double* BVector = createVector(size);
		//заполняем матрицу рандомными числами
	randomizeMatrix(size,AMatrix);
	randomizeVector(size, BVector);
	printMatrix(size, AMatrix);


	// Matrix-vector multiplication
	//SerialResultCalculation(pMatrix, pVector, pResult, Size);
	// Program termination
	//ProcessTermination(pMatrix, pVector, pResult);
	return 0;
}
