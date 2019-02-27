

#include <iostream>
#include <windows.h>
#include <random>
#include <ctime>
#include <cstdlib>
#include <omp.h>

using namespace std;
//возвращает рандомное число double
double getRandomNumber(int min, int max)
{
	return (double)min + (double)rand() / (double)RAND_MAX;
}

//заполняет математический объект, матрицу или вектор, рандомными числами
void randomizeSmth(double* vector, int size) {
	for (int i = 0; i < size; i++) {
		vector[i] = getRandomNumber(1, 10);
	}
}

//создает пустой объект - матрицу или вектор. Для создания квадратной матрицы достаточно в параметре передать квадрат размерности: size*size
double* createNullable(int size) {
	double* matrix = new double[size];
	for (int i = 0; i < size; i++)
		matrix[i] = 0;
	return matrix;
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
// умножение вектора и матрицы вектор справа
void matrixVectorMultiplication(double* pMatrix, double* pVector, double* pResult, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			pResult[i] += pVector[j] * pMatrix[i*size + j];//умножаем строку на столбец
		}
	}
}
// Генерация положительно-определенной симметричной матрицы с использованием рандомной генерации
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


// ТЕСТИРОВОЧНЫЙ МЕТОД
void initializeMatrix(double* &pMatrix, double* &pVector, double* &pResult) {
	pMatrix = new double[4]{ 3, -1, -1, 3 };
	pVector = new double[2]{ 3, 7 };
	pResult = new double[2]{ 0,0 };
}

// Первый шаг алгоритма - вычисление градиента на текущем шаге (currentG)
void compute_gradient(double* pMatrix, double* pVector, double* previousX, double* &currentG, int size) {
#pragma omp parallel for num_threads(4)
	for (int i = 0; i<size; i++) {
		currentG[i] = -pVector[i];
		for (int j = 0; j<size; j++)
			currentG[i] += pMatrix[i*size + j] * previousX[j];
	}
}

// Второй шаг алгоритма - вычисление вектора направления (currentD)
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

// Третий шаг алгоритма - вычисление величины смещения по выбранному направлению (step)
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

// Четвертый шаг алгоритма - вычисление нового приближения, промежуточного результата (currentX)
void compute_x(double* previousX, double step, double* currentD, double* &currentX, int size) {
#pragma omp parallel for num_threads(4)
	for (int i = 0; i<size; i++) {
		currentX[i] = previousX[i] + step * currentD[i];
	}
}

/* Сравнивает две последние итерации на факт выполнения условия остановки. Для этого вычисляется максимальное различие между координатами векторов
по модулю и потом делится на длину последнего вектора. В итоге получается относительная погрешность.*/
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

/* возвращает относительную погрешность между итоговыми векторами*/
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

//возвращает абсолютную погрешность между Vector1 и Vector2 в виде числа
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

// Полное вычисление алгоритма метода сопряженных градиентов
void algorithmCalculation(double* pMatrix, double* pVector, double* pResult, int size, float accuracy) {
	double *currentX, *previousX;//x
	double *currentG, *previousG;//g
	double *currentD, *previousD;//d
	double *denom;//знаменатель?
	double step; // смещение
				 // Точное предназначение подсчёта итераций и их ограничения по максимуму не до конца понятно - возможно придётся удалить в конце
	int numberOfIterations = 1, maxNumberOfIterations = size + 1;

	//инициализируем вектора для алгоритма
	currentX = createNullable(size);
	previousX = createNullable(size);//=0
	currentG = createNullable(size);
	previousG = createNullable(size);
	currentD = createNullable(size);//=0
	previousD = createNullable(size);
	denom = createNullable(size);

	//задаем изначальные значения векторам
	for (int i = 0; i<size; i++) {
		previousG[i] = -pVector[i];
	}

	do {
		// Перед каждой итерацией (кроме первой) необходимо переопределять вектора. Те вектора, что были текущими станут предыдущими.
		if (numberOfIterations > 1) {
			for (int i = 0; i < size; i++) {
				previousX[i] = currentX[i];
				previousG[i] = currentG[i];
				previousD[i] = currentD[i];
			}
		}
		// Последовательное выполнение четырёх шагов алгоритма с выводом промежуточных результаов
		compute_gradient(pMatrix, pVector, previousX, currentG, size);//определили градиент
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
	// Запись в результирующий вектор
	for (int i = 0; i<size; i++)
		pResult[i] = currentX[i];
	//DeleteVectors TODO  - Необходимо организовать освобождение памяти, выделенной под матрицы и вектора после вычислений. Т.к. при организации вычислений на нескольких размерностях без этого перезойдёт переполнение оперативнй памяти.
}


//проверка работы метода путем перемножения матрицы на результат
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


// ГЛАВНЫЙ МЕТОД 
int main(int argc, char **argv) {
	srand(static_cast<unsigned int>(time(0))); //для рандома

	double* pMatrix; //объявили матрицу
	double* pVector; //вектор
	double* pResult; //и вектор для результата

	int size = 1000;//Размер матрицы. Данный параметр мутабельый

	int dimensions[9] = { 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000};
	int iterations[9] = { 100, 100, 100, 100, 80, 50, 50, 50, 25};

	float accuracy = 0.01f; // Погрешность вычислений - эпсилон. Данный параметр мутабельный
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
