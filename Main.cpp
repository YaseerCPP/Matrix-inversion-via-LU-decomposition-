#include <iostream>
#include <vector>
#include <iomanip>

// Function to perform LU Decomposition
bool luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    int n = A.size();
    L = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    U = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        // Upper triangular matrix U
        for (int k = i; k < n; ++k) {
            double sum = 0;
            for (int j = 0; j < i; ++j) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }

        // Lower triangular matrix L
        for (int k = i; k < n; ++k) {
            if (i == k)
                L[i][i] = 1; // Diagonal as 1
            else {
                double sum = 0;
                for (int j = 0; j < i; ++j) {
                    sum += (L[k][j] * U[j][i]);
                }
                if (U[i][i] == 0)
                    return false; // Singular matrix
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
    return true;
}

// Function to solve the system L*Y = B using forward substitution
std::vector<double> forwardSubstitution(const std::vector<std::vector<double>>& L, const std::vector<double>& B) {
    int n = L.size();
    std::vector<double> Y(n, 0);
    for (int i = 0; i < n; ++i) {
        Y[i] = B[i];
        for (int j = 0; j < i; ++j) {
            Y[i] -= L[i][j] * Y[j];
        }
        Y[i] /= L[i][i];
    }
    return Y;
}

// Function to solve the system U*X = Y using back substitution
std::vector<double> backSubstitution(const std::vector<std::vector<double>>& U, const std::vector<double>& Y) {
    int n = U.size();
    std::vector<double> X(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        X[i] = Y[i];
        for (int j = i + 1; j < n; ++j) {
            X[i] -= U[i][j] * X[j];
        }
        X[i] /= U[i][i];
    }
    return X;
}

// Function to calculate the inverse of a matrix using LU Decomposition
bool inverseMatrix(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& invA) {
    int n = A.size();
    std::vector<std::vector<double>> L, U;

    if (!luDecomposition(A, L, U))
        return false; // Singular matrix

    invA = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));

    // Solve A * X = I using LU decomposition
    for (int i = 0; i < n; ++i) {
        std::vector<double> e(n, 0);
        e[i] = 1;
        std::vector<double> Y = forwardSubstitution(L, e);
        std::vector<double> X = backSubstitution(U, Y);
        for (int j = 0; j < n; ++j) {
            invA[j][i] = X[j];
        }
    }
    return true;
}

// Function to print the matrix
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example matrix (3x3)
    std::vector<std::vector<double>> A = {
        {4, 3, 2},
        {3, 7, 1},
        {2, 5, 3}
    };

    std::cout << "Original Matrix:" << std::endl;
    printMatrix(A);

    std::vector<std::vector<double>> invA;
    if (inverseMatrix(A, invA)) {
        std::cout << "Inverse Matrix:" << std::endl;
        printMatrix(invA);
    } else {
        std::cout << "Matrix is singular and cannot be inverted." << std::endl;
    }

    return 0;
}
