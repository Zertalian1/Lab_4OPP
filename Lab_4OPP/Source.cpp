#include <iostream>
#include <cstring>
#include <cmath>
#include "mpi.h"

#define EPSILON 1e-6
#define MAX_DIFF_DELTA 1e2
#define A 1e8
#define START_VALUE_N 80

#define X0 -1.0
#define Y0 -1.0
#define Z0 -1.0

#define DX 2.0
#define DY 2.0
#define DZ 2.0

using namespace std;

void calculateNodeCoords(int i, int j, int k, long double& x, long double& y, long double& z, const long double& Hx, const long double& Hy, const long double& Hz) {
    x = X0 + i * Hx;
    y = Y0 + j * Hy;
    z = Z0 + k * Hz;
}

long double F(long double x, long double y, long double z) {
    return (x * x + y * y + z * z);
}

void startValues(long double* phiSolid, int Nx, int Ny, int Nz, const long double& Hx, const long double& Hy, const long double& Hz) {
    memset(phiSolid, 0, Nx * Ny * Nz);

    long double x, y, z;

    for (int s = 0; s < 2; ++s) {
        // phi(xi, yj, 1), phi(xi, yj, -1)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                calculateNodeCoords(i, j, s * (Nz - 1), x, y, z, Hx, Hy, Hz);
                phiSolid[i + j * Nx + s * (Nz - 1) * Nx * Ny] = F(x, y, z);
            }
        }

        // phi(xi, 1, zk), phi(xi, -1, zk)
        for (int k = 0; k < Nz; ++k) {
            for (int i = 0; i < Nx; ++i) {
                calculateNodeCoords(i, s * (Ny - 1), k, x, y, z, Hx, Hy, Hz);
                phiSolid[i + s * (Ny - 1) * Nx + k * Nx * Ny] = F(x, y, z);
            }
        }

        // phi(1, yj, zk), phi(-1, yj, zk)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                calculateNodeCoords(s * (Nx - 1), j, k, x, y, z, Hx, Hy, Hz);
                phiSolid[s * (Nx - 1) + j * Nx + k * Nx * Ny] = F(x, y, z);
            }
        }
    }

}


void cutStartPhiValuesMatrix(long double** phiValuesPart, long double* phiValuesSolid, const int& NxPart, const int& NyPart, const int& NzPart) {
    MPI_Scatter(phiValuesSolid, NxPart * NyPart * NzPart, MPI_LONG_DOUBLE, phiValuesPart[0], NxPart * NyPart * NzPart, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    memcpy(phiValuesPart[1], phiValuesPart[0], NxPart * NyPart * NzPart * sizeof(long double));
}

void getBoundaries(long double** boundaries, int procRankRecv, int count, MPI_Request* reqr, int procCount) {
    if (procRankRecv != 0) {
        MPI_Irecv(boundaries[0], count, MPI_LONG_DOUBLE, procRankRecv - 1, 0, MPI_COMM_WORLD, &reqr[1]);
    }

    if (procRankRecv != procCount - 1) {
        MPI_Irecv(boundaries[1], count, MPI_LONG_DOUBLE, procRankRecv + 1, 0, MPI_COMM_WORLD, &reqr[0]);
    }
}

void sendBoundaries(long double** phiValuesPart, int const* boundariesOffset, int procRankSend, int count, MPI_Request* reqs, int procCount) {
    if (procRankSend != 0) {
        MPI_Isend(phiValuesPart[0] + boundariesOffset[0], count, MPI_LONG_DOUBLE, procRankSend - 1, 0, MPI_COMM_WORLD, &reqs[0]);
    }

    if (procRankSend != procCount - 1) {
        MPI_Isend(phiValuesPart[0] + boundariesOffset[1], count, MPI_LONG_DOUBLE, procRankSend + 1, 0, MPI_COMM_WORLD, &reqs[1]);
    }
}

long double rho(long double phiValue) {
    return (6.0 - A * phiValue);
}

void calculateMaxDiffLocalAndDeltaLocal(long double** phiValuesPart, long double precisePhiValue, int NxPart, int NyPart,
    long double& maxDiffLocal, long double& deltaLocal, int i, int j, int k) {

    if (abs(phiValuesPart[1][i + j * NxPart + k * NxPart * NyPart] - phiValuesPart[0][i + j * NxPart + k * NxPart * NyPart]) > maxDiffLocal) {
        maxDiffLocal = abs(phiValuesPart[1][i + j * NxPart + k * NxPart * NyPart] - phiValuesPart[0][i + j * NxPart + k * NxPart * NyPart]);
    }
    if (abs(phiValuesPart[0][i + j * NxPart + k * NxPart * NyPart] - precisePhiValue) > deltaLocal) {
        deltaLocal = abs(phiValuesPart[0][i + j * NxPart + k * NxPart * NyPart] - precisePhiValue);
    }

}

void waitMessages(MPI_Request* reqr, MPI_Request* reqs, int procRankSend, int procCount) {
    if (procRankSend != 0) {
        MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
    }
    if (procRankSend != procCount - 1) {
        MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        MPI_Wait(&reqr[0], MPI_STATUS_IGNORE);
    }
}

void calculateBoundaries(long double** phiValuesPart, int NxPart, int NyPart, int Nz, const long double& Hx, const long double& Hy, const long double& Hz,
    long double& maxDiffLocal, long double& deltaLocal, long double** boundaries, int procRank, int procCount, int* H, int* L) {

    long double a[4], b[4];
    long double precisePhiValue[2];
    long double x, y, z;
    long double divisor = (2.0 / (Hx * Hx) + 2.0 / (Hy * Hy) + 2.0 / (Hz * Hz) + A);

    for (int j = 1; j < NyPart - 1; ++j) {
        for (int i = 1; i < NxPart - 1; ++i) {
            if (procRank != 0) { // считаем низ
                a[0] = phiValuesPart[0][(i + 1) + j * NxPart + H[0] * NxPart * NyPart] + phiValuesPart[0][(i - 1) + j * NxPart + H[0] * NxPart * NyPart];
                a[1] = phiValuesPart[0][i + (j + 1) * NxPart + H[0] * NxPart * NyPart] + phiValuesPart[0][i + (j - 1) * NxPart + H[0] * NxPart * NyPart];
                a[2] = phiValuesPart[0][i + j * NxPart + (H[0] + 1) * NxPart * NyPart] + boundaries[0][i + j * NxPart];

                calculateNodeCoords(i, j, L[0], x, y, z, Hx, Hy, Hz);
                precisePhiValue[0] = F(x, y, z);
                a[3] = -rho(precisePhiValue[0]);

                phiValuesPart[1][i + j * NxPart + H[0] * NxPart * NyPart] = (a[0] / (Hx * Hx) + a[1] / (Hy * Hy) + a[2] / (Hz * Hz) + a[3]) / divisor;

                calculateMaxDiffLocalAndDeltaLocal(phiValuesPart, precisePhiValue[0], NxPart, NyPart, maxDiffLocal, deltaLocal, i, j, H[0]);
            }

            if (procRank != procCount - 1) { // считаем верх
                b[0] = phiValuesPart[0][(i + 1) + j * NxPart + H[1] * NxPart * NyPart] + phiValuesPart[0][(i - 1) + j * NxPart + H[1] * NxPart * NyPart];
                b[1] = phiValuesPart[0][i + (j + 1) * NxPart + H[1] * NxPart * NyPart] + phiValuesPart[0][i + (j - 1) * NxPart + H[1] * NxPart * NyPart];
                b[2] = boundaries[1][i + j * NxPart] + phiValuesPart[0][i + j * NxPart + (H[1] - 1) * NxPart * NyPart];

                calculateNodeCoords(i, j, L[1], x, y, z, Hx, Hy, Hz);
                precisePhiValue[1] = F(x, y, z);
                b[3] = -rho(precisePhiValue[1]);

                phiValuesPart[1][i + j * NxPart + H[1] * NxPart * NyPart] = (b[0] / (Hx * Hx) + b[1] / (Hy * Hy) + b[2] / (Hz * Hz) + b[3]) / divisor;

                calculateMaxDiffLocalAndDeltaLocal(phiValuesPart, precisePhiValue[1], NxPart, NyPart, maxDiffLocal, deltaLocal, i, j, H[1]);
            }

        }
    }

}

void calculateMPlusOnePhiValue(long double** phiValuesPart, int NxPart, int NyPart, int NzPart, const long double& Hx, const long double& Hy, const long double& Hz,
    long double& maxDiffLocal, long double& deltaLocal, long double** boundaries, int procRank, int procCount, int Nz) {
    
    int NzAddition = NzPart % 2;
    int medValZ = NzPart / 2;
    int K[2];

    MPI_Request reqs[2] = { 0, 0 };
    MPI_Request reqr[2] = { 0, 0 };

    long double a[4]; // сохраняем промежуточные значения phi[M]_{i+?, j+?, k+?}
    long double b[4];
    long double precisePhiValue[2];

    maxDiffLocal = 0;
    deltaLocal = 0;

    long double divisor = (2.0 / (Hx * Hx) + 2.0 / (Hy * Hy) + 2.0 / (Hz * Hz) + A);

    long double x, y, z;

    // границы 0 - верх , 1 - низ
    int H[2] = { 0, NzPart - 1 };                                                               // границы
    int L[2] = { procRank * NzPart, (procRank + 1) * NzPart - 1 };                              // сдвиги
    int boundariesOffsetS[2] = { 0, (NzPart - 1) * NxPart * NyPart };

    getBoundaries(boundaries, procRank, NxPart * NyPart, reqr, procCount);
    sendBoundaries(phiValuesPart, boundariesOffsetS, procRank, NxPart * NyPart, reqs, procCount);

    for (int k = 0; k < medValZ + NzAddition - 1; ++k) { // идем из центра вниз и из цкнтра вверх
        for (int j = 1; j < NyPart - 1; ++j) {
            for (int i = 1; i < NxPart - 1; ++i) {

                K[0] = medValZ + k; // сдвиг для движения вверх
                K[1] = medValZ - 1 + NzAddition - k; // сдвиг для движения вниз

                // phi[M]_{i+1, j, k} + phi[M]_{i-1, j, k}
                a[0] = phiValuesPart[0][(i + 1) + j * NxPart + K[0] * NxPart * NyPart] + phiValuesPart[0][(i - 1) + j * NxPart + K[0] * NxPart * NyPart];

                // phi[M]_{i, j+1, k} + phi[M]_{i, j-1, k}
                a[1] = phiValuesPart[0][i + (j + 1) * NxPart + K[0] * NxPart * NyPart] + phiValuesPart[0][i + (j - 1) * NxPart + K[0] * NxPart * NyPart];

                // phi[M]_{i, j, k+1} + phi[M]_{i, j, k-1}
                a[2] = phiValuesPart[0][i + j * NxPart + (K[0] + 1) * NxPart * NyPart] + phiValuesPart[0][i + j * NxPart + (K[0] - 1) * NxPart * NyPart];

                // rho_{i, j, k}
                calculateNodeCoords(i, j, K[0], x, y, z, Hx, Hy, Hz);
                precisePhiValue[0] = F(x, y, z);
                a[3] = -rho(precisePhiValue[0]);

                // phi[M+1]_{i, j, k}
                phiValuesPart[1][i + j * NxPart + K[0] * NxPart * NyPart] = (a[0] / (Hx * Hx) + a[1] / (Hy * Hy) + a[2] / (Hz * Hz) + a[3]) / divisor;

                calculateMaxDiffLocalAndDeltaLocal(phiValuesPart, precisePhiValue[0], NxPart, NyPart, maxDiffLocal, deltaLocal, i, j, K[0]);

                // phi[M]_{i+1, j, k} + phi[M]_{i-1, j, k}
                b[0] = phiValuesPart[0][(i + 1) + j * NxPart + K[1] * NxPart * NyPart] + phiValuesPart[0][(i - 1) + j * NxPart + K[1] * NxPart * NyPart];

                // phi[M]_{i, j+1, k} + phi[M]_{i, j-1, k}
                b[1] = phiValuesPart[0][i + (j + 1) * NxPart + K[1] * NxPart * NyPart] + phiValuesPart[0][i + (j - 1) * NxPart + K[1] * NxPart * NyPart];

                // phi[M]_{i, j, k+1} + phi[M]_{i, j, k-1}
                b[2] = phiValuesPart[0][i + j * NxPart + (K[1] + 1) * NxPart * NyPart] + phiValuesPart[0][i + j * NxPart + (K[1] - 1) * NxPart * NyPart];

                // rho_{i, j, k}
                calculateNodeCoords(i, j, K[1], x, y, z, Hx, Hy, Hz);
                precisePhiValue[1] = F(x, y, z);
                b[3] = -rho(precisePhiValue[1]);

                // phi[M+1]_{i, j, k}
                phiValuesPart[1][i + j * NxPart + K[1] * NxPart * NyPart] = (b[0] / (Hx * Hx) + b[1] / (Hy * Hy) + b[2] / (Hz * Hz) + b[3]) / divisor;

                calculateMaxDiffLocalAndDeltaLocal(phiValuesPart, precisePhiValue[1], NxPart, NyPart, maxDiffLocal, deltaLocal, i, j, K[1]);
            
            }
        }
    }

    waitMessages(reqr, reqs, procRank, procCount);  // подождём границы
    if (procCount != 1) {
        calculateBoundaries(phiValuesPart, NxPart, NyPart, Nz, Hx, Hy, Hz, maxDiffLocal, deltaLocal, boundaries, procRank, procCount, H, L); // теперь можно посчитать края
    }

    memcpy(phiValuesPart[0], phiValuesPart[1], NxPart * NyPart * NzPart * sizeof(long double));
}

void printMaxTimeTaken(double& startTime, double& endTime, int& procRank) {
    double minimalStartTime;
    double maximumEndTime;
    MPI_Reduce(&endTime, &maximumEndTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&startTime, &minimalStartTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (procRank == 0) {
        std::cout << "Time taken: " << maximumEndTime - minimalStartTime;
    }
}



int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);

    long double* phiValuesPart[2];
    long double* boundaries[2]; // 0 - upper boundary, 1 - lower boundary
    long double* phiValuesSolid = NULL;

    long double maxDiff;
    long double delta;

    int procCount;
    int procRank;

    double startTime = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &procCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    
    int Nx = START_VALUE_N;         // размеры поля
    int Ny = START_VALUE_N;
    int Nz = START_VALUE_N;

    if (Nz % procCount != 0) {
        cout << "Nz % processes count != 0. Change Ni or processes count\n";
        return 1;
    }

    int NxPart = Nx;
    int NyPart = Ny;
    int NzPart = Nz / procCount;

    int iterationsCount = 0;

    long double Hx = DX / ((long double)Nx - 1.0);
    long double Hy = DY / ((long double)Ny - 1.0);
    long double Hz = DZ / ((long double)Nz - 1.0);

    long double maxDiffLocal;
    long double deltaLocal;

    for (int i = 0; i < 2; ++i) {
        boundaries[i] = new long double[NyPart * NxPart];               
        phiValuesPart[i] = new long double[NzPart * NyPart * NxPart];   // 1 новые значения, 0 старые значения
    }
    int jek = 0;
    
    if (procRank == 0) {
        phiValuesSolid = new long double[Nx * Ny * Nz];
        startValues(phiValuesSolid, Nx, Ny, Nz, Hx, Hy, Hz);
        
        jek = 1;
        
    }
    cutStartPhiValuesMatrix(phiValuesPart, phiValuesSolid, NxPart, NyPart, NzPart);
    
   
    do {
        calculateMPlusOnePhiValue(phiValuesPart, NxPart, NyPart, NzPart, Hx, Hy, Hz, maxDiffLocal, deltaLocal, boundaries, procRank, procCount, Nz);

        MPI_Allreduce(&maxDiffLocal, &maxDiff, 1, MPI_LONG_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&deltaLocal, &delta, 1, MPI_LONG_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (procRank == 0) {
            iterationsCount++;
        }

        if (maxDiff < EPSILON) {
            break;
        }

    } while (true);

    if (procRank == 0) {

        if (delta > MAX_DIFF_DELTA) {
            cout << "delta is bigger\n";
        }
        else {
            cout << "Function is found" << endl;
        }

        cout << "Iterations: " << iterationsCount << endl;

    }

    double endTime = MPI_Wtime();

    printMaxTimeTaken(startTime, endTime, procRank);
    for (int i = 0; i < 2; ++i) {
        delete[] boundaries[i];
        delete[] phiValuesPart[i];
    }

    if (procRank == 0) {
        delete[] phiValuesSolid;
    }

    MPI_Finalize();

}
// переписать коменты на английском