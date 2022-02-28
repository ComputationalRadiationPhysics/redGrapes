#include <iostream>
#include <cblas.h>
#include <lapacke.h>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/label.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

void print_matrix(std::vector<redGrapes::IOResource<double*>> A, int n_blocks, int blocksize);

int main(int argc, char* argv[])
{
    int nblks;
    int blksz;
    int n_threads = 1;

    if(argc >= 3)
    {
        nblks = (long) atoi(argv[1]);
        blksz = (long) atoi(argv[2]);
    }
    else
    {
        printf("usage: %s nblks blksz [n_threads]\n\n", argv[0]);
        printf("  example: %s 256 16\n", argv[0]);
        exit(0);
    }

    if(argc >= 4)
        n_threads = atoi(argv[3]);

    redGrapes::RedGrapes<redGrapes::LabelProperty> rg(n_threads);
    using TaskProperties = decltype(rg)::TaskProps;

    size_t N = nblks * blksz;

    // allocate input matrix
    double* Alin = (double*) malloc(N * N * sizeof(double));

    // fill the matrix with random values
    for(int i = 0; i < N * N; i++)
        Alin[i] = ((double) rand()) / ((double) RAND_MAX);

    // make it positive definite
    for(size_t i = 0; i < N; i++)
        Alin[i * N + i] += N;


    // initialize tiled matrix in column-major layout
    std::vector<redGrapes::IOResource<double*>> A(nblks * nblks);

    // allocate each tile (also in column-major layou)
    for(int j = 0; j < nblks; ++j)
        for(int i = 0; i < nblks; ++i)
            A[j * nblks + i] = new double[blksz * blksz];

    /* ia: row of outer matrix
       ib: row of inner matrix
       ja: col of outer matrix
       jb: col of inner matrix */
    for(int ia = 0; ia < nblks; ++ia)
        for(int ib = 0; ib < blksz; ++ib)
            for(int ja = 0; ja < nblks; ++ja)
                for(int jb = 0; jb < blksz; ++jb)
                    (*A[ja * nblks + ia])[jb * blksz + ib] = Alin[(ia * blksz + ib) + (ja * blksz + jb) * N];

    print_matrix(A, nblks, blksz);

    // calculate cholesky decomposition
    for(size_t j = 0; j < nblks; j++)
    {
        for(size_t k = 0; k < j; k++)
        {
            for(size_t i = j + 1; i < nblks; i++)
            {
                // A[i,j] = A[i,j] - A[i,k] * (A[j,k])^t
                rg.emplace_task(
                    [blksz, nblks, i, j, k](auto a, auto b, auto c)
                    {
                        spdlog::info("dgemm");
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    blksz, blksz, blksz, -1.0, *a, blksz, *b, blksz, 1.0, *c, blksz);
                    },
                    TaskProperties::Builder().label("gemm"),
                    A[k * nblks + i].read(),
                    A[k * nblks + j].read(),
                    A[j * nblks + i].write());
            }
        }

        for(size_t i = 0; i < j; i++)
        {
            // A[j,j] = A[j,j] - A[j,i] * (A[j,i])^t
            rg.emplace_task(
                [i, j, blksz, nblks](auto a, auto c)
                {
                    spdlog::info("dsyrk");
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                blksz, blksz, -1.0, *a, blksz, 1.0, *c, blksz);
                },
                A[i * nblks + j].read(),
                A[j * nblks + j].write());
        }

        // Cholesky Factorization of A[j,j]
        rg.emplace_task(
            [j, blksz, nblks](auto a)
            {
                spdlog::info("dpotrf");
                LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', blksz, *a, blksz);
            },
            A[j * nblks + j].write());

        for(size_t i = j + 1; i < nblks; i++)
        {
            // A[i,j] <- A[i,j] = X * (A[j,j])^t
            rg.emplace_task(
                [blksz, nblks, i, j](auto a, auto b)
                {
                    spdlog::info("dtrsm");
                    cblas_dtrsm(CblasColMajor,
                                CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                                blksz, blksz, 1.0, *a, blksz, *b, blksz);
                },
                A[j * nblks + j].read(),
                A[j * nblks + i].write());
        }
    }

    rg.wait_for_all();

    print_matrix(A, nblks, blksz);

    return 0;
}

void print_matrix(std::vector<redGrapes::IOResource<double*>> A, int nblks, int blocksize)
{
    for(int ia = 0; ia < nblks; ++ia)
    {
        for(int ib = 0; ib < blocksize; ++ib)
        {
            for(int ja = 0; ja < nblks; ++ja)
            {
                for(int jb = 0; jb < blocksize; ++jb)
                {
                    std::cout << (*A[ja * nblks + ia])[jb * blocksize + ib] << "; ";
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}
