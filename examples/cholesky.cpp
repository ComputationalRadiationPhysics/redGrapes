/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <cblas.h>
#include <lapacke.h>

#include <iostream>

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
                    std::cout << (*A[ja * nblks + ia].getObject())[jb * blocksize + ib] << "; ";
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    size_t nblks;
    size_t blksz;
    unsigned n_threads = 1;

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

    auto rg = redGrapes::init(n_threads);

    size_t N = nblks * blksz;

    // allocate input matrix
    double* Alin = (double*) malloc(N * N * sizeof(double));

    // fill the matrix with random values
    for(size_t i = 0; i < N * N; i++)
        Alin[i] = ((double) rand()) / ((double) RAND_MAX);

    // make it positive definite
    for(size_t i = 0; i < N; i++)
        Alin[i * N + i] += N;

    // initialize tiled matrix in column-major layout
    std::vector<redGrapes::IOResource<double*>> A(nblks * nblks);

    // allocate each tile (also in column-major layout)
    for(size_t j = 0; j < nblks; ++j)
        for(size_t i = 0; i < nblks; ++i)
            A[j * nblks + i] = new double[blksz * blksz];

    /* ia: row of outer matrix
       ib: row of inner matrix
       ja: col of outer matrix
       jb: col of inner matrix */
    for(size_t ia = 0; ia < nblks; ++ia)
        for(size_t ib = 0; ib < blksz; ++ib)
            for(size_t ja = 0; ja < nblks; ++ja)
                for(size_t jb = 0; jb < blksz; ++jb)
                    (*A[ja * nblks + ia].getObject())[jb * blksz + ib]
                        = Alin[(ia * blksz + ib) + (ja * blksz + jb) * N];

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
                    [blksz](auto a, auto b, auto c)
                    {
                        spdlog::debug("dgemm");
                        cblas_dgemm(
                            CblasColMajor,
                            CblasNoTrans,
                            CblasTrans,
                            blksz,
                            blksz,
                            blksz,
                            -1.0,
                            *a,
                            blksz,
                            *b,
                            blksz,
                            1.0,
                            *c,
                            blksz);
                    },
                    A[k * nblks + i].read(),
                    A[k * nblks + j].read(),
                    A[j * nblks + i].write());
            }
        }

        for(size_t i = 0; i < j; i++)
        {
            // A[j,j] = A[j,j] - A[j,i] * (A[j,i])^t
            rg.emplace_task(
                [blksz](auto a, auto c)
                {
                    spdlog::debug("dsyrk");
                    cblas_dsyrk(
                        CblasColMajor,
                        CblasLower,
                        CblasNoTrans,
                        blksz,
                        blksz,
                        -1.0,
                        *a,
                        blksz,
                        1.0,
                        *c,
                        blksz);
                },
                A[i * nblks + j].read(),
                A[j * nblks + j].write());
        }

        // Cholesky Factorization of A[j,j]
        rg.emplace_task(
            [blksz](auto a)
            {
                spdlog::debug("dpotrf");
                LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', blksz, *a, blksz);
            },
            A[j * nblks + j].write());

        for(size_t i = j + 1; i < nblks; i++)
        {
            // A[i,j] <- A[i,j] = X * (A[j,j])^t
            rg.emplace_task(
                [blksz](auto a, auto b)
                {
                    spdlog::debug("dtrsm");
                    cblas_dtrsm(
                        CblasColMajor,
                        CblasRight,
                        CblasLower,
                        CblasTrans,
                        CblasNonUnit,
                        blksz,
                        blksz,
                        1.0,
                        *a,
                        blksz,
                        *b,
                        blksz);
                },
                A[j * nblks + j].read(),
                A[j * nblks + i].write());
        }
    }

    print_matrix(A, nblks, blksz);

    return 0;
}
