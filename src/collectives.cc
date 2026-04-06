#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

static const int NUM_SIZES = 15;
// Sizes from 4 B to 1 MB (doubles as element count of int, so *4 bytes each)
static const int MSG_COUNTS[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 4096, 16384, 65536, 131072, 262144
};

static const int WARMUP     = 3;
static const int ITERATIONS = 30;

/* ---------- helpers ---------- */

static double bench_bcast(int count, int rank, int nprocs)
{
    std::vector<int> buf(count, rank);

    for (int w = 0; w < WARMUP; w++)
        MPI_Bcast(buf.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Bcast(buf.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

static double bench_reduce(int count, int rank, int nprocs)
{
    std::vector<int> send(count, 1), recv(count, 0);

    for (int w = 0; w < WARMUP; w++)
        MPI_Reduce(send.data(), recv.data(), count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Reduce(send.data(), recv.data(), count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

static double bench_allreduce(int count, int rank, int nprocs)
{
    std::vector<int> send(count, 1), recv(count, 0);

    for (int w = 0; w < WARMUP; w++)
        MPI_Allreduce(send.data(), recv.data(), count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Allreduce(send.data(), recv.data(), count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

static double bench_scatter(int count, int rank, int nprocs)
{
    // Root holds count*nprocs ints; each rank receives count ints
    std::vector<int> sendbuf(count * nprocs, 0);
    std::vector<int> recvbuf(count, 0);
    if (rank == 0)
        for (int i = 0; i < count * nprocs; i++) sendbuf[i] = i;

    for (int w = 0; w < WARMUP; w++)
        MPI_Scatter(sendbuf.data(), count, MPI_INT,
                    recvbuf.data(), count, MPI_INT,
                    0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Scatter(sendbuf.data(), count, MPI_INT,
                    recvbuf.data(), count, MPI_INT,
                    0, MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

static double bench_alltoall(int count, int rank, int nprocs)
{
    std::vector<int> sendbuf(count * nprocs, rank);
    std::vector<int> recvbuf(count * nprocs, 0);

    for (int w = 0; w < WARMUP; w++)
        MPI_Alltoall(sendbuf.data(), count, MPI_INT,
                     recvbuf.data(), count, MPI_INT,
                     MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Alltoall(sendbuf.data(), count, MPI_INT,
                     recvbuf.data(), count, MPI_INT,
                     MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

/* ---------- main ---------- */

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        printf("operation,bytes,latency_us,nprocs\n");
    }
    for (int s = 0; s < NUM_SIZES; s++) {
        int  count = MSG_COUNTS[s];
        long bytes = (long)count * sizeof(int);
        fprintf(stderr, "[rank %d/%d] entered main()\n", rank, nprocs);
        double t_bcast     = bench_bcast    (count, rank, nprocs);
        double t_reduce    = bench_reduce   (count, rank, nprocs);
        double t_allreduce = bench_allreduce(count, rank, nprocs);
        double t_scatter   = bench_scatter  (count, rank, nprocs);
        double t_alltoall  = bench_alltoall (count, rank, nprocs);
        if (rank == 0) {
            printf("collectives benchmark started with %d ranks\n", nprocs);
            fflush(stdout);
        }
        if (rank == 0) {
            printf("bcast,%ld,%.4f,%d\n",     bytes, t_bcast,     nprocs);
            printf("reduce,%ld,%.4f,%d\n",    bytes, t_reduce,    nprocs);
            printf("allreduce,%ld,%.4f,%d\n", bytes, t_allreduce, nprocs);
            printf("scatter,%ld,%.4f,%d\n",   bytes, t_scatter,   nprocs);
            printf("alltoall,%ld,%.4f,%d\n",  bytes, t_alltoall,  nprocs);
        }
        if (rank == 0) {
            std::ofstream out("collectives1d.csv");
            if (!out) {
                std::cerr << "Failed to open collectives.csv for writing\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            out << "operation,bytes,latency_us,nprocs\n";
            vector<string> header = {"bcast", "reduce", "allreduce", "scatter", "alltoall"};
            vector<vector<int>> mapv={
                { (bytes), (t_bcast), (nprocs)},
                { (bytes), (t_reduce), (nprocs)},
                {(bytes), (t_allreduce), (nprocs)},
                { (bytes), (t_scatter), (nprocs)},
                { (bytes), (t_alltoall), (nprocs)}
            };
            for (int i=0;i<header.size();i++){ {
                out << header[i] << ","
                    << mapv[i][0] << ","
                    << mapv[i][1] << ","
                    << mapv[i][2] << "\n";
            }
            }
        out.close();
    }
    }

    MPI_Finalize();
    return 0;
}
