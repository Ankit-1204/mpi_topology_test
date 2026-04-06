#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace std;

static const int NUM_SIZES = 16;   // 16 message sizes
static const int MSG_COUNTS[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 4096, 16384, 65536, 131072, 262144
};

static const int WARMUP     = 3;
static const int ITERATIONS = 30;

static const char* CSV_FILE = "collectives1d.csv";

/* ---------- helpers ---------- */

static bool file_exists(const std::string& path)
{
    std::ifstream f(path);
    return f.good();
}

static void append_csv_row(int seq, const std::string& op,
                           long bytes, double latency_us, int nprocs)
{
    bool exists = file_exists(CSV_FILE);
    std::ofstream out(CSV_FILE, std::ios::app);
    if (!out) {
        std::cerr << "Failed to open " << CSV_FILE << " for writing\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!exists) {
        out << "run_id,seq,operation,bytes,latency_us,nprocs\n";
    }

    out << seq << ","
        << op << ","
        << bytes << ","
        << latency_us << ","
        << nprocs << "\n";
}

static double bench_bcast(int count, int rank)
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

static double bench_reduce(int count, int rank)
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

static double bench_allreduce(int count)
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
    std::vector<int> sendbuf(count * nprocs, 0);
    std::vector<int> recvbuf(count, 0);

    if (rank == 0) {
        for (int i = 0; i < count * nprocs; i++) sendbuf[i] = i;
    }

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

// Point-to-point benchmark: ring exchange with MPI_Sendrecv.
// Every rank sends to rank+1 and receives from rank-1.
static double bench_p2p_ring(int count, int rank, int nprocs)
{
    std::vector<int> sendbuf(count, rank);
    std::vector<int> recvbuf(count, -1);

    int send_to = (rank + 1) % nprocs;
    int recv_from = (rank - 1 + nprocs) % nprocs;

    for (int w = 0; w < WARMUP; w++) {
        MPI_Sendrecv(sendbuf.data(), count, MPI_INT, send_to,   100,
                     recvbuf.data(), count, MPI_INT, recv_from, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        MPI_Sendrecv(sendbuf.data(), count, MPI_INT, send_to,   100,
                     recvbuf.data(), count, MPI_INT, recv_from, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return (MPI_Wtime() - t0) * 1e6 / ITERATIONS;
}

/* ---------- main ---------- */

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    long run_id = static_cast<long>(std::time(nullptr));
    int seq = 0;

    if (rank == 0) {
        std::cout << "collectives benchmark started with " << nprocs << " ranks\n";
    }

    for (int s = 0; s < NUM_SIZES; s++) {
        int count = MSG_COUNTS[s];
        long bytes = static_cast<long>(count) * sizeof(int);

        double t_bcast     = bench_bcast(count, rank);
        double t_reduce    = bench_reduce(count, rank);
        double t_allreduce = bench_allreduce(count);
        double t_scatter   = bench_scatter(count, rank, nprocs);
        double t_alltoall  = bench_alltoall(count, rank, nprocs);

        double t_p2p_local = bench_p2p_ring(count, rank, nprocs);
        double t_p2p_avg   = 0.0;
        MPI_Reduce(&t_p2p_local, &t_p2p_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            t_p2p_avg /= nprocs;
        }

        if (rank == 0) {
            append_csv_row(seq++, "bcast",     bytes, t_bcast,     nprocs);
            append_csv_row(seq++, "reduce",    bytes, t_reduce,    nprocs);
            append_csv_row(seq++, "allreduce", bytes, t_allreduce, nprocs);
            append_csv_row(seq++, "scatter",   bytes, t_scatter,   nprocs);
            append_csv_row(seq++, "alltoall",  bytes, t_alltoall,  nprocs);
            append_csv_row(seq++, "p2p_ring",  bytes, t_p2p_avg,   nprocs);
        }
    }

    MPI_Finalize();
    return 0;
}