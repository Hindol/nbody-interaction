#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sys/time.h>

//#define BODYMAX (2048)
#define BODYMAX (1024)
#define GFORCE 4.
#define TIMESTEP 0.01
#define MINDIST (0.00001)
#define STEPLIMIT (1000.*TIMESTEP)

using namespace std;

timespec timediff(const timespec &begin, const timespec &end)
{
    timespec diff;
    if ((end.tv_nsec - begin.tv_nsec) < 0)
    {
        diff.tv_sec = end.tv_sec - begin.tv_sec - 1;
        diff.tv_nsec = 1000000000 + end.tv_nsec - begin.tv_nsec;
    } else {
        diff.tv_sec = end.tv_sec - begin.tv_sec;
        diff.tv_nsec = end.tv_nsec - begin.tv_nsec;
    }
    return diff;
}

struct bodytype {
    double pos[3];
    double vel[3];
    double acc[3];
    double mass;
} body[BODYMAX];

double rrange(double low, double high)
{
    return (double)rand()*(high - low)/(double)RAND_MAX + low;
}

void startBodies(int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int j;

        // Three different for loops with the same start and end values of j are
        //  combined into a single loop
        for (j = 0; j < 3; ++j)
        {
            body[i].pos[j] = rrange(-50000., 50000.);
            body[i].vel[j] = rrange(-100., 100.);
            body[i].acc[j] = 0.;
        }

        body[i].mass = rrange(10., 500.);
    }
}

/**
 * Note: This function writes to body acceleration values but never uses values from it
 *  except for body mass which is a constant. So this function should be parallelizable.
 */
void addAcc(int i, int j) {

    // Compute the force between bodies and apply to each as an acceleration

    // compute the distance between them
    double dx = body[i].pos[0]-body[j].pos[0];
    double dy = body[i].pos[1]-body[j].pos[1];
    double dz = body[i].pos[2]-body[j].pos[2];

    double distsq = dx*dx + dy*dy + dz*dz;
    if (distsq < MINDIST) distsq = MINDIST;
    double dist = sqrt(distsq);

    // compute the unit vector from j to i
    double ud[3];
    ud[0] = dx/dist;
    ud[1] = dy/dist;
    ud[2] = dz/dist;

    // F = G*mi*mj/distsq, but F = ma, so ai = G*mj/distsq
    double Gdivd = GFORCE/distsq;
    double ai = Gdivd*body[j].mass;
    double aj = Gdivd*body[i].mass;

    // apply acceleration components using unit vectors
    for (int k = 0; k < 3; ++k)
    {
        // The following lines may lead to race condition, so put in critical section
        #pragma omp critical { body[j].acc[k] += aj*ud[k]; }
        #pragma omp critical { body[i].acc[k] -= ai*ud[k]; }
    }
}

void runSerialBodies(int n)
{
    // Run the simulation over a fixed range of time steps
    for (double s = 0.; s < STEPLIMIT; s += TIMESTEP) {
        int i, j;

        // Compute the accelerations of the bodies
        #pragma omp parallel for
        for (i = 0; i < n - 1; ++i)
            for (j = i + 1; j < n; ++j)
                addAcc(i, j);

        // apply accelerations and advance the bodies
        // body[i] is accessed only in the i'th loop, so this can be parallelized
        #pragma omp parallel for
        for (i = 0; i < n; ++i)
        {
            // The internal loop is unrolled to gain a tiny bit of performance
            body[i].vel[0] += body[i].acc[0] * TIMESTEP;
            body[i].pos[0] += body[i].vel[0] * TIMESTEP;
            body[i].acc[0] = 0.;

            body[i].vel[1] += body[i].acc[1] * TIMESTEP;
            body[i].pos[1] += body[i].vel[1] * TIMESTEP;
            body[i].acc[1] = 0.;

            body[i].vel[2] += body[i].acc[2] * TIMESTEP;
            body[i].pos[2] += body[i].vel[2] * TIMESTEP;
            body[i].acc[2] = 0.;
        }
    }
}

int main( int argc, char **argv)
{
    timespec begin;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

    for (int n=2; n <= BODYMAX; n <<= 1) {
        startBodies(n);
        runSerialBodies(n);
    }

    timespec end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    cout << "Total time: " << timediff(begin, end).tv_sec << " s " <<
            timediff(begin, end).tv_nsec << " ns" << endl;

    return 0;
}
