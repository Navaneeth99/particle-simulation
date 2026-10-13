#include "common.h"
#include <omp.h>
#include <vector>
#include <iostream>
#include <cmath>

// Put any static global variables here that you will use throughout the simulation.
static int m;
static std::vector<std::vector<int>> grid;
static std::vector<omp_lock_t> locks;

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
	double s = cutoff;
    m = std::ceil(size/s);
    grid.resize((m+1)*(m+1));
	locks.resize((m+1)*(m+1));
	for (int i = 0; i < (m+1)*(m+1); ++i)
        omp_init_lock(&locks[i]);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Write this function
	double s = cutoff;
    long x_cell, y_cell, c;
	
    #pragma omp parallel private(x_cell, y_cell, c)
	{
		#pragma omp for
		for (int i = 0; i< num_parts; i++)
		{
			x_cell = std::floor(parts[i].x / s);
			y_cell = std::floor(parts[i].y / s);
			c = x_cell*(m+1) + y_cell;

			omp_set_lock(&locks[c]);
			grid[c].push_back(i);
			omp_unset_lock(&locks[c]);
		}
	}

	#pragma omp parallel
	{
		int i, j, pi, pj;
		#pragma omp for
		for ( i=0; i < (m+1); i++)
		{
			for( j=0; j < (m+1); j++)
			{
				for (int gi : grid[(i)*(m+1) + (j)])
				{
					pi = 1;
					parts[gi].ax = parts[gi].ay = 0;
					while(pi >= -1)
					{   
						pj = 1;
						while(pj >= -1)
						{
							if (((j+pj) >= 0) &&((j+pj) < (m+1)) && ((i+pi) >= 0) && ((i+pi) < (m+1)))
							{
								for (int gj : grid[(i+pi)*(m+1) + (j+pj)])
								{
									if(gi == gj) continue;
									apply_force(parts[gi], parts[gj]);
								}
							}
						--pj;
						}
					--pi;
					}
				}
			}
		}
		    // Move Particles
		#pragma omp for
		for (int i = 0; i < num_parts; ++i) 
		{
			move(parts[i], size);
		}
	}

	#pragma omp parallel
	{
		#pragma omp for
		for (auto& cell : grid) 
		{
    		cell.clear();
    	}
	}
    

}
