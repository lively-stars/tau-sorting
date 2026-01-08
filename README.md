# Readme

This repository contains the source code and documentation for tau-sorting.

## Data files

├── G2_1D.dat                  - 1D atmospheric model data (height - ascending, density, pressure, temperature)
├── Makefile                   - Makefile to compile the c version tau-sorting code
├── ODF_nc_format.nc           - ODF data in netCDF format (p, T, n_bins, n_subbins)
├── continuumabs.dat           - Continuum absorption data (p, T, n_bins)
├── continuumscat.dat          - Continuum scattering data
├── continuumall.dat           - Continuum absorption + scattering data
├── diff_binning               - ignored directory
│   ├── global_tau.h_12bins
│   ├── global_tau.h_15bin
│   ├── global_tau.h_2bins
│   ├── global_tau.h_4bins
│   └── global_tau.h_grey
├── global_tau.h               - Header file with global variables for tau-sorting
├── p00big3.bdf                - Line absorption data file    
└── tausort.c                  - C source code for tau-sorting


❯ ncdump -h ODF_nc_format.nc
netcdf ODF_nc_format {
dimensions:
	np = 150 ;                            - number of pressure points
	nt = 300 ;                            - number of temperature points               
	nbins = 328 ;                         - number of lambda bins
	nsubbins = 12 ;                       - number of sub-bins per lambda bin
	numfp = 329 ;                         - number of lambda edges (lambda!!)
variables:
	short ODF(nt, np, nbins, nsubbins) ;  - ODF values as short integers - float value = 10^(ODF/1000)
	double FreqG(numfp) ;
	double P(np) ;
	double T(nt) ;
	double subbin(nbins, nsubbins) ;

// global attributes:
		:vturb = 2. ;
}

## Python implementation

### Overview

1. Inputs:
    - ODF_nc_format.nc - kappa (T, p, N_b, N_s)
    - continuumall.dat - continuum opacity (T, p, N_b)
    - G2_1D.dat - atmospheric model (height, rho, p, T)
2. Calculate reference kappa (rosseland, 500nm...) as
    $$ \kappa_\text{all}(T,p) = f(\kappa_{ODF} + \kappa_{cont}) $$
    $$ \kappa_\text{ross} = \frac{\integrate_0^\inf\kappa_\text{all} \frac{dB_\lambda}{dT} d\lambda}{\integrate_0^\inf \frac{dB_\lambda}{dT} d\lambda} $$
3. Interpolate kappa calcualted on the T,p grid from ODFs to the T,p grid of the atmospheric model