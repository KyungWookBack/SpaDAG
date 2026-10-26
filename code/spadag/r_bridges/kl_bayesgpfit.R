#!/usr/bin/env Rscript
# Closed-form Karhunen-Loeve eigenpairs of the modified squared-exponential
# (MSE) kernel of Kang, Reich & Staicu (2018), Biometrika 105(1):165-184,
#   kappa(v,v') = tau^2 exp{-a(||v||^2+||v'||^2) - b||v-v'||^2},
# evaluated at mesh nodes with BayesGPfit. Called by spadag.spatial_gp.
#
# Usage:
#   Rscript kl_bayesgpfit.R <points_csv> <a> <b> <poly_degree> <max_range> \
#                           <out_phi_csv> <out_lambda_csv>
#   points_csv : H x d coordinates (no header).
# Writes:
#   out_phi_csv    : H x M matrix, phi_l(s_h)   (M = choose(poly_degree+d, d)).
#   out_lambda_csv : M x 1 eigenvalues lambda_l (same column order as phi).
# spatial_gp.build_kl_design forms A = phi * sqrt(lambda), sorts by lambda
# and truncates to L.

local_lib <- path.expand("~/Rlibs")
.libPaths(c(local_lib, .libPaths()))
suppressMessages(library(BayesGPfit))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop("expected 7 args: points_csv a b poly_degree max_range out_phi out_lambda")
}
pts_csv     <- args[1]
a           <- as.numeric(args[2])
b           <- as.numeric(args[3])
poly_degree <- as.integer(args[4])
max_range   <- as.numeric(args[5])
out_phi     <- args[6]
out_lambda  <- args[7]

pts <- as.matrix(read.csv(pts_csv, header = FALSE))
storage.mode(pts) <- "double"
d <- ncol(pts)

# Centre and scale the coordinates to the range expected by the eigenfunctions.
g <- GP.std.grids(pts, max_range = max_range)

phi    <- GP.eigen.funcs.fast(g, poly_degree = poly_degree, a = a, b = b)  # H x M
lambda <- GP.eigen.value(poly_degree = poly_degree, a = a, b = b, d = d)    # M

write.table(phi,    out_phi,    row.names = FALSE, col.names = FALSE, sep = ",")
write.table(lambda, out_lambda, row.names = FALSE, col.names = FALSE, sep = ",")
