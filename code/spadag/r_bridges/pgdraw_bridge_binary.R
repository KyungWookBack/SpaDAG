#!/usr/bin/env Rscript
# Persistent bridge serving PG(1, z) draws from BayesLogit::rpg over stdin/stdout.
#
# Little-endian binary protocol with signed 32-bit integer headers. R on
# Windows cannot open stdout as a binary connection, so with the optional
# `hex-output` argument each response is written as one hexadecimal line.
#   request 1 (DRAW):      int32 n, float64[n] tilts
#   response:              int32 status=0, int32 n, float64[n] draws
#   request 2 (GET_STATE): no payload
#   response:              int32 status=0, int32 n, int32[n] RNG state
#   request 3 (SET_STATE): int32 n, int32[n] RNG state
#   response:              int32 status=0, int32 n=0
#   request 4 (QUIT):      no response

local_lib <- path.expand("~/Rlibs")
.libPaths(c(local_lib, .libPaths()))
suppressMessages(library(BayesLogit))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1L) {
  seed <- as.integer(args[1])
  if (!is.na(seed)) {
    set.seed(seed)
  }
}
hex_output <- length(args) >= 2L && identical(args[2L], "hex-output")

input <- suppressWarnings(file("stdin", "rb"))
output <- if (hex_output) stdout() else suppressWarnings(file("/dev/stdout", "wb"))

read_int <- function(n = 1L) {
  readBin(input, integer(), n = n, size = 4L, endian = "little")
}

write_response <- function(n, payload = raw()) {
  header <- writeBin(
    as.integer(c(0L, as.integer(n))), raw(), size = 4L, endian = "little"
  )
  response <- c(header, payload)
  if (hex_output) {
    cat(paste(sprintf("%02x", as.integer(response)), collapse = ""), "\n", sep = "")
  } else {
    writeBin(response, output)
  }
  flush(output)
}

repeat {
  opcode <- read_int()
  if (length(opcode) == 0L || opcode == 4L) {
    break
  }

  if (opcode == 1L) {
    n <- read_int()
    if (length(n) != 1L || is.na(n) || n < 0L) {
      quit(status = 2L)
    }
    z <- readBin(input, double(), n = n, size = 8L, endian = "little")
    if (length(z) != n || anyNA(z)) {
      quit(status = 3L)
    }
    draws <- BayesLogit::rpg(num = n, h = 1, z = z)
    payload <- writeBin(
      as.double(draws), raw(), size = 8L, endian = "little"
    )
    write_response(n, payload)
    next
  }

  if (opcode == 2L) {
    if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      runif(1L)
    }
    state <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    payload <- writeBin(
      as.integer(state), raw(), size = 4L, endian = "little"
    )
    write_response(length(state), payload)
    next
  }

  if (opcode == 3L) {
    n <- read_int()
    if (length(n) != 1L || is.na(n) || n <= 0L) {
      quit(status = 4L)
    }
    state <- read_int(n)
    if (length(state) != n || anyNA(state)) {
      quit(status = 5L)
    }
    assign(".Random.seed", as.integer(state), envir = .GlobalEnv)
    write_response(0L)
    next
  }

  quit(status = 6L)
}
