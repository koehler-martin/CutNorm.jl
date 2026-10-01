# CutNorm.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://koehler-martin.github.io/CutNorm.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://koehler-martin.github.io/CutNorm.jl/dev)
[![Test workflow status](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/koehler-martin/CutNorm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/koehler-martin/CutNorm.jl)
[![Docs workflow Status](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

A Julia package for computing the **cut norm** of matrices using multistart nonlinear optimization, a parallel tabu search, exact brute-force enumeration, or exact optimization-based formulations solved with [JuMP](https://jump.dev/).

Given a matrix $A \in \mathbb{R}^{m \times n}$, the cut norm is defined as

$$\lVert A\rVert_{\square} = \max_{S \subseteq [m], T \subseteq [n]} \left| \sum_{i \in S, j \in T} A_{ij} \right|.$$

## Installation

```julia
using Pkg
Pkg.add("CutNorm")
```

**Requires Julia 1.10 or later.**

## Quick Start

```julia
using CutNorm

A = [ 1.0  1  1  1  1;
      1    1  1  1  1;
      1    1  1  1  1;
     -1   -1 -1 -1 -1;
     -1   -1 -1 -1 -1]

sol = cutnorm(A)

sol.value   # cut norm estimate
sol.S       # optimal row indicator
sol.T       # optimal column indicator
```

## Examples Showing Different Methods

This package provides two **multistart heuristics** that repeatedly solve a relaxation of a binary bilinear formulation of the cut norm: `MultistartSigned{S}` and `MultistartAugmented{S}`.
The following subsolvers `S` are available: `AlternatingLinearSearch`, `TronSolver`, and `GreedySolver`.

For **large matrices**, `TabuSearch` works on the row and column sets directly: each iteration adds or removes the best row or column that is not tabu, at a cost of only $O(m + n)$ per iteration, and independent searches run in parallel on all Julia threads (start Julia with `julia -t auto`).
The search runs until a limit is reached, by default for 10 seconds.

**Exact solutions** can be computed via `BruteForce` or with JuMP, using a solver compatible with the chosen problem formulation: `INLP` (Integer Nonlinear Program), `ILP` (Integer Linear Program), and `QUBO` (Quadratic Unconstrained Binary Optimization Problem).

```julia
# Default: multistart, signed method with alternating linear search
sol = cutnorm(A)

# Trust-region Newton with the signed method
sol = cutnorm(A; method=MultistartSigned{TronSolver}())

# Greedy solver, scaled by m*n, limited to 500 restarts
sol = cutnorm(A; method=MultistartAugmented{GreedySolver}(),
              scaled=true, max_restarts=500)

# Parallel tabu search, for large matrices
sol = cutnorm(A; method=TabuSearch())                          # 10 s on all threads
sol = cutnorm(A; method=TabuSearch(), max_time=300.0)          # larger time budget
sol = cutnorm(A; method=TabuSearch(), max_iter=10^6, seed=1)   # reproducible

# Exact brute force (small matrices only)
sol = cutnorm(A; method=BruteForce())

# Exact methods via JuMP (load a solver of your choice first)
using HiGHS
sol = cutnorm(A; method=ILP(HiGHS.Optimizer))      # MILP — HiGHS is enough

using Gurobi
sol = cutnorm(A; method=INLP(Gurobi.Optimizer))      # integer nonlinear program
sol = cutnorm(A; method=QUBO(Gurobi.Optimizer))      # binary-quadratic program

# Time-limited exact solve (returns the best solution found so far)
sol = cutnorm(A; method=QUBO(Gurobi.Optimizer), max_time=20.0)
```

## Options

All keyword arguments are forwarded to the respective settings type:

| Option | Default | Description |
|---|---|---|
| `max_restarts::Int` | `1000` | Maximum number of restarts (multistart heuristics only) |
| `max_iter::Int` | `typemax(Int)` | Iteration limit of each search (tabu search only) |
| `ntasks::Int` | `Threads.nthreads()` | Number of parallel searches (tabu search only) |
| `max_time::Float64` | `3600.0` | Wall-clock time limit in seconds (`10.0` for the tabu search) |
| `scaled::Bool` | `false` | Divide the result by $m \cdot n$ |
| `print_level::Int` | `0` | Verbosity of solver output |

For more options, see the documentation.

## Solution Object

`cutnorm` returns a solution object whose type depends on the method.
The most relevant fields are:

| Field | Description |
|---|---|
| `value` | Best cut norm estimate found |
| `S`, `T` | Optimal row/column indicator vectors |
| `runtime` | Total elapsed time in seconds |
| `dims` | Problem dimensions $(m, n)$ |

More details can be found in the documentation.

## Author and Acknowledgements

This package was written by [Martin Köhler](https://www.tu-braunschweig.de/en/mo/team/koehler) at TU Braunschweig.

Funded by the European Union through ERC Consolidator Grant SCARCE ([101087662](https://cordis.europa.eu/project/id/101087662)).

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

**Note:** Parts of this package's code and documentation were written with AI assistance (Anthropic Claude) and subsequently reviewed by a human.

## References
Frieze, A., Kannan, R. *Quick Approximation to Matrices and Applications*. Combinatorica 19, 175–220 (1999). https://doi.org/10.1007/s004930050052

Alon, N., Naor, A. *Approximating the Cut-Norm via Grothendieck's Inequality*. SIAM Journal on Computing, vol. 35, no. 4, pp. 787–803, 2006. https://doi.org/10.1137/S0097539704441629
