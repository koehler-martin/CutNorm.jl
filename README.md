# CutNorm

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://koehler-martin.github.io/CutNorm.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://koehler-martin.github.io/CutNorm.jl/dev)
[![Test workflow status](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/koehler-martin/CutNorm.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)


# CutNorm.jl

A Julia package for computing the **cut norm** of matrices using multistart nonlinear optimization, exact brute-force enumeration, or exact optimization-based formulations solved with [JuMP](https://jump.dev/).

Given a matrix $A \in \mathbb{R}^{m \times n}$, the cut norm is defined as

$$\lVert A\rVert_{\square} = \max_{S \subseteq [m], T \subseteq [n]} \left| \sum_{i \in S, j \in T} A_{ij} \right|$$

CutNorm.jl relaxes the combinatorial problem to a continuous bilinear program over $[0, 1]^m \times [0, 1]^n$ and solves it with a multistart strategy using quasi-random (Sobol) initial points. It can also solve the problem **exactly** — either by brute-force enumeration, or by formulating it as an integer or quadratic program and handing it to a solver of your choice through JuMP.

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

## Methods

CutNorm.jl provides both **heuristic** (multistart) and **exact** methods.

### Heuristic Methods

| Method | Description |
|---|---|
| `MultistartSigned{S}()` | Solves both the $+$ and $-$ sign variants per restart. Two subproblems per restart but no augmentation needed. **(default)** |
| `MultistartAugmented{S}()` | Augments the matrix so that row/column sums vanish, removing the sign ambiguity. One subproblem per restart. |

The parametric type `S` denotes a specific subsolver.

### Subsolvers (Multistart only)

| Subsolver | Description |
|---|---|
| `AlternatingLinearSearch` | Fast coordinate-descent alternating between $s$ and $t$. **(default)** |
| `TronSolver` | Trust-region Newton method (from JSOSolvers.jl). Most accurate locally. |
| `GreedySolver` | Greedy single-coordinate flip with incremental gradient updates. *(not recommended)* |


### Exact Methods

| Method | Description |
|---|---|
| `BruteForce()` | Enumerates all $2^m \times 2^n$ binary vector pairs. No solver needed; exponential, so practical for small matrices only. |
| `INLP(optimizer)` | Integer **nonlinear** program: maximizes $(2\sigma - 1)\, s^\top A\, t$ over binary $s, t, \sigma$. Needs a global MINLP solver (e.g. Gurobi). |
| `ILP(optimizer)` | Integer **linear** program: McCormick linearization of the bilinear term plus a big-M absolute value. Works with any MILP solver, including HiGHS. |
| `QUBO(optimizer)` | Quadratic binary program on the augmented matrix (no sign variable needed). Needs a solver supporting (nonconvex) binary-quadratic objectives (e.g. Gurobi). |

The `INLP`, `ILP`, and `QUBO` methods build a JuMP model and solve it with the `optimizer` you pass in, so any compatible solver works — `HiGHS.Optimizer`,  `Gurobi.Optimizer`, and so on. The solver and its package are **not** dependencies of CutNorm.jl; install and load the one you want separately. Make sure it supports the model class (e.g. HiGHS handles the linear `ILP` but not the quadratic `QUBO` or the nonlinear `INLP`).

### Examples

```julia
# Default: signed method with alternating linear search
sol = cutnorm(A)

# Trust-region Newton with the signed method
sol = cutnorm(A; method=MultistartSigned{TronSolver}())

# Greedy solver, scaled by m*n, limited to 500 restarts
sol = cutnorm(A; method=MultistartAugmented{GreedySolver}(),
              scaled=true, max_restarts=500)

# Exact brute force (small matrices only)
sol = cutnorm(A; method=BruteForce())

# Exact methods via JuMP (load a solver of your choice first)
using HiGHS
sol = cutnorm(A; method=ILP(HiGHS.Optimizer))      # MILP — HiGHS is enough

using Gurobi
sol = cutnorm(A; method=INLP(Gurobi.Optimizer))      # integer nonlinear program
sol = cutnorm(A; method=QUBO(Gurobi.Optimizer))      # binary-quadratic program

# Time-limited exact solve (returns the best solution found so far)
sol = cutnorm(A; method=ILP(HiGHS.Optimizer), max_time=60.0)
```

## Options

### Multistart options

All keyword arguments are forwarded to `MultistartSettings`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `max_restarts` | `Int` | `1000` | Maximum number of multistart restarts |
| `max_time` | `Float64` | `3600.0` | Wall-clock time limit in seconds |
| `scaled` | `Bool` | `false` | Divide the result by $m \cdot n$ |
| `save_all_solutions` | `Bool` | `false` | Store the solution from every restart |
| `print_level` | `Int` | `0` | Verbosity (see below) |

**Print levels (Multistart):**

| Level | Output |
|---|---|
| 0 | Silent **(default)** |
| 1 | Improvements only |
| 2 | Improvements + logarithmic restarts (1--5, then every 10, 100, 1000, ...) |
| 3 | Every restart |
| 4 | Every restart + Subsolver output |

### BruteForce options

| Keyword | Type | Default | Description |
|---|---|---|---|
| `max_time` | `Float64` | `3600.0` | Wall-clock time limit in seconds |
| `scaled` | `Bool` | `false` | Divide the result by $m \cdot n$ |
| `print_level` | `Int` | `0` | Verbosity (see below) |

**Print levels (BruteForce):**

| Level | Output |
|---|---|
| 0 | Silent **(default)** |
| 1 | Improvements only |
| 2 | Improvements + logarithmic iterations |
| 3 | Every iteration |

### Exact JuMP method options (`INLP`, `ILP`, `QUBO`)

Forwarded to the corresponding settings object (`INLPSettings`, `ILPSettings`, `QUBOSettings`):

| Keyword | Type | Default | Description |
|---|---|---|---|
| `max_time` | `Float64` | `3600.0` | Time limit in seconds, passed to the solver via `set_time_limit_sec` |
| `scaled` | `Bool` | `false` | Divide the result by $m \cdot n$ |
| `print_level` | `Int` | `0` | Verbosity (see below) |

**Print levels (`INLP` / `ILP` / `QUBO`):**

| Level | Output |
|---|---|
| 0 | Silent **(default)** |
| 1 | CutNorm.jl header and summary; underlying solver silent |
| ≥ 2 | Header and summary **plus** the solver's own log |

## Solution Objects

`cutnorm` returns a solution object whose type depends on the method.

### MultistartAugmentedSolution / MultistartSignedSolution

| Field | Description |
|---|---|
| `value` | Best cut norm estimate found |
| `S`, `T` | Optimal row/column indicator vectors |
| `dims` | Problem dimensions $(m, n)$ |
| `restarts` | Total restarts performed |
| `improvements` | Number of times the best value improved |
| `best_restart` | Index of the restart that produced the best value |
| `runtime` | Total elapsed time in seconds |
| `termination_status` | `:max_restarts` or `:max_time` |
| `sign` | Best sign $\pm 1$ (`MultistartSignedSolution` only) |
| `all_solutions` | Per-restart results (when `save_all_solutions=true`) |

### BruteForceSolution

| Field | Description |
|---|---|
| `value` | Exact cut norm (if completed) |
| `S`, `T` | Optimal row/column indicator vectors |
| `dims` | Problem dimensions $(m, n)$ |
| `iterations` | Total $(s, t)$ pairs evaluated |
| `improvements` | Number of times the best value improved |
| `best_iteration` | Iteration that produced the best value |
| `runtime` | Total elapsed time in seconds |
| `termination_status` | `:optimal` or `:max_time` |

### INLPSolution / ILPSolution / QUBOSolution

Returned by the exact JuMP-based methods.

| Field | Description |
|---|---|
| `value` | Cut norm from the solver's objective (optimal when `termination_status == :optimal`) |
| `S`, `T` | Optimal row/column indicator vectors (size $m$ and $n$) |
| `dims` | Problem dimensions $(m, n)$ |
| `runtime` | Total elapsed wall-clock time in seconds |
| `solve_time` | Time reported by the solver itself (`JuMP.solve_time`) |
| `termination_status` | `:optimal`, `:max_time`, or another JuMP status as a `Symbol` |

## Advanced Usage

For fine-grained control you can construct solvers and models directly:

```julia
using CutNorm, SolverCore

# Build a multistart solver manually
solver = MultistartAugmentedSolver(A, AlternatingLinearSearch;
             max_restarts=200, print_level=1)
sol = solve!(solver)

# Build a brute-force solver manually
solver = BruteForceSolver(A; print_level=1)
sol = solve!(solver)

# Build an exact JuMP solver manually (the model is constructed once, here)
using HiGHS
solver = ILPSolver(A, HiGHS.Optimizer; max_time=60.0, print_level=1)
sol = solve!(solver)              # re-solvable; only optimize! runs again
```

## Author and Acknowledgements

This package was written by [Martin Köhler](https://www.tu-braunschweig.de/en/mo/team/koehler) at TU Braunschweig.

Funded by the European Union through ERC Consolidator Grant SCARCE ([101087662](https://cordis.europa.eu/project/id/101087662)).

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References
Frieze, A., Kannan, R. *Quick Approximation to Matrices and Applications*. Combinatorica 19, 175–220 (1999). https://doi.org/10.1007/s004930050052

Alon, N., Naor, A. *Approximating the Cut-Norm via Grothendieck's Inequality*. SIAM Journal on Computing, vol. 35, no. 4, pp. 787–803, 2006. https://doi.org/10.1137/S0097539704441629
