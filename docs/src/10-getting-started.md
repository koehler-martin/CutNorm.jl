```@meta
CurrentModule = CutNorm
```

# Getting started

This page walks through a complete session: computing a cut norm, reading the result, checking it, and watching the solver work.

## A first solve

Everything goes through [`cutnorm`](@ref). It takes the matrix and returns a solution object.

```jldoctest tour
julia> using CutNorm

julia> A = Float64[ 2 -1  0;
                   -1  3 -1;
                    0 -1  2];

julia> sol = cutnorm(A);

julia> sol.value
4.0
```

## Reading the solution

The interesting part of a cut norm is usually not the number but the rows and column subsets attaining it.
`sol.S` and `sol.T` are the indicator vectors of the optimal row and column sets, with entries `0.0` and `1.0`:

```jldoctest tour
julia> sol.S
3-element Vector{Float64}:
 1.0
 0.0
 1.0

julia> sol.T
3-element Vector{Float64}:
 1.0
 0.0
 1.0
```

So the optimum is attained by the first and third row against the first and third column.
Turn the indicators into index sets with `findall`, and use them to verify the value independently:

```jldoctest tour
julia> S, T = findall(isone, sol.S), findall(isone, sol.T)
([1, 3], [1, 3])

julia> abs(sum(A[S, T])) == sol.value
true
```

## Scaling

Cut norms grow with the size of the matrix, so it is often more useful to compare the value per entry.
Pass `scaled = true` to divide the result by ``m \cdot n``:

```jldoctest tour
julia> cutnorm(A; scaled = true).value
0.4444444444444444
```

## Stopping criteria

The default method is a multistart heuristic, so it needs a stopping rule.
Both `max_restarts` (default `1000`) and `max_time` in seconds (default `3600.0`) apply, and whichever is reached first ends the run.
The `termination_status` field says which one it was:

```jldoctest tour
julia> sol = cutnorm(A; max_restarts = 10);

julia> sol.restarts, sol.termination_status
(10, :max_restarts)
```

More restarts cost time but improve the chance of hitting the global optimum.
The `improvements` and `best_restart` fields tell you how much of the budget actually paid off if the best value was found in restart 4 out of 1000, you can safely spend less:

```jldoctest tour
julia> sol.improvements, sol.best_restart
(3, 4)
```

Note that the starting points come from a Sobol sequence, not from a random number generator: the same call on the same matrix gives the same answer, with no seed to set.

## Watching the solver

`print_level` controls the log. Level `0` is silent, level `1` prints a header, a row
whenever the incumbent improves, and a summary; higher levels print more (see
[Options](30-options.md)):

```julia
julia> sol = cutnorm(A; max_restarts = 100, print_level = 1);
================================================================================
                               CutNorm.jl - v0.1.0
                                (c) Martin Köhler
                               TU Braunschweig 2026
================================================================================
  Problem size:      3 x 3
  Method:            Signed Multistart
  Description:       Solves both +/- objectives per restart
  Subsolver:         AlternatingLinearSearch
  Max restarts:      100
  Max time:          3600.0s
```

## An exact answer

For small matrices you can skip the heuristic entirely. [`BruteForce`](@ref) needs no
external solver but evaluates all ``2^m \cdot 2^n`` pairs, so it is practical only for
small `m + n`:

```jldoctest tour
julia> exact = cutnorm(A; method = BruteForce());

julia> exact.value, exact.iterations, exact.termination_status
(4.0, 64, :optimal)
```

`termination_status == :optimal` is what certifies the value as exact. Here it
confirms that the heuristic had already found the global optimum.
If the time limit hits first, the status is `:max_time` and the value is only a lower bound.

For matrices that are too large to enumerate but still small enough to optimize exactly, use one of the JuMP-based methods with a solver of your choice:

```julia
using HiGHS
sol = cutnorm(A; method = ILP(HiGHS.Optimizer), max_time = 60.0)
```

The solver packages are not dependencies of CutNorm.jl. Install and load the one you want yourself.
See [Methods](20-methods.md) for which method needs which kind of solver.

## Next steps

- [Methods](20-methods.md) — choosing between the heuristic and exact methods
- [Options](30-options.md) — the full list of keyword arguments
- [Solution objects](40-solutions.md) — every field of every solution type
- [Advanced usage](50-advanced.md) — reusing a solver across many solves
