```@meta
CurrentModule = CutNorm
```

# Solution objects

[`cutnorm`](@ref) and [`solve!`](@ref) return a solution object whose type depends on the method.
They all share a core set of fields, and add whatever else their method can report.

## Common fields

| Field                | Description                                                        |
|:---------------------|:-------------------------------------------------------------------|
| `value`              | Best cut norm value found, divided by ``m \cdot n`` if `scaled`     |
| `S`, `T`             | Row and column indicator vectors, entries `0.0` or `1.0`            |
| `dims`               | Problem size `(m, n)` of the original matrix                        |
| `runtime`            | Total elapsed wall-clock time in seconds                            |
| `termination_status` | Why the solver stopped                                              |

`S` has length `m` and `T` has length `n`, also for the augmented methods, which map their extra pivot bits away before reporting.
`findall(isone, sol.S)` turns an indicator into an index set:

```jldoctest sol
julia> using CutNorm

julia> A = Float64[ 2 -1  0;
                   -1  3 -1;
                    0 -1  2];

julia> sol = cutnorm(A);

julia> sol.dims, sol.value
((3, 3), 4.0)

julia> findall(isone, sol.S), findall(isone, sol.T)
([1, 3], [1, 3])
```

### Termination status

| Status           | Meaning                                                            |
|:-----------------|:-------------------------------------------------------------------|
| `:optimal`       | The method finished and `value` is the exact cut norm               |
| `:max_restarts`  | The multistart budget was exhausted                                |
| `:max_iter`      | The tabu search did `max_iter` iterations in every search           |
| `:target`        | A tabu search reached the `target` value                           |
| `:max_time`      | The time limit hit; `value` is the best found so far                |
| `:unknown`       | The solution has not been used yet                                 |
| other `Symbol`   | A JuMP/MOI status name, for the solver-based methods                |

Only `:optimal` certifies a value as the true cut norm.
Everything else is a lower bound, a valid one, attained by the reported sets:

```jldoctest sol
julia> abs(sum(A[findall(isone, sol.S), findall(isone, sol.T)])) == sol.value
true
```

## Multistart solutions

[`MultistartSignedSolution`](@ref) and [`MultistartAugmentedSolution`](@ref) add the history of the multistart loop.

| Field                | Description                                                       |
|:---------------------|:------------------------------------------------------------------|
| `restarts`           | Number of restarts performed                                      |
| `improvements`       | How often the incumbent improved                                  |
| `best_restart`       | Index of the restart that produced `value`                        |
| `best_initial_guess` | The Sobol point `x = [s; t]` behind the best value                |
| `all_solutions`      | Per-subproblem `(objective, S, T)` tuples, if `save_all_solutions` |
| `sign`               | Sign ``\pm 1`` of the winning objective (signed method only)       |

`improvements` and `best_restart` are the practical way to size the budget: if the last improvement came early, further restarts are unlikely to help.

```jldoctest sol
julia> sol = cutnorm(A; max_restarts = 10);

julia> sol.restarts, sol.improvements, sol.best_restart
(10, 3, 4)

julia> sol.sign
-1
```

With `save_all_solutions = true`, every subproblem is recorded, one entry per restart for the augmented method, two for the signed method, which solves both signs per restart:

```jldoctest sol
julia> sol = cutnorm(A; max_restarts = 4, save_all_solutions = true);

julia> length(sol.all_solutions)
8

julia> sol.all_solutions[1].objective
1.0
```

`best_initial_guess` has length `m + n` for the signed method and `m + n + 2` for the augmented one, matching the model each of them solves.

## Tabu search solutions

[`TabuSearchSolution`](@ref) reports the parallel searches.
The counters are summed over all searches, while the `best_*` fields refer to the search that produced `value`.

| Field            | Description                                                          |
|:-----------------|:---------------------------------------------------------------------|
| `sign`           | Sign ``\pm 1`` of the sum of `A` over `S × T`                         |
| `iterations`     | Number of flips, summed over all searches                             |
| `perturbations`  | Number of perturbations of the elite, summed over all searches        |
| `restarts`       | Number of random restarts, summed over all searches                   |
| `best_task`      | Index of the search that produced `value`                             |
| `best_iteration` | Iteration of that search that produced `value`                        |
| `improvements`   | How often the best value over all searches improved, checked at the end of every phase |
| `time_to_best`   | Seconds from the start of the solve until `value` was found           |

```jldoctest sol
julia> sol = cutnorm(A; method = TabuSearch(), max_iter = 1000, ntasks = 2, seed = 1);

julia> sol.value, sol.iterations, sol.termination_status
(4.0, 2000, :max_iter)
```

`time_to_best` compared with `runtime` is the practical way to size the budget, much like `best_restart` for the multistart methods: if the best value was found early, a shorter run would have done.
When several searches reach the same value, `best_task` is the one that got there in the fewest iterations.
`improvements` depends on the order in which the searches end their phases, so unlike the other fields it can differ between two otherwise identical runs.

## Brute-force solutions

[`BruteForceSolution`](@ref) reports the enumeration instead of restarts.

| Field            | Description                                  |
|:-----------------|:---------------------------------------------|
| `iterations`     | Number of ``(s, t)`` pairs evaluated          |
| `improvements`   | How often the incumbent improved              |
| `best_iteration` | Iteration that produced `value`               |

```jldoctest sol
julia> sol = cutnorm(A; method = BruteForce());

julia> sol.iterations, sol.termination_status
(64, :optimal)
```

`iterations` reaches ``2^m \cdot 2^n`` exactly when the run completed.

## Solutions of the JuMP methods

[`INLPSolution`](@ref), [`ILPSolution`](@ref) and [`QUBOSolution`](@ref) have identical fields: the common ones plus the solver's own timing.

| Field        | Description                                             |
|:-------------|:--------------------------------------------------------|
| `solve_time` | Time reported by the solver itself, `JuMP.solve_time`    |

`runtime` measures the whole [`solve!`](@ref) call, so the difference between the two is the overhead of applying settings and reading the solution back, the model itself is built once when the solver is constructed, not here.

For these methods, always look at `termination_status` first.
If the solver returned no values at all, infeasible, interrupted, or out of time before finding an incumbent, then `value`, `S` and `T` are left at zero rather than filled with something meaningless.
The status is `:optimal` or `:max_time` where those apply, and otherwise the MOI status name as a `Symbol`, for example `:INFEASIBLE`.