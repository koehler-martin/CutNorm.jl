```@meta
CurrentModule = CutNorm
```

# Options

Apart from `method`, every keyword argument of [`cutnorm`](@ref) is forwarded to the settings object of the chosen method.
The same keywords work when you construct a solver directly, and again when you call [`solve!`](@ref) on it, the settings object lives in the solver and is updated in place before each run:

```julia
solver = MultistartSignedSolver(A, AlternatingLinearSearch; max_restarts = 100)
sol = solve!(solver)                       # 100 restarts
sol = solve!(solver; max_restarts = 5000)  # same solver, larger budget
```

An unknown keyword is an error, not a silent no-op, and the message lists what the method does accept:

```jldoctest
julia> using CutNorm

julia> cutnorm(Float64[1 -1; -1 1]; max_iterations = 10)
ERROR: ArgumentError: unknown setting: max_iterations. Allowed keywords: (:max_restarts, :max_time, :scaled, :save_all_solutions, :print_level)
[...]
```

## Common to all methods

| Keyword       | Type      | Default  | Description                                        |
|:--------------|:----------|:---------|:---------------------------------------------------|
| `max_time`    | `Float64` | `3600.0` | Wall-clock time limit in seconds, `10.0` for [`TabuSearch`](@ref) |
| `scaled`      | `Bool`    | `false`  | Divide the reported value by ``m \cdot n``         |
| `print_level` | `Int`     | `0`      | Verbosity, `0` is silent                           |

`max_time` is checked by the solver itself for the multistart, tabu search and brute-force methods; for the JuMP-based methods it is handed to the solver via `JuMP.set_time_limit_sec`.
Either way, a time-out yields `termination_status == :max_time` and a value that is only a lower bound.

`scaled` affects only the reported `value` — `S` and `T` are unchanged.
Note that the scaling always uses the dimensions of the original matrix, also for the augmented methods.

## Multistart methods

Settings object: [`MultistartSettings`](@ref), used by [`MultistartSigned`](@ref) and
[`MultistartAugmented`](@ref).

| Keyword              | Type      | Default  | Description                                  |
|:---------------------|:----------|:---------|:---------------------------------------------|
| `max_restarts`       | `Int`     | `1000`   | Maximum number of restarts                   |
| `max_time`           | `Float64` | `3600.0` | Wall-clock time limit in seconds             |
| `scaled`             | `Bool`    | `false`  | Divide the value by ``m \cdot n``            |
| `save_all_solutions` | `Bool`    | `false`  | Keep the rounded solution of every restart   |
| `print_level`        | `Int`     | `0`      | Verbosity, see below                         |

Both stopping criteria are checked after each restart, so the run ends at the first of `max_restarts` and `max_time` to be reached, and `termination_status` becomes `:max_restarts` or `:max_time` accordingly.

`save_all_solutions` fills the `all_solutions` field with one `(objective, S, T)` named tuple per subproblem, that is one entry per restart for the augmented method, and **two** for the signed method, which solves both signs.
It is useful for studying the distribution of local optima, but it allocates two vectors per entry, so leave it off for long runs.

### Print levels

| Level | Output                                                                     |
|:------|:---------------------------------------------------------------------------|
| `0`   | Silent (default)                                                           |
| `1`   | Header, footer, and a row whenever the best value improves                  |
| `2`   | As `1`, plus logarithmically spaced restarts: 1–5, then every 10th, 100th, … |
| `3`   | A row for every restart                                                    |
| `4`   | As `3`, plus the log of the subsolver                                      |

Level `2` is the useful setting for long runs: the output stays short while still
showing that the solver is alive.

## Tabu search

Settings object: [`TabuSearchSettings`](@ref), used by [`TabuSearch`](@ref).
With ``N = m + n`` the number of bits of the search:

| Keyword             | Type                          | Default              | Description                                                       |
|:--------------------|:------------------------------|:---------------------|:------------------------------------------------------------------|
| `max_iter`          | `Int`                         | `typemax(Int)`       | Iteration limit of each search                                    |
| `max_time`          | `Float64`                     | `10.0`               | Wall-clock time limit in seconds                                  |
| `target`            | `Float64`                     | `Inf`                | Stop all searches once one of them reaches this value             |
| `tenure`            | `Int`                         | `0`                  | Base tabu tenure; `0` picks ``\max(1, \operatorname{round}(0.005 N))`` |
| `tenure_rand`       | `Int`                         | `10`                 | A flipped bit stays tabu for `tenure + rand(0:tenure_rand)` iterations |
| `stall`             | `Int`                         | `0`                  | Iterations without improvement before diversifying; `0` picks ``\max(100, 5N)`` |
| `perturb_frac`      | `Float64`                     | `0.1`                | Fraction of the ``N`` bits flipped when the elite is perturbed     |
| `max_perturbations` | `Int`                         | `3`                  | Failed perturbations in a row before a random restart with the opposite sign |
| `refresh`           | `Int`                         | `0`                  | Recompute ``A t`` and ``A^\top s`` every `refresh` iterations; `0` picks ``\max(1000, 10N)`` |
| `ntasks`            | `Int`                         | `Threads.nthreads()` | Number of independent searches, run in parallel                   |
| `seed`              | `UInt64`                      | random               | Search `k` is seeded with `seed + k`                              |
| `S0`, `T0`          | `Union{Nothing,Vector{Bool}}` | `nothing`            | Initial row and column indicators; `nothing` starts at random      |
| `scaled`            | `Bool`                        | `false`              | Divide the value by ``m \cdot n``                                 |
| `print_level`       | `Int`                         | `0`                  | Verbosity, see below                                              |

The search always runs until a limit is hit: each search stops after `max_iter` iterations or `max_time` seconds, and all of them stop as soon as one reaches `target`.
`termination_status` becomes `:target`, `:max_iter` or `:max_time` accordingly.
`target` is compared with the value in the units it is reported in, i.e. divided by ``m \cdot n`` if `scaled = true`.
Since there is no iteration limit by default, `max_time` defaults to 10 seconds rather than the hour of the other methods.
`tenure_rand` is clamped to `0:N÷4`, and the number of perturbed bits to `1:N`.

Keep `ntasks ≤ Threads.nthreads()`: a search never yields its thread, so surplus searches only start once others have finished, that is, after the time limit.
Every [`solve!`](@ref) reseeds the searches from `seed`, so repeated solves with the same `seed` and `max_iter` as the only binding limit return the same result.
The default `seed` is drawn once, when the settings are created.

`S0` and `T0` accept any vector with entries in ``\{0, 1\}``, for example `sol.S` and `sol.T` of another method.
Every search starts from them, and if both are given, the odd searches take the sign of the sum over `S0 × T0` and the even ones the opposite sign.
A wrong length raises a `DimensionMismatch`.

### Print levels

The log is printed while the searches run, one row per *phase*: the iterations of a search from its start, a perturbation or a restart up to the next one, or until it stops.
A phase is the counterpart of one restart of the multistart methods.

| Level | Output                                                                     |
|:------|:---------------------------------------------------------------------------|
| `0`   | Silent (default)                                                           |
| `1`   | Header, footer, and a row whenever a phase improves the best value          |
| `2`   | As `1`, plus logarithmically spaced phases of every search: 1–5, then every 10th, 100th, … |
| `3`   | A row for every phase                                                      |

```text
  Thread    Phase    Best Value   Improv       Iter           Obj   Time (s)
--------------------------------------------------------------------------------
       1        1    1.1079e+04        1       9088    1.1079e+04       0.43  *
       3        1    1.1108e+04        2      22738    1.1108e+04       0.46  *
```

`Thread` is the search, from `1` to `ntasks`, followed by its phase number, the best value over all searches and how often it improved, and the iterations of the phase.
`Obj` is the sum ``s^\top A t`` at the best point of the phase; its sign is the sign the phase searched with.
A `*` marks the phases that improved the best value.
After all searches have stopped, a summary row per search and the footer follow.
A search prints at the end of its phases, so its rows are at least `stall` iterations apart, which on very large matrices can be several seconds.
Lowering `stall` gives more frequent output, but also makes the search diversify more often.

## Brute force

Settings object: [`BruteForceSettings`](@ref).

| Keyword       | Type      | Default  | Description                        |
|:--------------|:----------|:---------|:-----------------------------------|
| `max_time`    | `Float64` | `3600.0` | Wall-clock time limit in seconds   |
| `scaled`      | `Bool`    | `false`  | Divide the value by ``m \cdot n``  |
| `print_level` | `Int`     | `0`      | Verbosity, see below               |

There is no iteration limit — the enumeration is either exhaustive (`termination_status == :optimal`) or cut short by `max_time`.

### Print levels

| Level | Output                                                                       |
|:------|:-----------------------------------------------------------------------------|
| `0`   | Silent (default)                                                             |
| `1`   | Header, footer, and a row whenever the best value improves                    |
| `2`   | As `1`, plus logarithmically spaced iterations: 1–5, then every 10th, 100th, … |
| `3`   | A row for every enumerated pair — expect a lot of output                      |

## INLP, ILP and QUBO

Settings objects: [`INLPSettings`](@ref), [`ILPSettings`](@ref), [`QUBOSettings`](@ref). All three accept the same three keywords.

| Keyword       | Type      | Default  | Description                                            |
|:--------------|:----------|:---------|:-------------------------------------------------------|
| `max_time`    | `Float64` | `3600.0` | Time limit in seconds, via `JuMP.set_time_limit_sec`    |
| `scaled`      | `Bool`    | `false`  | Divide the value by ``m \cdot n``                      |
| `print_level` | `Int`     | `0`      | Verbosity, see below                                   |

### Print levels

| Level | Output                                                            |
|:------|:------------------------------------------------------------------|
| `0`   | Silent (default)                                                  |
| `1`   | CutNorm.jl header and summary; the underlying solver stays silent  |
| `≥ 2` | Header and summary **plus** the solver's own log                   |

Levels `0` and `1` call `JuMP.set_silent` on the model, level `≥ 2` calls `JuMP.unset_silent`, and both `max_time` and `print_level` are re-applied on every [`solve!`](@ref).
Anything else you want to configure, MIP gaps, thread counts,
solver-specific attributes, can be set directly on `solver.model` with `JuMP.set_optimizer_attribute`; see [Advanced usage](50-advanced.md).
