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
| `max_time`    | `Float64` | `3600.0` | Wall-clock time limit in seconds                   |
| `scaled`      | `Bool`    | `false`  | Divide the reported value by ``m \cdot n``         |
| `print_level` | `Int`     | `0`      | Verbosity, `0` is silent                           |

`max_time` is checked by the solver itself for the multistart and brute-force methods; for the JuMP-based methods it is handed to the solver via `JuMP.set_time_limit_sec`.
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
