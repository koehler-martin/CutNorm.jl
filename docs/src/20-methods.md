```@meta
CurrentModule = CutNorm
```

# Methods

The `method` keyword of [`cutnorm`](@ref) selects the algorithm. All method values are
subtypes of [`AbstractCutNormMethod`](@ref) and carry no problem data — for the
JuMP-based ones, only the optimizer to use.

```julia
sol = cutnorm(A; method = MultistartSigned{TronSolver}())
```

## Choosing a method

| Method                                          | Result       | Needs                             | Use when |
|:------------------------------------------------|:-------------|:----------------------------------|:---------|
| [`MultistartSigned{S}()`](@ref MultistartSigned)       | lower bound  | nothing                     | the default; any size |
| [`MultistartAugmented{S}()`](@ref MultistartAugmented) | lower bound  | nothing                     | one subproblem per restart is preferable |
| [`BruteForce()`](@ref BruteForce)               | exact        | nothing                           | `m + n` small (say ≲ 25) |
| [`ILP(optimizer)`](@ref ILP)                    | exact        | any MILP solver                   | exact answers beyond enumeration |
| [`INLP(optimizer)`](@ref INLP)                  | exact        | global MINLP solver               | comparing formulations |
| [`QUBO(optimizer)`](@ref QUBO)                  | exact        | nonconvex binary-quadratic solver | comparing formulations |

The heuristic methods return a valid cut in every case, so their value is always a
lower bound on ``\lVert A\rVert_\square`` — attained by the reported sets. The exact methods
return the true cut norm, but only if they ran to completion: check
`termination_status == :optimal` before treating a value as exact.

## The relaxation

Both heuristic methods work on the bilinear relaxation of the combinatorial problem,

```math
\max_{s \in [0,1]^m,\, t \in [0,1]^n} \left| s^\top A\, t \right| ,
```

whose objective is linear in each block of variables separately. The maximum is
therefore attained at a vertex of the box, i.e. at a genuine ``\{0,1\}`` indicator
pair, which is why rounding a local solution always yields a cut. Since the problem is
nonconvex, the local solutions differ, and the package attacks that with a multistart
strategy: the starting points come from a Sobol sequence, so the box is covered
quasi-uniformly and the results are reproducible without a seed.

The absolute value is what the two heuristic methods handle differently.

### The signed method

[`MultistartSigned`](@ref) keeps the original matrix and solves both sign variants —
minimizing ``+s^\top A t`` and ``-s^\top A t`` — in every restart, keeping the better
one. That is two subproblems per restart, and the winning sign is reported in the
`sign` field of the solution. This is the default.

```julia
sol = cutnorm(A)                                            # the default
sol = cutnorm(A; method = MultistartSigned{GreedySolver}())
```

### The augmented method

[`MultistartAugmented`](@ref) instead augments the matrix to

```math
A_{\text{aug}} = \begin{pmatrix} A & -A\mathbf{1} \\
                                 -\mathbf{1}^\top A & \mathbf{1}^\top A \mathbf{1}
                 \end{pmatrix},
```

an ``(m+1) \times (n+1)`` matrix whose rows and columns all sum to zero (see
[`CutNorm.augment_matrix`](@ref)). Flipping all bits of an indicator vector then leaves
the objective unchanged, so the sign ambiguity disappears and **one** subproblem per
restart suffices. The price is one extra row and column, plus undoing the resulting
bit-flip symmetry: the two pivot entries are used to normalize the solution before it
is cut back to the original dimensions.

```julia
sol = cutnorm(A; method = MultistartAugmented{AlternatingLinearSearch}())
```

## Subsolvers

The type parameter `S` of the multistart methods picks the local solver used in each
restart. It is mandatory — there is no default when the method is written out — and it
must be one of these three:

| Subsolver                          | Idea                                                                 |
|:-----------------------------------|:---------------------------------------------------------------------|
| [`AlternatingLinearSearch`](@ref)  | Alternate between the blocks, moving each to the bound its partial gradient dictates. Fastest, and the default. |
| `TronSolver`                       | Trust-region Newton method from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl), re-exported for convenience. Most accurate locally, most expensive per restart. |
| [`GreedySolver`](@ref)             | Round to a vertex, then apply the single most improving bit flip until none is left, updating the gradient incrementally. |

Any other `AbstractOptimizationSolver` raises an error naming the three.

Since all three end at a vertex, they differ in how thoroughly each restart is
explored rather than in the kind of answer they give. If a fixed time budget matters,
the cheap subsolvers buy you more restarts; if each restart should count, `TronSolver`
is the careful choice.

## Exact methods

### Exhaustive enumeration

[`BruteForce`](@ref) enumerates all ``2^m \cdot 2^n`` indicator pairs. The outer loop
runs over the column indicators, so ``A t`` is formed once per `t` and reused for all
``2^m`` row indicators, leaving one dot product per pair. No external solver is
involved, and `max_time` gives you a safety net: on a time-out the best value so far is
returned with status `:max_time`.

```jldoctest
julia> using CutNorm

julia> sol = cutnorm(Float64[1 -1; -1 1]; method = BruteForce());

julia> sol.value, sol.iterations, sol.termination_status
(1.0, 16, :optimal)
```

### The MILP formulation

[`ILP`](@ref) formulates the problem as a mixed-integer **linear** program. The product
``x_i y_j`` of the binary indicators is replaced by a variable ``w_{ij}`` with the
McCormick constraints

```math
w_{ij} \le x_i, \qquad w_{ij} \le y_j, \qquad w_{ij} \ge x_i + y_j - 1,
```

which are exact for binary ``x, y``, and the absolute value is modeled with a binary
sign variable and a big-M reformulation (see [`CutNorm.compute_bigM`](@ref)). No
indicator constraints are used, so plain MILP solvers such as
[HiGHS](https://highs.dev/) work:

```julia
using HiGHS
sol = cutnorm(A; method = ILP(HiGHS.Optimizer), max_time = 60.0)
```

This is the exact method to reach for first — MILP solvers are widely available, and
the model is the easiest of the three for a solver to handle.

### The MINLP formulation

[`INLP`](@ref) keeps the bilinear objective and adds a binary sign variable ``\sigma``,
maximizing ``(2\sigma - 1)\, s^\top A t`` over binary ``s``, ``t``, ``\sigma``. The
model is compact but nonconvex and nonlinear in the binaries, so it needs a global
MINLP solver — HiGHS cannot do it:

```julia
using Gurobi
sol = cutnorm(A; method = INLP(Gurobi.Optimizer))
```

### The QUBO formulation

[`QUBO`](@ref) runs on the augmented matrix, where no sign variable is needed, and
maximizes the quadratic form ``\sum_{ij} (A_{\text{aug}})_{ij}\, x_i y_j`` over
binaries. The formulation is the smallest of the three in terms of variables and
constraints, but the objective is a nonconvex binary quadratic, so the solver must
support that class:

```julia
using Gurobi
sol = cutnorm(A; method = QUBO(Gurobi.Optimizer))
```

As for the multistart-augmented method, the pivot bits of the augmented solution are
used to map the indicators back to the original dimensions.

### A note on solvers

The `INLP`, `ILP` and `QUBO` methods build a JuMP model and hand it to the `optimizer`
you pass in, so any compatible solver works. The solver packages are deliberately
**not** dependencies of CutNorm.jl: install and load the one you want yourself, and
make sure it supports the model class — HiGHS handles the linear `ILP`, but neither the
quadratic `QUBO` nor the nonlinear `INLP`.

If you want to inspect or tune the model, build the solver yourself and reach for
`solver.model`; see [Advanced usage](50-advanced.md).
