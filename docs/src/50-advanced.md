```@meta
CurrentModule = CutNorm
```

# Advanced usage

[`cutnorm`](@ref) is a thin wrapper: it builds a solver, calls [`solve!`](@ref) once,
and returns the solution. Doing those steps yourself buys you three things — reusing
the setup across solves, access to the underlying model, and the ability to reuse the
solution storage as well.

## Reusing a solver

Every method has a solver type, and its constructor takes the same keyword arguments as
[`cutnorm`](@ref):

| Method                        | Solver                                |
|:------------------------------|:--------------------------------------|
| [`MultistartSigned`](@ref)    | [`MultistartSignedSolver`](@ref)      |
| [`MultistartAugmented`](@ref) | [`MultistartAugmentedSolver`](@ref)   |
| [`BruteForce`](@ref)          | [`BruteForceSolver`](@ref)            |
| [`INLP`](@ref)                | [`INLPSolver`](@ref)                  |
| [`ILP`](@ref)                 | [`ILPSolver`](@ref)                   |
| [`QUBO`](@ref)                | [`QUBOSolver`](@ref)                  |

```jldoctest adv
julia> using CutNorm

julia> A = Float64[ 2 -1  0;
                   -1  3 -1;
                    0 -1  2];

julia> solver = MultistartSignedSolver(A, AlternatingLinearSearch; max_restarts = 20);

julia> sol = solve!(solver);

julia> sol.value
4.0
```

The solver owns the model, the subsolver, its statistics object, the Sobol sequence and
the settings, and all of it is reused, so a second [`solve!`](@ref) allocates nothing
new. Keywords passed to [`solve!`](@ref) update the settings first:

```jldoctest adv
julia> sol = solve!(solver; max_restarts = 100);

julia> sol.restarts
100
```

One thing to keep in mind for the multistart solvers: the Sobol sequence keeps
advancing across calls. A second `solve!` therefore explores *different* starting points
than the first — useful if you want to continue searching, surprising if you expected
the same answer twice. Build a fresh solver when you want to repeat a run exactly.

For the JuMP-based solvers, reuse matters more, because the model is constructed in the
solver's constructor and only `JuMP.optimize!` runs again:

```julia
using HiGHS
solver = ILPSolver(A, HiGHS.Optimizer; max_time = 60.0, print_level = 1)
sol = solve!(solver)
sol = solve!(solver; max_time = 600.0)   # same model, larger budget
```

## Reusing the solution

Each [`solve!`](@ref) also accepts a solution object, which it resets and writes into.
This keeps the indicator vectors and the `all_solutions` buffer from being reallocated
when you solve many times:

```jldoctest adv
julia> sol = MultistartSignedSolution(A);

julia> solve!(solver, sol; max_restarts = 10);

julia> sol.value, sol.restarts
(4.0, 10)
```

The solution constructors take a matrix, a pair of dimensions, or an element type and a
pair of dimensions — see [`MultistartSignedSolution`](@ref) and its siblings. For the
augmented method, pass the dimensions of the *original* matrix.

## Working with the JuMP model

The `model` field of [`INLPSolver`](@ref), [`ILPSolver`](@ref) and [`QUBOSolver`](@ref)
is a plain `JuMP.Model`, so anything JuMP can do to a model, you can do here — setting
solver attributes, warm-starting, or inspecting the formulation:

```julia
using HiGHS, JuMP
solver = ILPSolver(A, HiGHS.Optimizer)

set_optimizer_attribute(solver.model, "mip_rel_gap", 1e-6)
set_optimizer_attribute(solver.model, "threads", 8)

sol = solve!(solver)

println(solution_summary(solver.model))
```

The variables are registered, so they can be retrieved by name:
`model[:x]` and `model[:y]` for the [`ILP`](@ref) and [`QUBO`](@ref) models, `model[:s]`,
`model[:t]` and `model[:σ]` for the [`INLP`](@ref) one. Note that `max_time` and
`print_level` from the settings are re-applied on every [`solve!`](@ref), so set those
through the settings rather than on the model.

If you only want the model and not the solve, the builders are available directly:
[`CutNorm.build_ilp_model`](@ref), [`CutNorm.build_model`](@ref) and
[`CutNorm.build_qubo_model`](@ref).

## The bilinear model

The multistart solvers optimize an `NLPModel` from
[NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), so the relaxation
is usable on its own — with any JSO-compatible solver, or for your own experiments.
[`BilinearModel`](@ref) minimizes ``s^\top A t`` over the unit box in the stacked
variable ``x = [s; t]``, and [`SignedBilinearModel`](@ref) adds a mutable sign:

```jldoctest adv
julia> using NLPModels

julia> nlp = SignedBilinearModel(A);

julia> nlp.meta.nvar, nlp.meta.lvar, nlp.meta.uvar
(6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

julia> x = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

julia> obj(nlp, x)
4.0

julia> nlp.sign = Int8(-1);

julia> obj(nlp, x)
-4.0
```

Since the models minimize, the cut norm value of a vertex is `-obj(nlp, x)`.

Besides the usual `obj`, `grad!`, `objgrad!` and `hprod!`, the models provide the
block-wise gradients [`grad_s!`](@ref) and [`grad_t!`](@ref). They are what makes the
alternating scheme cheap: for fixed `t` the objective is linear in `s`, so `A*t` alone
determines the optimal `s`.

```jldoctest adv
julia> nlp = BilinearModel(A);

julia> gs = zeros(3);

julia> grad_s!(nlp, [1.0, 0.0, 1.0], gs)
3-element Vector{Float64}:
  2.0
 -2.0
  2.0
```

The Hessian is constant — the objective is bilinear — which is why `hprod!` is a pair of
matrix-vector products and needs no second-derivative information.

## Writing a subsolver

The subsolvers are `SolverCore.AbstractOptimizationSolver`s, so they follow the JSO
interface: a constructor taking the model, a `SolverCore.reset!` method, and

```julia
SolverCore.solve!(subsolver, model, stats; x, max_iter, max_time, verbose)
```

writing into a `GenericExecutionStats`. [`AlternatingLinearSearch`](@ref) and
[`GreedySolver`](@ref) are compact examples of exactly that, and `TronSolver` from
JSOSolvers.jl slots in unchanged.

To make a new one usable from the multistart solvers, add a
[`CutNorm.get_subsolver`](@ref) method returning the solver and a matching statistics
object:

```julia
CutNorm.get_subsolver(::Type{MySolver}, model) =
    MySolver(model), GenericExecutionStats(model)
```

after which `cutnorm(A; method = MultistartSigned{MySolver}())` works. Bear in mind
that the multistart loop rounds whatever the subsolver returns to the nearest vertex, so
a subsolver that stops at an interior point is not wrong — just less informative than
one that ends at a vertex.
