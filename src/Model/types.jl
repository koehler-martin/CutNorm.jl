"""
    AbstractBilinearModel{T,S} <: AbstractNLPModel{T,S}

Supertype of the `NLPModel`s used by the multistart solvers: bound-constrained
bilinear programs in the stacked variable `x = [s; t]` over `[0,1]^m × [0,1]^n`.

Subtypes implement the `NLPModels` API (`obj`, `grad!`, `objgrad!`, `hprod!`) plus the
block-wise gradients [`grad_s!`](@ref) and [`grad_t!`](@ref), and answer
`Base.sign(nlp)` with the sign of their objective.

Concrete subtypes: [`BilinearModel`](@ref), [`SignedBilinearModel`](@ref).
"""
abstract type AbstractBilinearModel{T,S} <: AbstractNLPModel{T,S} end

"""
    BilinearData(A, dims)

Internal container holding the matrix `A` of a bilinear model together with its
dimensions `dims = (m, n)`. The models keep their own copy of `A`, converted to the
element type of the model.
"""
struct BilinearData{T<:AbstractFloat,M<:AbstractMatrix{T}}
    A::M
    dims::Tuple{Int,Int}
end

struct BilinearModel{T,S<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractBilinearModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    data::BilinearData{T,M}
end

"""
    BilinearModel(A)

An `NLPModel` for the bound-constrained bilinear program

    min  s'At    s.t.  s ∈ [0,1]^m,  t ∈ [0,1]^n,

where `A` is an `m × n` matrix. The optimization variable is `x = [s; t] ∈ ℝ^(m+n)`.

An integer matrix is converted to floating point, and `A` is copied into the model, so
later changes to the input do not affect it. The model is bound-constrained with
`lvar = 0` and `uvar = 1`, and `x0` is where the multistart solvers write their Sobol
points.

Since the model *minimizes*, the cut norm value is `-obj(nlp, x)` at the optimum. Use
this model on the augmented matrix, where the sign ambiguity is already gone; for the
original matrix use [`SignedBilinearModel`](@ref), whose sign can be flipped.

# Examples

```jldoctest
julia> nlp = BilinearModel(Float64[1 -1; -1 1]);

julia> nlp.meta.nvar, sign(nlp)
(4, 1)

julia> using NLPModels

julia> obj(nlp, [1.0, 0.0, 1.0, 0.0])
1.0
```

See also [`SignedBilinearModel`](@ref), [`CutNorm.augment_matrix`](@ref),
[`grad_s!`](@ref), [`grad_t!`](@ref).
"""
BilinearModel(A::AbstractMatrix) = BilinearModel(float(A))

function BilinearModel(A::M) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    m, n = size(A)
    dims = (m, n)
    d = m + n
    meta = NLPModelMeta{T,Vector{T}}(d, lvar=zeros(T, d), uvar=ones(T, d))
    counters = Counters()
    data = BilinearData{T,M}(similar(A, T), dims)
    copyto!(data.A, A)

    return BilinearModel(
        meta, counters, data
    )
end
Base.sign(::BilinearModel) = Int8(1)


mutable struct SignedBilinearModel{T,S<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractBilinearModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    data::BilinearData{T,M}
    sign::Int8
end

"""
    SignedBilinearModel(A; sign=Int8(1))

An `NLPModel` for the bound-constrained bilinear program

    min  σ·s'At    s.t.  s ∈ [0,1]^m,  t ∈ [0,1]^n,

where `A` is an `m × n` matrix, `σ ∈ {+1, -1}` is the sign, and the
optimization variable is `x = [s; t] ∈ ℝ^(m+n)`.

The `sign` field is mutable: setting `nlp.sign = Int8(-1)` switches the objective to
the other branch without rebuilding the model, which is what
[`MultistartSignedSolver`](@ref) does twice per restart to cover both signs of
`|s'At|`. All the evaluation routines pick the sign up automatically, and
`Base.sign(nlp)` returns it.

An integer matrix is converted to floating point, and `A` is copied into the model, so
later changes to the input do not affect it. The model is bound-constrained with
`lvar = 0` and `uvar = 1`. Since it *minimizes*, the cut norm value is `-obj(nlp, x)`
at the optimum.

# Examples

```jldoctest
julia> using NLPModels

julia> nlp = SignedBilinearModel(Float64[1 -1; -1 1]);

julia> obj(nlp, [1.0, 0.0, 1.0, 0.0])
1.0

julia> nlp.sign = Int8(-1);

julia> obj(nlp, [1.0, 0.0, 1.0, 0.0])
-1.0
```

See also [`BilinearModel`](@ref), [`grad_s!`](@ref), [`grad_t!`](@ref).
"""
SignedBilinearModel(A::AbstractMatrix; kwargs...) = SignedBilinearModel(float(A); kwargs...)

function SignedBilinearModel(A::M; sign::Int8=Int8(1)) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    m, n = size(A)
    dims = (m, n)
    d = m + n
    meta = NLPModelMeta{T,Vector{T}}(d, lvar=zeros(T, d), uvar=ones(T, d))
    counters = Counters()
    data = BilinearData{T,M}(similar(A, T), dims)
    copyto!(data.A, A)

    return SignedBilinearModel(
        meta, counters, data, sign
    )
end
Base.sign(nlp::SignedBilinearModel) = nlp.sign