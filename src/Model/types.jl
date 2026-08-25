abstract type AbstractBilinearModel{T,S} <: AbstractNLPModel{T,S} end

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