"""
    obj(nlp, x)

Evaluate the bilinear objective `f(x) = σ·s'At`, where `s = x[1:m]`, `t = x[m+1:m+n]`,
and `σ = sign(nlp)`.
"""
function _obj(nlp::AbstractBilinearModel, x::AbstractVector)
    NLPModels.increment!(nlp, :neval_obj)
    m, n = nlp.data.dims
    A = nlp.data.A
    s = @view x[1:m]
    t = @view x[m+1:m+n]
    return dot(s, A, t)
end

function NLPModels.obj(nlp::BilinearModel, x::AbstractVector)
    return _obj(nlp, x)
end

function NLPModels.obj(nlp::SignedBilinearModel, x::AbstractVector)
    return nlp.sign * _obj(nlp, x)
end

"""
    grad!(nlp, x, g)

Compute the gradient `g = σ·[A*t; A'*s]` in-place.
"""
function _grad!(
    nlp::AbstractBilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    @lencheck nlp.meta.nvar x g
    NLPModels.increment!(nlp, :neval_grad)
    m, n = nlp.data.dims

    A = nlp.data.A

    s = @view x[1:m]
    t = @view x[m+1:m+n]

    @views mul!(g[1:m], A, t)
    @views mul!(g[m+1:m+n], transpose(A), s)

    return g
end

function NLPModels.grad!(
    nlp::BilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    _grad!(nlp, x, g)
end

function NLPModels.grad!(
    nlp::SignedBilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    _grad!(nlp, x, g)
    rmul!(g, sign(nlp))
    return g
end

"""
    objgrad!(nlp, x, g)

Compute the objective and gradient simultaneously. Returns `(σ·s'At, g)`.
"""
function _objgrad!(
    nlp::AbstractBilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    @lencheck nlp.meta.nvar x g
    NLPModels.increment!(nlp, :neval_obj)
    NLPModels.increment!(nlp, :neval_grad)

    m, n = nlp.data.dims
    A = nlp.data.A

    s = @view x[1:m]
    t = @view x[m+1:m+n]

    @views mul!(g[1:m], A, t)
    @views mul!(g[m+1:m+n], transpose(A), s)

    val = dot(s, @view g[1:m])

    return val, g
end

function NLPModels.objgrad!(
    nlp::BilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    _objgrad!(nlp, x, g)
end

function NLPModels.objgrad!(
    nlp::SignedBilinearModel,
    x::AbstractVector,
    g::AbstractVector
)
    val, g = _objgrad!(nlp, x, g)
    σ = sign(nlp)
    rmul!(g, σ)
    return σ * val, g
end

"""
    hprod!(nlp, x, v, Hv; obj_weight=1.0)

Hessian-vector product `Hv = obj_weight·σ·[A*v_t; A'*v_s]`. The Hessian is constant.
"""
function _hprod!(
    nlp::AbstractBilinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight::Real=1.0
)
    @lencheck nlp.meta.nvar x v Hv
    NLPModels.increment!(nlp, :neval_hprod)

    m, n = nlp.data.dims
    A = nlp.data.A

    s = @view v[1:m]
    t = @view v[m+1:m+n]

    @views mul!(Hv[1:m], A, t)
    @views mul!(Hv[m+1:m+n], transpose(A), s)

    rmul!(Hv, obj_weight)

    return Hv
end

function NLPModels.hprod!(
    nlp::BilinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight::Real=1.0
)
    _hprod!(nlp, x, v, Hv; obj_weight)
end

function NLPModels.hprod!(
    nlp::SignedBilinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight::Real=1.0
)
    _hprod!(nlp, x, v, Hv; obj_weight=sign(nlp) * obj_weight)
end

"""
    grad_s!(nlp, t, gs)

Partial gradient w.r.t. `s`: `gs = σ·A*t`.
"""
function _grad_s!(nlp::AbstractBilinearModel, t::AbstractVector, gs::AbstractVector)
    @lencheck nlp.data.dims[2] t
    @lencheck nlp.data.dims[1] gs

    NLPModels.increment!(nlp, :neval_grad)
    A = nlp.data.A
    mul!(gs, A, t)
    return gs
end

function grad_s!(nlp::BilinearModel, t::AbstractVector, gs::AbstractVector)
    _grad_s!(nlp, t, gs)
end

function grad_s!(nlp::SignedBilinearModel, t::AbstractVector, gs::AbstractVector)
    _grad_s!(nlp, t, gs)
    rmul!(gs, sign(nlp))
end

"""
    grad_t!(nlp, s, gt)

Partial gradient w.r.t. `t`: `gt = σ·A'*s`.
"""
function _grad_t!(nlp::AbstractBilinearModel, s::AbstractVector, gt::AbstractVector)
    @lencheck nlp.data.dims[1] s
    @lencheck nlp.data.dims[2] gt

    NLPModels.increment!(nlp, :neval_grad)
    A = nlp.data.A
    mul!(gt, transpose(A), s)
    return gt
end

function grad_t!(nlp::BilinearModel, s::AbstractVector, gt::AbstractVector)
    _grad_t!(nlp, s, gt)
end

function grad_t!(nlp::SignedBilinearModel, s::AbstractVector, gt::AbstractVector)
    _grad_t!(nlp, s, gt)
    rmul!(gt, sign(nlp))
end
