@testset "SignedBilinearModel" begin
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2
    m, n = size(A)
    x = [0.5, 0.5, 0.5, 0.5, 0.5]

    # Positive sign should match BilinearModel
    nlp_pos = SignedBilinearModel(A; sign=Int8(1))
    nlp_ref = BilinearModel(A)

    @test nlp_pos.meta.nvar == m + n
    @test all(nlp_pos.meta.lvar .== 0)
    @test all(nlp_pos.meta.uvar .== 1)
    @test nlp_pos.sign == Int8(1)

    @test obj(nlp_pos, x) ≈ obj(nlp_ref, x)

    g_pos = similar(x)
    g_ref = similar(x)
    grad!(nlp_pos, x, g_pos)
    grad!(nlp_ref, x, g_ref)
    @test g_pos ≈ g_ref

    val_pos, _ = objgrad!(nlp_pos, x, g_pos)
    val_ref, _ = objgrad!(nlp_ref, x, g_ref)
    @test val_pos ≈ val_ref
    @test g_pos ≈ g_ref

    v = [1.0, 0.0, 0.0, 1.0, 0.0]
    Hv_pos = similar(x)
    Hv_ref = similar(x)
    hprod!(nlp_pos, x, v, Hv_pos)
    hprod!(nlp_ref, x, v, Hv_ref)
    @test Hv_pos ≈ Hv_ref

    # Negative sign should negate everything
    nlp_neg = SignedBilinearModel(A; sign=Int8(-1))
    @test nlp_neg.sign == Int8(-1)

    @test obj(nlp_neg, x) ≈ -obj(nlp_ref, x)

    g_neg = similar(x)
    grad!(nlp_neg, x, g_neg)
    @test g_neg ≈ -g_ref

    val_neg, _ = objgrad!(nlp_neg, x, g_neg)
    @test val_neg ≈ -val_ref
    @test g_neg ≈ -g_ref

    Hv_neg = similar(x)
    hprod!(nlp_neg, x, v, Hv_neg)
    @test Hv_neg ≈ -Hv_ref

    # obj_weight composes with sign
    hprod!(nlp_neg, x, v, Hv_neg, obj_weight=3.0)
    @test Hv_neg ≈ -3.0 .* Hv_ref
end

@testset "SignedBilinearModel Partial Gradients" begin
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2
    m, n = size(A)

    s = [0.5, 0.25, 0.75]
    t = [0.4, 0.6]

    nlp_pos = SignedBilinearModel(A; sign=Int8(1))
    nlp_neg = SignedBilinearModel(A; sign=Int8(-1))

    # Positive sign matches A*t and A'*s
    gs_pos = zeros(m)
    gt_pos = zeros(n)
    grad_s!(nlp_pos, t, gs_pos)
    grad_t!(nlp_pos, s, gt_pos)
    @test gs_pos ≈ A * t
    @test gt_pos ≈ A' * s

    # Negative sign negates
    gs_neg = zeros(m)
    gt_neg = zeros(n)
    grad_s!(nlp_neg, t, gs_neg)
    grad_t!(nlp_neg, s, gt_neg)
    @test gs_neg ≈ -(A * t)
    @test gt_neg ≈ -(A' * s)

    # Consistency with full gradient
    x = [s; t]
    g = zeros(m + n)
    grad!(nlp_neg, x, g)
    @test gs_neg ≈ g[1:m]
    @test gt_neg ≈ g[m+1:m+n]
end

@testset "SignedBilinearModel Construction" begin
    # Integer matrix auto-converts to float
    A_int = [1 2; 3 4]
    nlp = SignedBilinearModel(A_int; sign=Int8(-1))
    @test nlp.data.A isa Matrix{Float64}
    @test nlp.sign == Int8(-1)

    # Default sign is +1
    nlp_default = SignedBilinearModel([1.0 2.0; 3.0 4.0])
    @test nlp_default.sign == Int8(1)

    # Int argument requires Int8
    @test_throws TypeError SignedBilinearModel([1.0 2.0; 3.0 4.0]; sign=-1)
end
