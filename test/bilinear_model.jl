@testset "BilinearModel" begin
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2
    nlp = BilinearModel(A)
    m, n = size(A)

    @test nlp.meta.nvar == m + n
    @test all(nlp.meta.lvar .== 0)
    @test all(nlp.meta.uvar .== 1)

    x = [0.5, 0.5, 0.5, 0.5, 0.5]  # s = [0.5,0.5,0.5], t = [0.5,0.5]

    # obj: s'*A*t = [0.5,0.5,0.5]*A*[0.5,0.5] = 0.5*(3+7) + 0.5*(3+7)... computed directly
    expected_obj = dot(x[1:m], A, x[m+1:m+n])
    @test obj(nlp, x) ≈ expected_obj

    # grad!
    g = similar(x)
    grad!(nlp, x, g)
    @test g[1:m]     ≈ A * x[m+1:m+n]
    @test g[m+1:m+n] ≈ A' * x[1:m]

    # objgrad!
    g2 = similar(x)
    val, _ = objgrad!(nlp, x, g2)
    @test val ≈ expected_obj
    @test g2 ≈ g

    # hprod!: H*v where H = [0 A; A' 0]
    v = [1.0, 0.0, 0.0, 1.0, 0.0]
    Hv = similar(x)
    hprod!(nlp, x, v, Hv)
    @test Hv[1:m]     ≈ A * v[m+1:m+n]
    @test Hv[m+1:m+n] ≈ A' * v[1:m]

    # obj_weight scaling
    hprod!(nlp, x, v, Hv, obj_weight = 2.0)
    @test Hv[1:m]     ≈ 2 * A * v[m+1:m+n]
    @test Hv[m+1:m+n] ≈ 2 * A' * v[1:m]
end

@testset "Partial Gradients" begin
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2
    nlp = BilinearModel(A)
    m, n = size(A)

    s = [0.5, 0.25, 0.75]
    t = [0.4, 0.6]

    # grad_s!
    gs = zeros(m)
    grad_s!(nlp, t, gs)
    @test gs ≈ A * t

    # grad_t!
    gt = zeros(n)
    grad_t!(nlp, s, gt)
    @test gt ≈ A' * s

    # consistency with grad!
    x = [s; t]
    g = zeros(m + n)
    grad!(nlp, x, g)
    @test gs ≈ g[1:m]
    @test gt ≈ g[m+1:m+n]

    # length checks
    @test_throws DimensionError grad_s!(nlp, zeros(m), gs)  # wrong length for t
    @test_throws DimensionError grad_t!(nlp, zeros(n), gt)  # wrong length for s
end
