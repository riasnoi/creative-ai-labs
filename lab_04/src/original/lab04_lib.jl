# ЛР 4. Исследование операций обработки изображений в Engee. Библиотека учебных функций.
# Требуется: среда Engee (проверено в версии 26.8.2-H3, Julia 1.12.4), пакеты Images, Random, Statistics, SHA;
# файл lab03_lib.jl из ЛР 3 в той же папке (из него берутся функции psnr_db и mse).

using Images, Random, Statistics, SHA
isdefined(Main, :psnr_db) || include("lab03_lib.jl")

# Серая сцена n×n: тёмный диагональный градиент фона + яркий круг (диск) + (необязательно) шахматная добавка + гауссов шум.
# Случайные числа — из локального генератора MersenneTwister(seed). При noise = 0 получается «чистая» сцена (эталон).
function make_gray4(seed::Integer; n::Int=128, bg::Tuple{Float64,Float64}=(0.10, 0.40), disc::Float64=0.75, noise::Float64=0.05, checker::Int=0)
    rng = MersenneTwister(seed)
    x = zeros(n, n)
    for i in 1:n, j in 1:n
        v = bg[1] + (bg[2] - bg[1]) * (i + j) / (2n)
        if checker > 0 && (div(i - 1, checker) + div(j - 1, checker)) % 2 == 0
            v += 0.10
        end
        if (i - n ÷ 2)^2 + (j - n ÷ 2)^2 <= (n ÷ 4)^2
            v = disc
        end
        x[i, j] = clamp(v + noise * randn(rng), 0, 1)
    end
    return x
end

# Истинная маска диска (эталон сегментации).
truth_mask(n::Int=128) = [(i - n ÷ 2)^2 + (j - n ÷ 2)^2 <= (n ÷ 4)^2 for i in 1:n, j in 1:n]

# Пять сцен (контексты вариантов).
scene4(c) = c == 1 ? (noise=0.02, checker=0, disc=0.75, bg=(0.10, 0.40)) : c == 2 ? (noise=0.05, checker=0, disc=0.75, bg=(0.10, 0.40)) : c == 3 ? (noise=0.10, checker=0, disc=0.75, bg=(0.10, 0.40)) : c == 4 ? (noise=0.05, checker=8, disc=0.75, bg=(0.10, 0.40)) : (noise=0.03, checker=0, disc=0.55, bg=(0.35, 0.50))
clean4(c) = make_gray4(1; merge(scene4(c), (noise=0.0,))...)

# Операции.
gauss4(x, s) = imfilter(x, Kernel.gaussian(s))
box4(x, k) = mapwindow(mean, x, (k, k))
unsharp4(x) = clamp.(x .+ (x .- gauss4(x, 1)), 0, 1)
sobelmag(x) = (g = imgradients(x, KernelFactors.sobel); hypot.(g[1], g[2]))
edge_energy(x) = mean(sobelmag(x))
otsu_mask(x) = x .> otsu_threshold(Gray.(x))
open_mask(bw) = dilate(erode(bw))
iou(a, b) = sum(a .& b) / sum(a .| b)

# Пара «результат / базовый вариант / эталон» для воздействия k = 1..6 (нормированные значения в [0,1]).
function pair4(x, clean, k)
    if k <= 4
        res = k == 1 ? box4(x, 5) : k == 2 ? gauss4(x, 1) : k == 3 ? gauss4(x, 3) : unsharp4(x)
        return (res, x, clean)
    elseif k == 5
        cm = maximum(sobelmag(clean))
        nrm(z) = min.(sobelmag(z) ./ cm, 1.0)
        return (nrm(gauss4(x, 1)), nrm(x), nrm(clean))
    else
        bw = otsu_mask(x)
        return (Float64.(open_mask(bw)), Float64.(bw), Float64.(truth_mask(size(x, 1))))
    end
end

# Сводка варианта n = 1..30 по серии seed: метрики psnr (относительно эталона), edge (энергия границ результата), contrast (стандартное отклонение).
# Для каждой: база, разброс s, значение после воздействия, Δ, |Δ|/s, признак |Δ| > 2s. Для воздействия 6 дополнительно IoU с истинной маской.
function variant_summary4(n; seeds=101:105)
    q, r = divrem(n - 1, 6); c, k = q + 1, r + 1
    clean = clean4(c)
    B = Dict(:psnr => Float64[], :edge => Float64[], :contrast => Float64[])
    A = Dict(:psnr => Float64[], :edge => Float64[], :contrast => Float64[])
    ious = Float64[]
    for s in seeds
        x = make_gray4(s; scene4(c)...)
        res, base, ref = pair4(x, clean, k)
        push!(B[:psnr], psnr_db(base, ref)); push!(A[:psnr], psnr_db(res, ref))
        push!(B[:edge], edge_energy(base)); push!(A[:edge], edge_energy(res))
        push!(B[:contrast], std(base)); push!(A[:contrast], std(res))
        k == 6 && push!(ious, iou(open_mask(otsu_mask(x)), truth_mask(size(x, 1))))
    end
    rows = Any[]
    for m in (:psnr, :edge, :contrast)
        d = mean(A[m]) - mean(B[m]); sb = std(B[m])
        push!(rows, (m, mean(B[m]), sb, mean(A[m]), d, abs(d) / sb, abs(d) > 2 * sb))
    end
    return (variant=n, ctx=c, fac=k, iou=isempty(ious) ? NaN : mean(ious), rows=rows)
end
