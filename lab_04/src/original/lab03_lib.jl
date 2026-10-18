# ЛР 3. Подготовка изображений и визуализация в Engee. Библиотека учебных функций.
# Среда проверки: Engee 26.8.2-H3, Julia 1.12.4 (19.09.2026). Требуются пакеты Images, Random, Statistics, SHA.
# Тексты функций ниже вводились в командную строку Engee без комментариев; файл целиком через include автором не запускался.

using Images, Random, Statistics, SHA

# Синтетическая сцена: диагональный градиент + круг + (необязательно) шахматная структура + гауссов шум.
# Все случайные числа берутся из локального генератора MersenneTwister(seed).
function make_scene(seed::Integer; n::Int=128, lo::Float64=0.15, hi::Float64=0.85, noise::Float64=0.02, checker::Int=0)
    rng = MersenneTwister(seed)
    img = Array{RGB{Float64}}(undef, n, n)
    c, rad = n ÷ 2, n ÷ 4
    for i in 1:n, j in 1:n
        t = (i + j) / (2n)
        r = lo + (hi - lo) * t
        g = lo + (hi - lo) * (1 - t)
        b = lo + (hi - lo) * j / n
        if checker > 0 && (div(i - 1, checker) + div(j - 1, checker)) % 2 == 0
            m = (r + g + b) / 3
            r, g, b = m, m, m
        end
        if (i - c)^2 + (j - c)^2 <= rad^2
            r, g, b = hi, lo, lo
        end
        img[i, j] = RGB(clamp(r + noise * randn(rng), 0, 1), clamp(g + noise * randn(rng), 0, 1), clamp(b + noise * randn(rng), 0, 1))
    end
    return img
end

# Свойства изображения: размер, тип элемента, диапазон значений, объём массива в байтах.
describe_image(img) = (size = size(img), eltype = string(eltype(img)), min = Float64(minimum(channelview(img))), max = Float64(maximum(channelview(img))), bytes = sizeof(img))

# Яркость: Rec.601 «вручную» и стандартным преобразованием Images (значения Float64 в [0,1]).
luma_manual(img) = [0.299 * red(p) + 0.587 * green(p) + 0.114 * blue(p) for p in img]
luma_images(img) = Float64.(Gray.(img))

# Квантование яркости [0,1] до заданного числа бит и обратно.
quantize(x::AbstractArray, bits::Int) = round.(x .* (2^bits - 1)) ./ (2^bits - 1)

# Среднеквадратичная ошибка и PSNR в дБ для массивов [0,1] (maxval = 1).
mse(a, b) = mean((Float64.(a) .- Float64.(b)) .^ 2)
psnr_db(a, b) = (m = mse(a, b); m == 0 ? Inf : 10 * log10(1 / m))

# Контрольная сумма массива (SHA-256 по байтам Float64) — для проверки воспроизводимости.
array_sha(x::AbstractArray) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(x)))))

# Гистограмма в bins равных интервалах [0,1] и энтропия гистограммы (бит на отсчёт).
hist_counts(x::AbstractArray, bins::Int) = (cn = zeros(Int, bins); for v in x; k = min(bins, max(1, ceil(Int, v * bins))); cn[k] += 1; end; cn)
hist_entropy(x::AbstractArray, bins::Int) = (cn = hist_counts(x, bins); pr = cn ./ sum(cn); -sum(q * log2(q) for q in pr if q > 0))

# Пять сцен (контексты вариантов) и шесть воздействий (факторы вариантов).
scene_preset(c) = c == 1 ? (lo=0.15, hi=0.85, noise=0.02, checker=0) : c == 2 ? (lo=0.40, hi=0.60, noise=0.02, checker=0) : c == 3 ? (lo=0.02, hi=0.30, noise=0.02, checker=0) : c == 4 ? (lo=0.15, hi=0.85, noise=0.08, checker=0) : (lo=0.15, hi=0.85, noise=0.02, checker=8)
png_roundtrip(g) = (p = joinpath(tempdir(), "lab03_rt.png"); save(p, Gray{N0f8}.(g)); Float64.(load(p)))
apply_factor(g, k) = k == 1 ? quantize(g, 4) : k == 2 ? imresize(imresize(g, ratio=0.5), size(g)) : k == 3 ? imresize(imresize(g, ratio=0.25), size(g)) : k == 4 ? (g .- minimum(g)) ./ (maximum(g) - minimum(g)) : k == 5 ? g .^ 0.5 : png_roundtrip(g)
metrics3(x) = (mean = mean(x), contrast = std(x), entropy = hist_entropy(x, 32))

# Сводка варианта n = 1..30 по серии seed: для метрик mean, contrast, entropy — база, разброс s, значение после воздействия, Δ, |Δ|/s, признак |Δ| > 2s.
function variant_summary(n; seeds=101:105)
    q, r = divrem(n - 1, 6); c, f = q + 1, r + 1
    mb = []; ma = []; ps = Float64[]
    for s in seeds
        g = luma_images(make_scene(s; scene_preset(c)...)); h = apply_factor(g, f)
        push!(mb, metrics3(g)); push!(ma, metrics3(h)); push!(ps, psnr_db(g, h))
    end
    out = Any[]
    for m in (:mean, :contrast, :entropy)
        b = [x[m] for x in mb]; a = [x[m] for x in ma]
        d = mean(a) - mean(b); sb = std(b)
        push!(out, (m, mean(b), sb, mean(a), d, abs(d) / sb, abs(d) > 2 * sb))
    end
    return (variant=n, ctx=c, fac=f, psnr=mean(ps), rows=out)
end
