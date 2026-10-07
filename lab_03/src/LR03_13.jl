# ЛР03, вариант 13. Запуск из папки lab_03: julia src/LR03_13.jl <новый каталог результатов>
# Учебная библиотека хранится отдельно и не изменяется.
using Dates, TOML, Base64, Printf, InteractiveUtils
if !isdefined(Main, :make_scene)
    include(joinpath(@__DIR__, "original", "lab03_lib.jl"))
end
using Plots

function run_lr03(out; engee_version="не зафиксирована", cabinet_version="не зафиксирована")
    isdir(out) && !isempty(readdir(out)) && error("Каталог результатов непуст. Укажите новый каталог.")
    mkpath(out)
    stamp() = string(now(UTC) + Hour(3)) * "+03:00"
    plan = Dict("recorded_at_msk" => stamp(), "variant" => 13,
        "hypothesis" => "Энтропия уменьшится; контраст может увеличиться; средняя яркость изменится слабо.",
        "relative_change_percent_gt" => 5.0, "psnr_db_lt" => 30.0,
        "luma_atol" => 1e-7, "luma_rtol" => 0.0,
        "seeds" => collect(101:105), "bins" => 32,
        "scene" => Dict("n" => 128, "lo" => 0.02, "hi" => 0.30, "noise" => 0.02, "checker" => 0))
    open(joinpath(out, "expectations.toml"), "w") do io
        TOML.print(io, plan)
    end
    started = stamp()
    img = make_scene(101; scene_preset(3)...)
    description = describe_image(img)
    g = luma_images(img)
    gm = luma_manual(img)
    h = apply_factor(g, 1)
    maxdiff = maximum(abs.(g .- gm))
    luma_equal = all(isapprox.(g, gm; atol=1e-7, rtol=0))
    hash1 = array_sha(g)
    hash2 = array_sha(luma_images(make_scene(101; scene_preset(3)...)))
    @assert hash1 == hash2
    @assert luma_equal
    @assert all((0 .<= g) .& (g .<= 1)) && all((0 .<= h) .& (h .<= 1))
    save(joinpath(out, "scene_rgb.png"), RGB{N0f8}.(img))
    save(joinpath(out, "before.png"), Gray{N0f8}.(g))
    save(joinpath(out, "after.png"), Gray{N0f8}.(h))
    # Сохраняются точные Float64 до кодирования PNG; порядок Julia column-major.
    for seed in 101:105
        a = luma_images(make_scene(seed; scene_preset(3)...))
        b = apply_factor(a, 1)
        write(joinpath(out, "before_$(seed).f64"), reinterpret(UInt8, vec(a)))
        write(joinpath(out, "after_$(seed).f64"), reinterpret(UInt8, vec(b)))
    end
    counts_before = hist_counts(g, 32)
    counts_after = hist_counts(h, 32)
    @assert sum(counts_before) == sum(counts_after) == 128^2
    centers = ((1:32) .- 0.5) ./ 32
    ymax = max(maximum(counts_before), maximum(counts_after)) * 1.08
    p1 = bar(centers, counts_before; title="Before", xlabel="Luminance [0,1]",
        ylabel="Pixels", legend=false, color=:gray, bar_width=1/32, xlims=(0,1), ylims=(0,ymax))
    p2 = bar(centers, counts_after; title="After: 4 bits", xlabel="Luminance [0,1]",
        ylabel="Pixels", legend=false, color=:black, bar_width=1/32, xlims=(0,1), ylims=(0,ymax))
    savefig(plot(p1,p2; layout=(1,2), size=(1200,460)), joinpath(out, "histograms.png"))
    open(joinpath(out, "histogram.csv"), "w") do io
        println(io, "bin,left,right,before,after")
        for k in 1:32
            println(io, join((k,(k-1)/32,k/32,counts_before[k],counts_after[k]), ','))
        end
    end
    summary = variant_summary(13)
    rows = [Dict("metric"=>string(r[1]), "base"=>r[2], "s"=>r[3], "after"=>r[4],
        "delta"=>r[5], "ratio"=>r[6], "statistical"=>r[7],
        "relative_percent"=>100*abs(r[5])/abs(r[2]),
        "practical"=>100*abs(r[5])/abs(r[2])>5) for r in summary.rows]
    perseed = [let
        a = luma_images(make_scene(seed; scene_preset(3)...)); b = apply_factor(a,1)
        ma = metrics3(a); mb = metrics3(b)
        Dict("seed"=>seed, "before"=>Dict(string(k)=>ma[k] for k in keys(ma)),
            "after"=>Dict(string(k)=>mb[k] for k in keys(mb)), "mse"=>mse(a,b),
            "psnr_db"=>psnr_db(a,b), "sha256"=>array_sha(a))
    end for seed in 101:105]
    # Дополнительная проверка демо из методички, не результаты варианта 13.
    demo = variant_summary(1)
    demo_description = describe_image(make_scene(101))
    demo_g = luma_images(make_scene(101))
    demo_values = Dict("variant"=>1, "psnr_db"=>demo.psnr, "sha256"=>array_sha(demo_g),
        "luma_maxdiff"=>maximum(abs.(demo_g.-luma_manual(make_scene(101)))),
        "description"=>Dict("size"=>collect(demo_description.size), "min"=>demo_description.min,
            "max"=>demo_description.max, "bytes"=>demo_description.bytes),
        "rows"=>[Dict("metric"=>string(r[1]),"base"=>r[2],"s"=>r[3],"after"=>r[4],"delta"=>r[5],"ratio"=>r[6]) for r in demo.rows])
    results = Dict("started_at_msk"=>started, "finished_at_msk"=>stamp(),
        "variant"=>13, "scene"=>plan["scene"], "seeds"=>plan["seeds"], "bins"=>32,
        "environment"=>Dict("julia"=>string(VERSION),"engee"=>engee_version,"cabinet"=>cabinet_version,
            "Images"=>string(Base.pkgversion(Images)),"Plots"=>string(Base.pkgversion(Plots)),
            "os"=>string(Sys.KERNEL),"arch"=>string(Sys.ARCH),"word_size"=>Sys.WORD_SIZE,
            "endian"=>(ENDIAN_BOM==0x04030201 ? "little" : "big")),
        "description"=>Dict("size"=>collect(description.size),"eltype"=>description.eltype,
            "min"=>description.min,"max"=>description.max,"bytes"=>description.bytes),
        "luma"=>Dict("maxdiff"=>maxdiff,"equal_at_tolerance"=>luma_equal,
            "sha256_first"=>hash1,"sha256_repeat"=>hash2,
            "range_before"=>[minimum(g),maximum(g)],"range_after"=>[minimum(h),maximum(h)],
            "occupied_levels_after"=>length(unique(h))),
        "psnr_mean_db"=>summary.psnr, "rows"=>rows, "per_seed"=>perseed, "demo"=>demo_values)
    open(joinpath(out, "results.toml"), "w") do io
        TOML.print(io, results)
    end
    versionlog = sprint(versioninfo)
    write(joinpath(out, "versioninfo.log"), versionlog)
    open(joinpath(out, "run.log"), "w") do io
        println(io,"Started: ",started,"\nJulia: ",VERSION,"\nVariant: 13\nDescription: ",description)
        println(io,"Luma maxdiff: ",maxdiff,"\nSHA256 first: ",hash1,"\nSHA256 repeat: ",hash2)
        println(io,"PSNR mean dB: ",summary.psnr)
        for r in rows; println(io,r); end
        println(io,"Finished: ",results["finished_at_msk"])
    end
    # Единый транспортный файл позволяет выгрузить все точные байты через интерфейс Engee.
    files = [Dict("name"=>name,"sha256"=>bytes2hex(sha256(read(joinpath(out,name)))),
        "base64"=>base64encode(read(joinpath(out,name)))) for name in sort(readdir(out)) if isfile(joinpath(out,name))]
    bundle = joinpath(dirname(out), "LR03_13_export.toml")
    isfile(bundle) && error("Транспортный файл уже существует; выберите другой родительский каталог.")
    open(bundle, "w") do io; TOML.print(io,Dict("files"=>files)); end
    println(read(joinpath(out,"run.log"),String))
    println("Export: ",bundle,"; bytes: ",filesize(bundle))
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Укажите новый каталог результатов: julia src/LR03_13.jl <каталог>")
    run_lr03(ARGS[1])
end
