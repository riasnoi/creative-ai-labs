# ЛР04. Вариант 13: усреднение 5×5 при σ_n = 0,10.
# Из папки lab_04: julia src/LR04_13.jl <новый каталог результатов>
using Dates, TOML, Base64, Printf, InteractiveUtils
include(joinpath(@__DIR__, "original", "lab04_lib.jl"))

"""Выполнить серию, сохранив точные Float64, PNG и паспорт параметров."""
function run_lr04(out; engee_version="не зафиксирована", cabinet_version="не зафиксирована")
    isdir(out) && !isempty(readdir(out)) && error("Укажите новый пустой каталог результатов")
    mkpath(out)
    stamp() = string(now(UTC) + Hour(3)) * "+03:00"
    plan = Dict("recorded_at_msk"=>stamp(), "variant"=>13,
        "hypothesis"=>"Усреднение 5×5 при шуме 0,10 увеличит PSNR минимум на 1 дБ; энергия границ и контраст уменьшатся.",
        "practical_threshold_db"=>1.0, "seeds"=>collect(101:105),
        "scene"=>Dict("n"=>128,"bg"=>[0.10,0.40],"disc"=>0.75,"noise"=>0.10,"checker"=>0),
        "operation"=>Dict("name"=>"box_mean","window"=>[5,5],"weight"=>1/25,"border"=>"replicate"),
        "additional_checks"=>["demo_variant_1","neighbor_variant_7","thresholds_0.5_1_2_db"])
    open(joinpath(out,"expectations.toml"),"w") do io; TOML.print(io,plan); end
    started = stamp()
    @assert size(make_gray4(101)) == (128,128)
    clean = clean4(3)
    write(joinpath(out,"reference.f64"), reinterpret(UInt8,vec(clean)))
    save(joinpath(out,"reference.png"), Gray{N0f8}.(clean))
    records = Any[]
    for seed in 101:105
        x = make_gray4(seed; scene4(3)...)
        y, base, ref = pair4(x,clean,1)
        @assert ref == clean && base == x
        @assert size(y) == (128,128) && all(0 .<= y .<= 1)
        # Явная проверка продолжения крайних пикселей для усредняющего фильтра.
        manual = [mean(x[clamp(i+di,1,128),clamp(j+dj,1,128)] for di in -2:2, dj in -2:2) for i in 1:128,j in 1:128]
        err = maximum(abs.(manual .- y)); @assert err < 1e-12
        @assert array_sha(x) == array_sha(make_gray4(seed; scene4(3)...))
        for (name,arr) in (("before",x),("after",y))
            write(joinpath(out,"$(name)_$(seed).f64"),reinterpret(UInt8,vec(arr)))
            seed == 101 && save(joinpath(out,"$(name).png"),Gray{N0f8}.(arr))
        end
        push!(records,Dict("seed"=>seed,
            "before"=>Dict("psnr"=>psnr_db(x,ref),"edge"=>edge_energy(x),"contrast"=>std(x),"mse"=>mse(x,ref)),
            "after"=>Dict("psnr"=>psnr_db(y,ref),"edge"=>edge_energy(y),"contrast"=>std(y),"mse"=>mse(y,ref)),
            "delta_psnr"=>psnr_db(y,ref)-psnr_db(x,ref),
            "border_check_max_abs"=>err,"before_sha256"=>array_sha(x),"after_sha256"=>array_sha(y),
            "clipped_zero"=>count(==(0),x),"clipped_one"=>count(==(1),x)))
    end
    rowdict(r) = Dict("metric"=>string(r[1]),"base"=>r[2],"s"=>r[3],"after"=>r[4],"delta"=>r[5],"ratio"=>r[6],"exceeds_2s"=>r[7])
    summary = variant_summary4(13)
    for r in summary.rows
        m=string(r[1]); b=[x["before"][m] for x in records]; a=[x["after"][m] for x in records]
        @assert isapprox(mean(b),r[2];atol=1e-13) && isapprox(mean(a),r[4];atol=1e-13)
    end
    delta = summary.rows[1][5]
    classify(d,t) = d >= t ? "полезна" : d <= -t ? "вредна" : "нейтральна"
    demo = variant_summary4(1)
    demo_expected = [33.982103137714965 0.07450305641014086 29.19575338868147 -4.786349749033494;
                     0.01692548814288977 0.00012750148456890567 0.008424262473284544 -0.008501225669605225;
                     0.20774642267754806 0.00017156629178605263 0.20065402164009138 -0.007092401037456675]
    demo_diff = maximum(abs(demo.rows[i][j+1]-demo_expected[i,j]) for i in 1:3,j in 1:4)
    neighbor = variant_summary4(7)
    results = Dict("variant"=>13,"started_at_msk"=>started,"finished_at_msk"=>stamp(),
        "environment"=>Dict("julia"=>string(VERSION),"engee"=>engee_version,"cabinet"=>cabinet_version,
            "Images"=>string(Base.pkgversion(Images)),"os"=>string(Sys.KERNEL),"arch"=>string(Sys.ARCH),
            "endian"=>(ENDIAN_BOM==0x04030201 ? "little" : "big")),
        "seeds"=>collect(101:105),"scene"=>plan["scene"],"operation"=>plan["operation"],
        "reference"=>Dict("name"=>"clean4(3)","sha256"=>array_sha(clean),"edge"=>edge_energy(clean),"contrast"=>std(clean)),
        "rows"=>rowdict.(summary.rows),"per_seed"=>records,"theoretical_psnr_db"=>-20log10(0.10),
        "noise_psnr_minus_theory_db"=>summary.rows[1][2]+20log10(0.10),
        "classification"=>classify(delta,1.0),"threshold_db"=>1.0,
        "sensitivity"=>[Dict("threshold_db"=>t,"classification"=>classify(delta,t)) for t in (0.5,1.0,2.0)],
        "demo"=>Dict("variant"=>1,"rows"=>rowdict.(demo.rows),"max_abs_difference"=>demo_diff,"matches_at_1e_10"=>demo_diff<1e-10),
        "neighbor"=>Dict("variant"=>7,"noise"=>0.05,"rows"=>rowdict.(neighbor.rows),"classification"=>classify(neighbor.rows[1][5],1.0)))
    open(joinpath(out,"results.toml"),"w") do io; TOML.print(io,results); end
    open(joinpath(out,"metrics_by_seed.csv"),"w") do io
        println(io,"seed,psnr_before,psnr_after,delta_psnr,edge_before,edge_after,contrast_before,contrast_after")
        for r in records
            println(io,join((r["seed"],r["before"]["psnr"],r["after"]["psnr"],r["delta_psnr"],r["before"]["edge"],r["after"]["edge"],r["before"]["contrast"],r["after"]["contrast"]),','))
        end
    end
    write(joinpath(out,"versioninfo.log"),sprint(versioninfo))
    open(joinpath(out,"run.log"),"w") do io
        println(io,"Started: ",started,"\nJulia: ",VERSION,"\nSize check: ",size(make_gray4(101)))
        for r in summary.rows; println(io,r); end
        println(io,"PSNR theory: ",results["theoretical_psnr_db"],"\nClassification: ",results["classification"])
        println(io,"Demo maximum difference: ",demo_diff,"\nNeighbor: ",neighbor)
        println(io,"Finished: ",results["finished_at_msk"])
    end
    # Транспорт всех результатов через один файл без потери исходных байтов.
    bundle = joinpath(dirname(out),basename(out)*"_export.toml")
    isfile(bundle) && error("Файл экспорта уже существует")
    files = [Dict("name"=>n,"sha256"=>bytes2hex(sha256(read(joinpath(out,n)))),"base64"=>base64encode(read(joinpath(out,n)))) for n in sort(readdir(out)) if isfile(joinpath(out,n))]
    open(bundle,"w") do io; TOML.print(io,Dict("files"=>files)); end
    println(read(joinpath(out,"run.log"),String)); println("Export: ",bundle)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)==1 || error("Использование: julia src/LR04_13.jl <новый каталог результатов>")
    run_lr04(ARGS[1])
end
