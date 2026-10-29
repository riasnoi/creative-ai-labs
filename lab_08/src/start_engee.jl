# Выполняется из отдельной папки ЛР08 с загруженными Python-исходниками.
using SHA, Dates, InteractiveUtils
@assert !isdir("artifacts") "Для повторного опыта нужна новая папка"
PY = "/user/.user_venvs/py3.11/bin/python3"
open("engee_environment.log", "w") do io
    println(io, "Recorded UTC: ", now(UTC))
    println(io, "VERSION = ", VERSION)
    versioninfo(io)
    for name in ["lab07_lib.py", "lab08_lib.py", "lab08_demo.py", "run_lab08.py", "start_engee.jl"]
        println(io, name, " ", filesize(name), " bytes SHA256 ", bytes2hex(sha256(read(name))))
    end
    run(pipeline(`$PY -c "import sys,torch,numpy,PIL; print(sys.version); print('torch',torch.__version__); print('CUDA',torch.cuda.is_available()); print('numpy',numpy.__version__); print('Pillow',PIL.__version__)"`, stdout=io, stderr=io))
end
println(read("engee_environment.log", String))
proc_lr08 = run(pipeline(`$PY -u run_lab08.py`, stdin=devnull, stdout="process.log", stderr="process_errors.log"), wait=false)
println("Обучение запущено; process.log содержит текущий ход опыта.")
