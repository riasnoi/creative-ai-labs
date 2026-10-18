# ЛР 6, демонстрационный вариант 1: бриф «афиша фестиваля робототехники» × число шагов (1 → 4), серия seed 101…105.
import lab05_lib as L
import lab06_lib as M

pipe = L.load_pipe()
clip = M.load_clip()
res = M.variant_summary(pipe, clip, 1)
print("Вариант", res["variant"], "фактор", res["factor"])
for m, a, s, b, d, ratio, sig in res["rows"]:
    print(f"{m:9s} A={a:.4f} s={s:.4f} B={b:.4f} d={d:+.4f} |d|/s={ratio:.2f} 2s={'да' if sig else 'нет'}")

# Намеренная ошибка: ожидание, что guidance_scale = 1.0 «включает» guidance у SD Turbo.
p = L.BRIEFS[0]
x0 = L.generate(pipe, p, 101, guidance=0.0)[0]
x1 = L.generate(pipe, p, 101, guidance=1.0)[0]
print("guidance 1.0: результат совпал с guidance 0.0 бит в бит:", L.sha(L.png_bytes(x0)) == L.sha(L.png_bytes(x1)), "| CFG активен:", pipe.do_classifier_free_guidance)
x3 = L.generate(pipe, p, 101, guidance=3.0)[0]
print("guidance 3.0: CFG активен:", pipe.do_classifier_free_guidance, "| PSNR относительно guidance 0.0, дБ:", round(L.psnr_db(x0, x3), 2))
