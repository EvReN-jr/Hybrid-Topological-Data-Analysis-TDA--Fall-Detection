# Seans Notları — 2026-05-16 (akşam)

**Klasör:** `/home/evo/YL_TEZ/Move/Graduate_Sections/`

---

## Bu seansta ne yapıldı

`methodology_section.tex` baştan yeniden yazıldı. Hedef: `kenan_boyabatli_proposal.tex` stilinde, YL Matematik Mühendisliği tezine yakışır derinlikte, math-article formatında methodology bölümü.

**Çıktı:**
- `methodology_section.tex` — 1312 satır
- `methodology_section.pdf` — 30 sayfa, 408 KB

---

## Kullanıcı kararları (anket)

| Soru | Cevap |
|------|-------|
| Yaklaşım | **Tam yeniden yazım** (mevcut 1000 satırlık dosya tamamen değiştirildi) |
| Bayes derinliği | **Tam matematiksel** (TPE türetişi, EI proposition'ı, nested CV) |
| Hipotezler (H1–H4) methodology'de yer alsın mı? | **Hayır**, proposal'da yeterli — methodology sadece teorik altyapı + test mekaniği içersin |

---

## Methodology bölüm yapısı (yeni hali)

1. **Formal Problem Setting** — $K \in \{\text{MobiFall, SisFall, FAD\_40Hz}\}$, $\lambda \in \Lambda$ tuple, $\phi_\lambda : w \to \R^{24}$, subject-general vs personalized karar kuralı
2. **Signal Representation** — SMV $\mathrm{SO}(3)$-invariance proposition'ı, Nyquist–Shannon teoremi, 50 Hz seçim gerekçesi (spectral / integer divisibility / computational), window extraction, standard scaling
3. **Phase-Space Reconstruction** — Delay embedding tanımı, Takens teoremi, Sauer–Yorke–Casdagli fractal extension, heuristic status remark, geometric interpretation (fall ↔ non-recurrent, ADL ↔ loop-forming), filtration scale control $\varepsilon_{\max}$
4. **Topological Preliminaries** — topolojik uzay, homotopi, abstract/geometric simplicial complex, geometric realisation
5. **Simplicial Homology** — chain group, boundary $\partial_k$, $\partial^2 = 0$ TAM İSPATLA, homology groups, Betti numbers, Euler–Poincaré (ispatlı), functoriality
6. **Persistent Homology** — filtration, Vietoris–Rips/Alpha/SparseRips, nerve teoremi (Borsuk), persistence module (functorial tanım), interleaving distance, structure teoremi (Crawley-Boevey), persistence diagram, bottleneck/Wasserstein, Cohen-Steiner stability, algebraic stability (Chazal et al.), Hausdorff stability corollary (ispat eskizi), reduction algoritması (uniqueness proposition'ı)
7. **Feature Vectorisation** — 10 PD istatistiği (H0, H1) + 4 sinyal özelliği = 24-boyutlu vektör
8. **Classification Models** — LR (konvekslik proposition'ı), SVM primal/dual/KKT türetiş, Platt scaling, class-weighted empirical risk
9. **Bayesian Hyperparameter Optimisation (YENİ)** — SMBO formal tanım, expected improvement, TPE'nin density-ratio proposition'ı (EI ∝ $\ell/g$), $\Lambda$ search space, nested LOSO + inner CV protokolü
10. **Evaluation Framework** — $F_2$ recall-weighting proposition'ı (4:1 oran), trivial baseline remark, decision threshold, üç protokol (Naive, General/LOSO, Personal), threshold sweep
11. **Statistical Inference (YENİ, derinleştirilmiş)** — $t$-CI teoremi, paired $t$-test, sign test (small-$n$ robustness), paired bootstrap (B=10000), permutation test (sign-flipping), distributional analysis of personalization, cross-applied $\lambda^*$ matrix, headline maxima correction
12. **Computational Infrastructure** — UHeM SLURM 4025462026, 2× Xeon Gold 6148, 190 GB RAM, 3h 45 min, ~10 GB peak

---

## Stil notları (kalıcı tercih)

- **Üslup referansı:** `kenan_boyabatli_proposal.tex`
- **Teorem-İspat:** Definition/Theorem/Proposition/Lemma/Corollary/Remark environment'ları kullanıldı
- **Hipotezler:** Proposal'da kalmalı; methodology onları tekrar etmiyor, sadece test prosedürlerini (paired-t, sign, bootstrap, permutation) tanımlıyor

---

## Dosyalar — şu anki durum

| Dosya | Satır/Sayfa | Durum |
|-------|-------------|-------|
| `methodology_section.tex` | 1312 satır | **GÜNCEL** (bu seans) |
| `methodology_section.pdf` | 30 sayfa | **GÜNCEL** |
| `experiment_section.tex` | 565 satır | Önceki seans (2026-05-16 öğlen) |
| `experiment_section.pdf` | ~10 sayfa | Önceki seans |
| `TEZ_CONTEXT.md` | — | Bağlam dokümanı |

**Üslup referansı:** `/home/evo/YL_TEZ/Move/Graduate_TDA_Fall_Detection/kenan_boyabatli_proposal.tex`

---

## Sonraki olası adımlar (TEZ_CONTEXT.md'den)

Önerilen sıra:
1. ~~Methodology~~ — TAMAMLANDI
2. **Introduction** — Motivasyon (yaşlı nüfus, WHO), TDA'nın potansiyeli, mevcut yöntemlerin kısıtları, contribution'lar (proposal §1'den adapte)
3. **Related Work** — Threshold-based, statistical features, spectral features, deep learning, TDA for time series
4. **Results & Discussion (genişletilmiş)** — Baseline karşılaştırması (henüz yok), ablation studies, hypothesis test özet
5. **Conclusion** — Katkılar, limitasyonlar, gelecek çalışmalar

---

## Açık sorular / gelecek seansa not

- Methodology'deki ispatlar yeterli mi yoksa daha çok detay isteniyor mu? (Stability teoremi şu an "proof sketch" olarak; istenirse tam ispatlar eklenebilir)
- Baseline (threshold-based, statistical, spectral) karşılaştırması henüz yapılmadı — Results & Discussion'a önce mi Related Work'e mi yazılsın?
- Ablation studies (proposal §4.4'te planlandı) henüz koşulmadı — bunun için ayrı bir deney koşusu gerekli mi?
- `newtxtext` paketi sistemde yok; methodology default fontlarla derleniyor (commented out)

---

## Derleme komutu

```bash
cd /home/evo/YL_TEZ/Move/Graduate_Sections
pdflatex -interaction=nonstopmode methodology_section.tex
pdflatex -interaction=nonstopmode methodology_section.tex   # cross-ref için ikinci pass
```
