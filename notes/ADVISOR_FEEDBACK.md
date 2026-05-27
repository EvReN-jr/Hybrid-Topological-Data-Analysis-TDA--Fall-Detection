# Danışman Geri Bildirimi — Uygulama Notları

**Alındı:** 2026-05-26  
**Bağlam:** Makale odak noktası = Eşik değer kararı (Threshold Decision) ikili sınıflandırmada.

---

## Uygulanan Değişiklikler (experiment + methodology)

### ✅ Methodology — Bayesian Opt karşılaştırması

`methodology_section.tex`'te, TPE alt bölümünün sonundaki "Why TPE rather than GP" remark'ı
**kaldırıldı** ve yerine tam bir `\subsection{Comparison with alternative search strategies}`
eklendi. İçerik:

- **Grid search**: $|\Lambda_\text{grid}|=17{,}820$ konfigürasyon hesabı (Denklem), SisFall için
  ~43 gün tahmin → infeasible.
- **Random search**: Tutarlı ama tarihi kullanmıyor, budget'ın yalnızca %1.1'ini kapsıyor.
- **GP-Bayes**: Discrete/conditional alanda kernel yok, $O(t^3)$ maliyet → uygunsuz.
- **TPE (adopted)**: $O(t)$ güncelleme, conditional ağacı native, EI∝ℓ/g teorik garantisi.

### ✅ Experiment — Kapsamlı Pipeline Diyagramı

`experiment_section.tex`'teki basit 2-satır diyagram **kapsamlı 3-satır + optimizasyon döngüsü**
versiyonuyla değiştirildi:

- **Row 1 (mavi arka plan)**: Preprocessing — Sensor Data → SMV(t) → Resample 50Hz → Window Extract
- **Row 2 (turuncu, kesik)**: Feature Extraction (λ-bağımlı) — Delay Embed(m,τ,r) → Filtration(c,ε) → Pers.Hom. H₀,H₁ → Vectorise φ_λ∈ℝ²⁴
- **Row 3 (yeşil)**: Classification — Std Scale → Classifier(LR/SVM) → p̂(w) → Threshold τ₀ → Decision
- **Bayesian loop (mavi, altta)**: TPE kutusu + eğri ok (bayes.west → emb.south) λ* ile etiketli

### ✅ Experiment — Threshold Analizi Genişletmesi

`experiment_section.tex` threshold bölümüne iki yeni paragraf eklendi:

1. **Computational cost of threshold personalisation**: No retraining gerektirmez.
   $O(|\mathcal{T}| \cdot |W_s|)$ maliyet → Bayesian search'den $10^3\times$ daha hızlı.
   **Key contribution**: Inference-time, zero-retraining personalization.

2. **Statistical analysis of threshold heterogeneity**: Matematiksel uniform-shift argümanı
   neden başarısız olduğu, $\Delta F_2(s)$ dağılımının sağ çarpık olduğunun anlamı,
   $\tau^*(s)$'nin subject-specific olduğunun kanıtı. H4 hipoteziyle bağlantı.

---

## Yazılacak Bölümler İçin Geri Bildirim Notları

### Introduction

**Mühendislik perspektifi ön plana çıkarılmalı:**
- WHO istatistikleri + yaşlı nüfus artışı = deployment motivasyonu
- Mevcut sistemlerin eksikliği: "trains once, deploys forever" anlayışı personalizasyonu ihmal eder
- TDA'nın engineering avantajı: teorik güvenceli (stability teoremi), hesaplama açısından ölçeklenebilir

**Matematiksel + istatistiksel denge:**
- Formal contribution list: $\phi_\lambda$ haritası, threshold sweep protokolü, LOSO evaluation framework
- İstatistiksel: heterogeneous personalization → kişiye özgü eşik ayarı, tek sabit eşikten daha iyi

**Threshold'un "ucuzluk" katkısı Introduction'da da belirtilmeli:**
> "...a post-hoc threshold adjustment that requires no model retraining and is deployable
> at inference time from a small per-subject calibration set."

---

### Related Work

**Threshold-based detectors (matematiksel):**
- Fixed-threshold methods: $\|\mathbf{a}(t)\| > T_{\text{fixed}}$ → no probability output,
  no principled threshold selection
- Adaptive threshold methods: heuristic rules, no statistical framework
- Karşılaştırma: TDA yaklaşımı calibrated probability output sağlıyor → threshold sweep principled

**Statistical feature methods:**
- Handcrafted features (time/freq domain): ad-hoc, no stability guarantees
- TDA karşılaştırması: Cohen-Steiner stability (d_B ≤ ‖f−g‖_∞) → provably noise-robust features
- Mühendislik avantajı: fixed 24-dim vector regardless of window content

**Deep learning approaches:**
- Computational cost karşılaştırması eklenecek:
  - DL training: hours to days, GPU required
  - TDA pipeline: ~75 min Bayesian search, no GPU, CPU-only → embedded deployment friendly
- Interpretability: TDA features (H0=clusters, H1=loops) interpretable; DL features opaque

**TDA for time series:**
- Bu çalışmanın farkı: giyilebilir sensör + recall-oriented + subject-personalized

---

### Results & Discussion (genişletilmiş)

**Baseline karşılaştırması henüz yapılmadı — critical gap:**
- Planlanmış baseline aileleri (proposal §5.4): threshold-based, statistical-feature, spectral-feature
- Baseline olmadan H1 (representation effectiveness) test edilemiyor
- **Not:** Baseline deneyleri koşulmadan bu bölüm tamamlanamaz

**Matematiksel + istatistiksel denge:**
- H1–H4 her biri için: gözlem (statistical) + açıklama (mathematical)
  - H3 (representation dominates): $|F2_{LR} - F2_{SVM}|$ küçük, $|F2_{tuned} - F2_{random}|$ büyük →
    matematiksel gerekçe: $\phi_\lambda$ haritası lineer olmayan ayrımı lineer rejime taşıyor
  - H4 (heterogeneous personalization): $\text{Var}(\Delta F_2(s))$ ve $\text{sign distribution}$ →
    statistical test: permutation test on sign pattern

**"Ucuzluk" katkısı Results'ta tekrar vurgulanmalı:**
- Threshold sweep runtime (~1s) vs Bayesian search runtime (~75 min) karşılaştırma tablosu/cümle

---

### Conclusion

**Mühendislik perspektifi — en kritik eksik:**
Danışman özellikle belirtti: bulgular saf matematiksel bağlamda bırakılmamalı.

Conclusion üç katmanda yazılmalı:
1. **Mathematical**: TDA'nın stability özellikleri (Cohen-Steiner), representation dominance (H3)
2. **Statistical**: Heterogeneous personalization (H4), LOSO protocol'ün subject-level generalization ölçmesi
3. **Engineering (EN KRİTİK):**
   - Zero-retraining threshold personalization: deployable on edge devices
   - No GPU required: entire pipeline runs on CPU, feasible for wearable processors
   - Calibrated probability output: enables principled threshold selection and explainability
   - Failure mode analysis: MobiFall personal protocol drop → insufficient per-subject data; system
     designer implication: minimum calibration set size required before personalization is beneficial

**Limitasyonlar (engineering framing):**
- Sensor placement assumed fixed (waist/pocket) → robustness to placement variation untested
- No ablation study confirming which of the 24 features are load-bearing
- Baseline comparison absent → absolute performance claim cannot be made

**Future work:**
- Online threshold adaptation (streaming calibration)
- Real-time embedded deployment benchmark
- Cross-dataset transfer of λ* (cross-application matrix already computed → extend to unseen datasets)

---

## Stil Kuralları (kalıcı)

- Her analiz: **matematiksel argüman + istatistiksel kanıt** çiftinde sunulmalı
- Threshold'un "ucuzluk" avantajı: her bölümde katkı olarak tekrarlanmalı
- Conclusion'da mühendislik perspektifi zorunlu
- Experiment sonuçları methodology'ye dağıtılmamalı → ayrı bölümde, en sonda
