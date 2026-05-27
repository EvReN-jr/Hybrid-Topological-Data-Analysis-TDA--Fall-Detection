# Tez Bağlam Dokümanı — Hızlı Başlangıç Kılavuzu
**Son güncelleme:** 2026-05-16  
**Çalışma dizini:** `/home/evo/YL_TEZ/Move/Graduate_Sections/`

---

## Tez Özeti

**Başlık:** Topological Data Analysis for Recall-Oriented Wearable Fall Detection: Subject-General and Subject-Personalized Evaluation  
**Türkçe:** Topolojik Veri Analizi Yöntemleriyle Giyilebilir Sensörlerden İnsan Düşmesi Tespiti  
**Tür:** Yüksek Lisans Bitirme Tezi  

Thesis'in merkezi iddiası: Gecikme gömme (delay embedding) + kalıcı homoloji (persistent homology) tabanlı 24-boyutlu öznitelik vektörleri, Bayesian hiperparametre aramasıyla optimize edildiğinde, lojistik regresyon ve SVM gibi basit sınıflandırıcılarla bile güçlü geri çağırma-odaklı (recall-oriented) düşme tespiti sağlayabilir. Bu başarının büyük bölümü temsil haritasından ($\phi_\lambda$) gelir, sınıflandırıcı seçiminden değil. Kişiselleştirme (threshold calibration) homojen bir iyileştirme sağlamaz; bazı deneklerde dramatik kazanım verirken diğerlerinde zarar bile verebilir.

---

## Veri Setleri

### MobiFall v2.0
- **Kaynak:** Kaggle — Samsung Galaxy S3 akıllı telefon, pantolon cebinde
- **Sensörler (orijinal):** Acc (3-eksen), Gyr (3-eksen), Mag (3-eksen), Bar (basınç), Azimuth/Pitch/Roll
- **Kullanılan sensörler:** Acc + Gyr (diğerleri: dimensionality reduction + diğer veri setleriyle tutarsızlık nedeniyle çıkarıldı)
- **Örnekleme hızı:** 87.5 Hz → 50 Hz
- **Fall türleri:** 4 (ileri/diz üstü/geri-sandalye/yana)
- **ADL:** 9 (yürüme, koşu, merdiven vb.)
- **Kullanılan denekler:** 9 (ID: 2,3,4,5,7,8,9,10,11) — diğerleri yalnızca fall kaydı var, ADL yok → LOSO değerlendirme için uygun değil
- **Özellik matrisi:** 1808 × 24

### SisFall
- **Sensörler (orijinal):** ADXL345 acc (±16g), MMA8451Q acc (±8g), ITG3200 gyr (±2000°/s) — waist mounted
- **Kullanılan sensörler:** ADXL345 + ITG3200 (MMA8451Q: düşük dinamik aralık, saturasyon riski; tek acc tutarlılığı için çıkarıldı)
- **Örnekleme hızı:** 200 Hz → 50 Hz (integer decimation, her 4. örnek)
- **Fall türleri:** 15, **ADL:** 19
- **Denekler:** 23 genç yetişkin + 15 yaşlı, toplam 38; 24 korundu (23 genç + 1 yaşlı)
- **Özellik matrisi:** 7738 × 24

### FallAllD — 40Hz Derived Variant (FAD_40Hz)
- **Orijinal FallAllD:**
  - 15 denek, 3 sensör pozisyonu (waist/wrist/neck)
  - 4 modalite: Acc, Gyr, Mag, Bar
  - Hızlar: 238 Hz (Acc+Gyr), 80 Hz (Mag), 10 Hz (Bar)
  - 135 aktivite türü (44 ADL + 91 fall varyantı)
  - 26.420 dosya
- **Neden 40Hz versiyonu:**
  1. 238 Hz → Bayesian arama (200 iter × dataset) için hesaplama maliyeti çok yüksek
  2. Mag+Bar çıkarıldı → multi-rate sensor fusion problemi ortadan kalktı
  3. Neck pozisyonu çıkarıldı
  4. Aktivite etiketleri filtrelendi (yalnızca ilgili fall + ADL)
- **Kullanılan sensörler:** Acc + Gyr (waist pozisyonu)
- **Örnekleme hızı:** 40 Hz → 50 Hz (upsampling, temporal interpolation)
- **Kullanılan denekler:** 13 (15'ten 2'si yetersiz trial nedeniyle çıkarıldı)
- **Özellik matrisi:** 9481 × 24

---

## Pipeline (Önişleme Adımları)

1. **SMV (Signal Magnitude Vector):** Her sensör için 3 eksen → $\sqrt{x^2+y^2+z^2}$ — yönelim bağımsız
2. **Resampling to 50 Hz:**
   - SisFall (200→50 Hz): Integer decimation (her 4. örnek)
   - FAD_40Hz (40→50 Hz): Upsampling → temporal interpolation (hedef grid üzerinde)
   - MobiFall (87.5→50 Hz): Non-integer ratio (7:4), rational sample-rate conversion
   - **50 Hz neden seçildi → Methodology bölümünde açıklanacak**
3. **Window extraction:** Fall trial → en yüksek SMV noktası etrafında tek pencere; ADL → ardışık örtüşmesiz pencereler
4. **Standard scaling:** Zero mean, unit variance — parametreler sadece training veriden tahmin edilir

---

## Öznitelik Seti (24 boyut)

- **H0 istatistikleri (1-10):** Bar count, Entropy, Max, Energy, Mean, Std, Median, Q1, Q3, 3rd moment
- **H1 istatistikleri (11-20):** Aynıları H1 için
- **Sinyal özellikleri (21-24):** Acc SMV(t) → Max, Std, Mean, Range

Teorik türetiş (delay embedding, persistence diagram, filtrasyon yapısı) → **Methodology bölümünde**

---

## Deneysel Sonuçlar

### Bayesian Hiperparametre Araması (200 iter/dataset)

| Dataset | Win (s) | Complex | Dim | Delay | Stride | Metric | Eps | F2 |
|---------|---------|---------|-----|-------|--------|--------|-----|-----|
| MobiFall | 5.0 | SparseRips | 5 | 2 | 2 | Chebyshev | 80% | 0.9746 |
| SisFall | 1.0 | Alpha | 5 | 4 | 4 | — | — | 0.9733 |
| FAD_40Hz | 1.0 | Rips | 3 | 2 | 1 | Manhattan | 40% | 0.9860 |

**Not:** Bu tablodaki değerler Bayesian aramanın cross-validation nesneleridir; aşağıdaki General protocol sonuçlarından farklıdır.

### Genel (General/LOSO) F2 Sonuçları

| Dataset | LR F2 | SVM F2 |
|---------|--------|--------|
| MobiFall | 0.9049 ± 0.0393 | 0.8918 ± 0.0327 |
| SisFall | 0.9799 ± 0.0058 | 0.9700 ± 0.0111 |
| FAD_40Hz | 0.9429 ± 0.0363 | 0.9592 ± 0.0234 |

### Threshold Optimizasyonu ΔF2 (General protocol üzerinde)

| Dataset | ΔF2 (LR) | ΔF2 (SVM) |
|---------|----------|-----------|
| MobiFall | +0.0348 | +0.0474 |
| SisFall | +0.0079 | +0.0042 |
| FAD_40Hz | +0.0250 | +0.0121 |

FAD_40Hz Subject 11: ΔF2 = +0.1165 (LR) — 0.8235 → 0.9400

---

## Değerlendirme Protokolleri (Methodology'de tanımlanacak)

- **Naive:** Random 70/30 split, 10 tekrar — üst sınır referansı
- **General (LOSO):** Leave-one-subject-out, 15 tekrar (random seed) — birincil protokol
- **Personal:** Her denek için ayrı model (kendi 70/30 verisi), 15 tekrar
- **Threshold sweep:** General model üzerinde τ ∈ [0.05, 0.95] (19 değer)

## Metrikler (Methodology'de tanımlanacak)

- Acc, Rec, Pre, F1, F2 (birincil metrik)
- $F_2 = 5 \frac{\text{Prec} \cdot \text{Rec}}{4\,\text{Prec} + \text{Rec}}$ — recall ağırlıklı
- **Neden F2 → Methodology'de**
- 95% CI: $\bar{x} \pm t_{0.025,14} \cdot s/\sqrt{15}$

---

## Sınıflandırıcılar

| Model | Config | Grid Search |
|-------|--------|------------|
| Logistic Regression | class_weight='balanced', solver='liblinear' | C ∈ {0.01,0.1,1,10,100}, penalty ∈ {l1,l2} |
| SVM | class_weight='balanced', probability=True (Platt) | C ∈ {0.1,1,10,100}, kernel ∈ {rbf,linear} |

---

## Dosya Yapısı

```
/home/evo/YL_TEZ/Move/Graduate_Sections/
├── experiment_section.tex     ← Ana experiment bölümü (GÜNCEL)
├── experiment_section.pdf     ← Derlenmiş PDF
└── TEZ_CONTEXT.md             ← Bu dosya

/home/evo/YL_TEZ/Move/Graduate_TDA_Fall_Detection/
├── kenan_boyabatli_proposal.tex  ← Proposal (üslup referansı)
├── TEZ_V1.tex                    ← Türkçe genel tez taslağı
├── TEZ_DataSet_TR.tex            ← Türkçe dataset bölümü (eski)
└── references.bib

/home/evo/YL_TEZ/Data/
├── archive_pckl/                 ← FallAllD ham + türetilmiş veri
├── fall_data/                    ← Diğer fall verileri
└── Data_Sources.txt

/home/evo/YL_TEZ/TDA_Features_Extraction_V3/all_pipeline_V3.py  ← TDA pipeline (V3)
/home/evo/YL_TEZ/DataFirstTouch/                                 ← Veri hazırlama scriptleri
```

---

## Tamamlanan Bölümler

1. **Proposal** (`kenan_boyabatli_proposal.tex`) — Tam ✓
   - Formal problem setting, hypotheses, methodology overview, expected contributions

2. **Experiments** (`experiment_section.tex`) — Güncellendi ✓ (2026-05-16)
   - Datasets, preprocessing, features, Bayesian opt, classifiers, results
   - Evaluation protocols + metrics + computational infrastructure → Methodology'ye taşındı

---

## Yazılacak Bölümler (Sıralı Öneri)

### 1. Methodology (Sonraki Bölüm — En Kritik)

İçerik:
- **Formal problem setting** (proposal'dan genişletilmiş)
- **50 Hz hedef frekans seçimi gerekçesi** (Nyquist, fall event spectral content ~10-20 Hz)
- **F2 metrik seçimi gerekçesi** (missed fall cost >> false alarm cost)
- **Evaluation protocols:**
  - Naive (70/30 random, 10 rep)
  - General / LOSO (15 rep per subject fold, 95% CI via t-distribution)
  - Personal (70/30 per subject, 15 rep, smaller C grid {0.1,1,10})
  - Threshold sweep (τ ∈ [0.05,0.95])
- **Delay embedding:** Takens' theorem, $(m, \tau)$ parametreleri
- **Persistent homology:** Vietoris-Rips / Alpha / SparseRips complex, H0/H1
- **Filtration construction ve barcode**
- **Vectorization:** 10 istatistik (barcode → scalar)
- **Computational infrastructure:** SLURM cluster, UHeM (allocation 4025462026), dual Intel Xeon Gold 6148, 190GB RAM, ~3h45m total, peak ~10GB memory

### 2. Introduction
- Motivasyon: Yaşlı nüfus, düşme riski, WHO istatistikleri
- TDA'nın düşme tespitindeki potansiyeli
- Mevcut yöntemlerin kısıtları (cross-dataset generalizability, personalization)
- Thesis contribution'ları (proposal Section 1'den adapte)

### 3. Related Work
- Threshold-based detectors
- Statistical feature methods
- Spectral feature methods
- Deep learning approaches
- TDA for time series (proposal references'larından)

### 4. Results & Discussion (genişletilmiş)
- Baseline karşılaştırması (henüz yapılmadı — TODOs'ta işaretli)
- Ablation studies (planned in proposal)
- Hypothesis test summary

### 5. Conclusion
- Katkılar özeti
- Limitasyonlar
- Gelecek çalışmalar

---

## Önemli Yazım Notları

- **Üslup referansı:** `kenan_boyabatli_proposal.tex` — kesin matematik notasyonu, pasif ses, doğrudan ifade, fazla kelime yok
- **Dil:** İngilizce (tez)
- **Teori açıklamaları:** Experiment bölümünde yok → "See Methodology Section" şeklinde yönlendir
- **Metrik/protokol tanımları:** Methodology bölümünde → Experiment bölümünde sadece referans ver
- **Yorum değil bulgular:** "This suggests X" formatında — assertion değil observation

---

## Hipotezler (Proposal'dan)

1. **H1 (Representation effectiveness):** SisFall + FAD'de yüksek, MobiFall'da daha düşük ama non-trivial F2
2. **H2 (Dataset-specific geometry):** Optimal λ* dataset'e göre farklılık gösterir (doğrulandı — farklı complex type, window)
3. **H3 (Representation dominates classifier):** LR vs SVM farkı küçük, tuned vs untuned λ farkı büyük
4. **H4 (Heterogeneous personalization):** Δs = F2_pers - F2_gen dağılımı heterojen (bazı denekler +, bazıları -)

---

## Teknik Detaylar (Önemli Notlar)

- **Öznitelik sayısı 24'te sabit** (hem H0 hem H1 için 10'ar istatistik + 4 sinyal özelliği)
- **Balanced class weights** tüm modellerde kullanıldı
- **Platt scaling** SVM probability outputs için
- **No-info-leakage:** Tüm scaling/threshold parameters → sadece training fold'dan
- **15 tekrar/denek** General ve Personal için → 95% CI via $t_{0.025,14}$
- **Cluster:** UHeM SLURM, allocation 4025462026
