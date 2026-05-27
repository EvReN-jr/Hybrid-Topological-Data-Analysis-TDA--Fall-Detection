# V15 DENEY SONUÇLARI: BULGULAR VE ÇIKARIMLAR
---

## İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Dataset Karşılaştırması](#dataset-karşılaştırması)
3. [Model Performans Analizi](#model-performans-analizi)
4. [Threshold Optimizasyonu Analizi](#threshold-optimizasyonu-analizi)
5. [Personal vs General Modeli](#personal-vs-general-modeli)
6. [Naive Baseline Analizi](#naive-baseline-analizi)
7. [TDA Parametre Analizi](#tda-parametre-analizi)
8. [Subject-Level Gözlemler](#subject-level-gözlemler)
9. [Özet ve Öneriler](#özet-ve-öneriler)

---

## Genel Bakış

V15, üç farklı düşme tespit veri seti üzerinde TDA (Topolojik Veri Analizi) tabanlı özellik çıkarımı ve Optuna optimizasyonu kullanılarak gerçekleştirilen kapsamlı bir deney serisidir. Üç katmanlı değerlendirme protokolü (Naive → General → Personal) ve eşik sweep analizi ile her veri setinde güçlü sonuçlar elde edilmiştir.

| Dataset | Best F2 (Optuna) | General G-F2 (LogReg) | General G-F2 (SVM) | Personal P-F2 (LogReg) | Personal P-F2 (SVM) |
|---------|-----------------|----------------------|--------------------|------------------------|---------------------|
| MobiFall | 0.9746 | 0.9049±0.0393 | 0.8918±0.0327 | 0.7325±0.0295 | 0.5981±0.0479 |
| SisFall | 0.9733 | 0.9709±0.0108 | 0.9700±0.0111 | 0.9405±0.0080 | 0.9196±0.0082 |
| FAD_40Hz | 0.9860 | 0.9429±0.0363 | 0.9592±0.0234 | 0.9168±0.0454 | 0.8724±0.0696 |

> **Referans:** Summary Comparison tablosu — [README_V15_REPORT.md → Summary Comparison](#)

---

## Dataset Karşılaştırması

### 1. En İyi Performans: FAD_40Hz

FAD_40Hz, Optuna optimizasyonunda en yüksek F2 skorunu (0.9860) elde etmiştir. Bu, büyük örnek boyutu (9481 × 24) ve düşük gürültülü sensör verisinin avantajını yansıtmaktadır. SVM modeli General katmanda G-F2 = 0.9592±0.0234 ile tüm konfigürasyonlar arasında en istikrarlı sonucu vermiştir.

| Kriter | MobiFall | SisFall | FAD_40Hz |
|--------|----------|---------|----------|
| Örnek sayısı | 1808 | 7738 | 9481 |
| Subject sayısı | 9 | 24 | 13 |
| Best Optuna F2 | 0.9746 | 0.9733 | **0.9860** |
| General LogReg G-F2 | 0.9049 | **0.9709** | 0.9429 |
| General SVM G-F2 | 0.8918 | 0.9700 | **0.9592** |
| Personal LogReg P-F2 | 0.7325 | **0.9405** | 0.9168 |
| Personal SVM P-F2 | 0.5981 | 0.9196 | 0.8724 |

**Çıkarım:** Optuna best F2'si FAD_40Hz'de en yüksek olsa da, General ve Personal katmanlar değerlendirildiğinde **SisFall en tutarlı ve dengeli performansı sergileyen dataset**'tir. Bu durum SisFall'un daha dengeli sınıf dağılımına ve daha fazla subject sayısına işaret etmektedir.

---

### 2. MobiFall: Kişiselleştirilmiş Modelde Beklenen Düşüş

MobiFall, Personal katmanda G-F2'ye kıyasla ciddi bir performans düşüşü yaşamaktadır. Özellikle SVM modelinde P-F2 = 0.5981 ile çok düşük bir değer elde edilmiştir.

| Model | G-F2 (General) | P-F2 (Personal) | Fark |
|-------|---------------|-----------------|------|
| LogReg | 0.9049±0.0393 | 0.7325±0.0295 | **-0.1724** |
| SVM | 0.8918±0.0327 | 0.5981±0.0479 | **-0.2937** |

**Çıkarım:** MobiFall'da subject başına veri miktarı düşük olduğundan (ortalama ~200 örnek/subject), daraltılmış grid ile yapılan Personal model eğitimi yetersiz kalmaktadır. Bu dataset için Personal modelin fayda sağlamadığı görülmektedir.

> **Referans:** [README_V15_REPORT.md → Dataset 1: MobiFall → Personal Model Performance Summary](#)

---

## Model Performans Analizi

### LogReg vs SVM: Genel Karşılaştırma

| Dataset | LogReg G-F2 | SVM G-F2 | Fark | Üstün Model |
|---------|------------|---------|------|-------------|
| MobiFall | 0.9049±0.0393 | 0.8918±0.0327 | +0.0131 | **LogReg** |
| SisFall | 0.9709±0.0108 | 0.9700±0.0111 | +0.0009 | ≈ Eşit |
| FAD_40Hz | 0.9429±0.0363 | 0.9592±0.0234 | -0.0163 | **SVM** |

**Çıkarım:** Hiçbir model her dataset'te tutarlı biçimde diğerini geçememiştir. Ancak SVM'in FAD_40Hz'de daha düşük varyans (±0.0234 vs ±0.0363) sergilemesi, büyük datasetlerde SVM'in daha güvenilir olduğuna işaret etmektedir.

### Precision vs Recall Dengesi

Tüm datasetlerde G-REC, G-PRE'ye kıyasla sistematik olarak daha yüksektir. Bu, F2 skorunun önceliği gereği beklenen bir durumdur.

| Dataset | Model | G-REC | G-PRE | Fark (REC-PRE) |
|---------|-------|-------|-------|----------------|
| MobiFall | LogReg | 0.9574 | ~0.73 (ortalama) | ~+0.23 |
| MobiFall | SVM | 0.9543 | ~0.72 | ~+0.23 |
| SisFall | LogReg | 0.9836 | ~0.93 | ~+0.05 |
| SisFall | SVM | 0.9742 | ~0.95 | ~-0.02 |
| FAD_40Hz | LogReg | 0.9929 | ~0.83 | ~+0.16 |
| FAD_40Hz | SVM | 0.9878 | ~0.89 | ~+0.10 |

**Çıkarım:** SisFall ve FAD_40Hz'de precision-recall dengesi çok daha iyidir. MobiFall'da precision hâlâ oldukça düşük kalmakta; bu durum yüksek false positive oranının devam ettiğini göstermektedir.

> **Referans:** [README_V15_REPORT.md → General Model Performance Summary](#)

---

## Threshold Optimizasyonu Analizi

V15'in en önemli yeniliklerinden biri, her subject için ayrı eşik sweep uygulamasıdır. Aşağıdaki tablo ortalama F2 kazanımlarını özetlemektedir:

| Dataset | Model | Mean ΔF2 | Max ΔF2 | Max ΔF2 Subject |
|---------|-------|----------|---------|-----------------|
| MobiFall | LogReg | +0.0348 | +0.0615 | Sub 8 |
| MobiFall | SVM | **+0.0474** | +0.1013 | Sub 9 |
| SisFall | LogReg | +0.0079 | +0.0527 | Sub 29 |
| SisFall | SVM | +0.0042 | +0.0169 | Sub 29 |
| FAD_40Hz | LogReg | +0.0250 | **+0.1165** | Sub 11 |
| FAD_40Hz | SVM | +0.0121 | +0.0691 | Sub 11 |

> **Referans:** [README_V15_REPORT.md → Threshold Sweep Summary tabloları](#)

**Çıkarım 1 — MobiFall'da yüksek kazanım:** MobiFall'da eşik optimizasyonu, diğer datasetlere kıyasla anlamlı ölçüde daha fazla F2 artışı sağlamaktadır (SVM için ortalama +0.0474). Bu, MobiFall'un varsayılan eşik değerlerinin (0.37–0.77 arası geniş dağılım) suboptimal olduğunu göstermektedir.

**Çıkarım 2 — SisFall'da düşük kazanım:** SisFall'da eşik optimizasyonu çok küçük kazanımlar (+0.0042 - +0.0079) sağlamaktadır. Bu, varsayılan eşiklerin SisFall için zaten yakın-optimal olduğuna işaret etmektedir; model kararları bu dataset üzerinde eşik değerine daha az duyarlıdır.

**Çıkarım 3 — FAD_40Hz Sub 11 anomalisi:** FAD_40Hz'de Sub 11 için LogReg modelinde ΔF2 = +0.1165, son derece yüksek bir kazanım ifade etmektedir. Bu subject'in varsayılan eşikle çok kötü performans göstermesi (DEF_F2=0.8046), sınıf özelliklerinin bu subject için belirgin biçimde farklı olduğunu düşündürmektedir.

### Optimal Eşik Yönleri

| Dataset | Model | Ağırlıklı Eğilim | Yorum |
|---------|-------|-----------------|-------|
| MobiFall | LogReg | Yukarı (0.80+) | Daha seçici karar sınırı gerekli |
| MobiFall | SVM | Yukarı (0.60-0.65) | Orta yükseklikte eşik |
| SisFall | LogReg | Küçük değişimler | Varsayılan yakın-optimal |
| SisFall | SVM | Küçük değişimler | Varsayılan yakın-optimal |
| FAD_40Hz | LogReg | Yukarı (0.65-0.75) | Daha yüksek eşik genellikle daha iyi |
| FAD_40Hz | SVM | Orta değişimler | Subject bazlı özelleştirme etkili |

---

## Personal vs General Modeli

V15'in üç katmanlı değerlendirmesi, General ve Personal modellerin farklı veri senaryolarında ne ölçüde fayda sağladığını ortaya koymaktadır.

### P-F2 / G-F2 Oranı (Kişiselleştirme Kazanımı)

| Dataset | Model | G-F2 | P-F2 | Oran (P/G) | Değerlendirme |
|---------|-------|------|------|-----------|---------------|
| MobiFall | LogReg | 0.9049 | 0.7325 | 0.81 | ❌ Kişiselleştirme zarar veriyor |
| MobiFall | SVM | 0.8918 | 0.5981 | 0.67 | ❌ Kişiselleştirme ciddi zarar veriyor |
| SisFall | LogReg | 0.9709 | 0.9405 | 0.97 | ✅ Kabul edilebilir, küçük kayıp |
| SisFall | SVM | 0.9700 | 0.9196 | 0.95 | ✅ Kabul edilebilir |
| FAD_40Hz | LogReg | 0.9429 | 0.9168 | 0.97 | ✅ Kabul edilebilir, küçük kayıp |
| FAD_40Hz | SVM | 0.9592 | 0.8724 | 0.91 | ⚠️ Dikkat, kayıp var |

> **Referans:** [README_V15_REPORT.md → Personal & General Model Performance Summary](#)

**Çıkarım:** Kişiselleştirilmiş model, ancak her subject için yeterli eğitim verisi olduğunda (SisFall ve kısmen FAD_40Hz) anlamlıdır. MobiFall gibi az örnekli datasetlerde Personal model, General modelin çok gerisinde kalmaktadır. Bu bulgu, klinik uygulamalar için minimum kişisel veri eşiği belirlenmesi gerekliliğini işaret etmektedir.

---

## Naive Baseline Analizi

Naive model, daraltılmış grid arama olmadan global parametrelerle tekrarlanan eğitimin performansını temsil eder.

### Naive F2 vs General G-F2 Karşılaştırması

| Dataset | Model | Naive F2 | General G-F2 | Fark |
|---------|-------|----------|-------------|------|
| MobiFall | LogReg | 0.8857 | 0.9049 | +0.0192 |
| MobiFall | SVM | 0.8573 | 0.8918 | +0.0345 |
| SisFall | LogReg | 0.9713 | 0.9709 | -0.0004 |
| SisFall | SVM | 0.9715 | 0.9700 | -0.0015 |
| FAD_40Hz | LogReg | 0.9502 | 0.9429 | -0.0073 |
| FAD_40Hz | SVM | 0.9603 | 0.9592 | -0.0011 |

**Çıkarım 1:** MobiFall için LOGO cross-validation (General), Naive'e göre belirgin biçimde daha iyi sonuç vermektedir. Bu, MobiFall'daki subject-level varyasyonun yüksekliğini gösterir; her subject'i dışarıda bırakarak değerlendirme, daha gerçekçi bir performans tahmini sunmaktadır.

**Çıkarım 2:** SisFall ve FAD_40Hz'de Naive ve General performans hemen hemen eşittir. Bu iki dataset için Naive baseline, üretim ortamı için yeterince güvenilir bir tahmin sunmaktadır.

---

## TDA Parametre Analizi

Her dataset için Optuna tarafından seçilen optimal parametreler karşılaştırıldığında:

| Parameter | MobiFall | SisFall | FAD_40Hz |
|-----------|----------|---------|----------|
| `win_sec` | **5.0** | 1.0 | 1.0 |
| `complex_type` | SparseRips | Alpha | Rips |
| `dim` | **5** | **5** | 3 |
| `delay` | 2 | 4 | 2 |
| `stride_factor` | 2 | **4** | 1 |
| `metric` | chebyshev | — | manhattan |
| `eps_percentile` | **80** | — | 40 |

> **Referans:** [README_V15_REPORT.md → Optimal TDA Parameters tabloları](#)

**Çıkarım 1 — win_sec:** MobiFall'da en uzun pencere (5.0 sn) seçilmiştir. Bu, MobiFall'daki düşme örüntülerinin daha uzun zaman bağlamı gerektirdiğini göstermektedir. SisFall ve FAD_40Hz, daha hızlı düşme algılama için 1.0 sn'lik kısa pencereyi tercih etmektedir.

**Çıkarım 2 — complex_type çeşitliliği:** Her dataset farklı bir simplicial complex tipi seçmiştir (SparseRips / Alpha / Rips). Bu, tek bir komplex tipinin tüm düşme senaryolarında evrensel biçimde üstün olmadığını ortaya koymaktadır.

**Çıkarım 3 — stride_factor:** SisFall'da stride_factor=4, yani özellik çıkarımı seyrekleştirilmiştir. Bu, SisFall'un 7738 örnekle en büyük dataset olduğunu ve hesaplama verimliliği için seyrek örneklemenin tercih edildiğini yansıtmaktadır.

**Çıkarım 4 — dim:** MobiFall ve SisFall için dim=5 (maksimum), FAD_40Hz için dim=3 seçilmiştir. Yüksek boyutlu topolojik özellikler daha küçük ve gürültülü datasetlerde (MobiFall) daha iyi ayrım yaparken, büyük ve kaliteli datasetlerde (FAD_40Hz) daha basit topoloji yeterlidir.

---

## Subject-Level Gözlemler

### En Zayıf Performans Gösteren Subjectler

Aşağıdaki tablo, G-F2 değerine göre en düşük performans sergiyleyen subjectleri listelemektedir:

| Dataset | Model | SUB | G-F2 | P-F2 | Olası Neden |
|---------|-------|-----|------|------|-------------|
| MobiFall | LogReg | 8 | 0.8164 | 0.6765 | Düşük precision (0.5698), yüksek FP |
| MobiFall | SVM | 8 | 0.8122 | 0.4749 | Benzer, SVM daha kötü |
| SisFall | LogReg | 29 | 0.8861 | 0.9211 | Düşük G-REC (0.8702) |
| SisFall | SVM | 29 | 0.8989 | 0.9129 | Düşük G-REC (0.8827) |
| FAD_40Hz | LogReg | 11 | 0.7516 | 0.7001 | Çok düşük precision (0.3788) |
| FAD_40Hz | SVM | 11 | 0.8417 | 0.5399 | Precision: 0.5195 |

**Çıkarım:** Sub 11 (FAD_40Hz) ve Sub 8 (MobiFall) tüm modellerde tutarsız biçimde düşük performans sergilemektedir. Bu subjectlerin ya atipik hareket örüntüleri ya da kayıt kalitesi sorunları barındırdığı değerlendirilmelidir. Veri temizleme veya bu subjectler için özel model yaklaşımı araştırılabilir.

### Threshold Sweep ile En Büyük Kazanım Sağlanan Subjectler

| Dataset | Model | SUB | DEF_F2 | OPT_F2 | ΔF2 |
|---------|-------|-----|--------|--------|-----|
| FAD_40Hz | LogReg | 11 | 0.8046 | 0.9211 | **+0.1165** |
| MobiFall | SVM | 9 | 0.8824 | 0.9836 | +0.1013 |
| MobiFall | SVM | 5 | 0.8955 | 0.9836 | +0.0881 |
| FAD_40Hz | SVM | 11 | 0.8642 | 0.9333 | +0.0691 |
| FAD_40Hz | LogReg | 12 | 0.9216 | 0.9958 | +0.0742 |

**Çıkarım:** Yüksek ΔF2 değerleri, o subject için varsayılan eşiğin (genellikle ~0.50-0.55) suboptimal olduğunu gösterir. Bu subjectlerin karar sınırları diğerlerinden belirgin biçimde farklıdır ve subject-specific threshold kalibrasyonu production ortamında önemli kazanımlar sağlayabilir.

---

## Özet ve Öneriler

### Temel Bulgular

| Bulgu | Detay | İlgili Tablo |
|-------|-------|--------------|
| **En iyi dataset (Optuna)** | FAD_40Hz: F2 = 0.9860 | Dataset Overview |
| **En tutarlı dataset (General+Personal)** | SisFall: tutarlı yüksek performans her katmanda | General & Personal Summary |
| **MobiFall'da Personal model sorunu** | SVM P-F2 = 0.5981 (G-F2'nin %67'si) | Personal Summary |
| **Eşik optimizasyonu en etkili** | MobiFall SVM: ortalama +0.0474 | THR-Sweep Summary |
| **Eşik optimizasyonu en az etkili** | SisFall SVM: ortalama +0.0042 | THR-Sweep Summary |
| **En sorunlu subject** | FAD_40Hz Sub 11: G-F2 = 0.7516 (LogReg) | Per-Subject Results |
| **Optimal pencere boyutu** | 1.0 sn (SisFall & FAD_40Hz), 5.0 sn (MobiFall) | Optimal TDA Parameters |
| **Tercih edilen complex** | Dataset'e bağlı: SparseRips / Alpha / Rips | Optimal TDA Parameters |

### Öneriler

1. **MobiFall için Personal Model Revizyonu:** Az örnekli subjectler için Personal model zararlı olabilir. Minimum örnek sayısı eşiği belirlenmeli veya veri artırma (augmentation) teknikleri denenmelidir.

2. **Subject-Specific Threshold Kalibrasyonu:** Özellikle MobiFall ve FAD_40Hz'de, her subject için ayrı optimal eşik kullanmak ortalama +0.035–0.047 F2 kazanımı sağlamaktadır. Bu, deployment senaryolarında değerlendirilmelidir.

3. **Sub 11 (FAD_40Hz) ve Sub 8 (MobiFall) için Veri İncelemesi:** Bu subjectlerin düşük ve tutarsız performansı, veri kalitesi veya hareket örüntüsü anomalisine işaret etmektedir. Ham verinin gözden geçirilmesi önerilir.

4. **SisFall Parametrelerinin Transfer Edilebilirliği:** SisFall, tüm katmanlarda en dengeli performansı sergilemiştir. Bu dataset üzerinde bulunan optimal parametreler (Alpha, win_sec=1.0, dim=5) gelecek versiyonlar için başlangıç noktası olarak değerlendirilebilir.

5. **FAD_40Hz'de SVM Tercihi:** FAD_40Hz'de SVM, LogReg'e kıyasla hem daha yüksek G-F2 (0.9592 vs 0.9429) hem de daha düşük varyans sunmaktadır. Bu dataset için SVM production modeli olarak önerilmektedir.

---

*Bu belge, V15 deney raporundaki ([README_V15_REPORT.md]) tablo ve sonuçlara atıfta bulunmaktadır.*
