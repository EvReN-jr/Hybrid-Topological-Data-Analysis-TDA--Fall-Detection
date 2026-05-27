# Tez Durum ve Devam Notları

**Son güncelleme:** 2026-05-27
**Çalışma dizini:** `/home/evo/YL_TEZ/Thesis_Final/`
**Dil:** Tüm içerik **İngilizce** (ÖZET dahil; tek istisna İTÜ'nün zorunlu kıldığı genişletilmiş Türkçe özet).

---

## 1. Tez Özeti

- **Başlık (EN):** Topological Data Analysis of Wearable Sensor Signals: Development and Evaluation of Recall-Oriented Subject-General and Subject-Personalized Models
- **Başlık (TR):** Giyilebilir Sensör Sinyalleriyle Topolojik Veri Analizi: Düşme Vakalarını Kaçırmamayı Önceleyen Genel ve Kişisel Modellerin Geliştirilmesi ve Değerlendirilmesi
- **Tür:** İTÜ Yüksek Lisans Tezi (Matematik Mühendisliği, Lisansüstü Eğitim Enstitüsü)
- **Yazar:** Kenan Evren BOYABATLI (No: 509231205)
- **Danışman:** Prof. Dr. Atabey KAYGUN
- **Şablon:** `itutez.cls` (APA, biblatex/biber)

**Merkezi iddia:** Gecikme gömme + kalıcı homoloji ile çıkarılan 24-boyutlu öznitelikler, Bayesçi optimizasyonla ayarlandığında basit sınıflandırıcılarla (LR/SVM) bile güçlü recall-odaklı düşme tespiti sağlar. Başarı sınıflandırıcıdan değil temsil haritasından (φ_λ) gelir.

---

## 2. Derleme

```bash
cd /home/evo/YL_TEZ/Thesis_Final
pdflatex -interaction=nonstopmode tez.tex
biber tez
pdflatex -interaction=nonstopmode tez.tex
pdflatex -interaction=nonstopmode tez.tex
```

**Mevcut çıktı:** 145 sayfa, hata yok, çözümlenmemiş referans/atıf yok, çift label yok.
Bölüm yapısı: 1.Introduction · 2.Related Work · 3.Methodology · 4.Experiments · 5.Results and Discussion · 6.Conclusion.

`Thesis_Final/` tamamen bağımsızdır (figürler `figures/` içinde, `graphicspath={./}`).

---

## 3. Bu Oturumda Tamamlananlar

### Birleştirme & temizlik
- 6 kaynak bölüm `Move/Graduate_Sections/`'tan `ch1–ch6.tex`'e dönüştürüldü ve birleştirildi.
- Figürler `Thesis_Final/figures/` altına kopyalandı (dış bağımlılık yok).
- ch3/ch4'te kalan eski `\begin{thebibliography}` blokları silindi (çift-label uyarısı gideriyordu).
- Kaynakçadaki 4 yinelenen kayıt tek anahtara indirgendi (sisfall/fallalld/optuna/cohen2007stability tutuldu). Kaynakçada artık 38 benzersiz giriş.

### Ön sayfalar
- **Önsöz (onsoz.tex):** UHeM teşekkürü eklendi (grant 4025462026) + danışman teşekkürü + imza.
- **Kısaltmalar (kisaltmalar.tex):** Şablon listesi silinip tezdeki gerçek kısaltmalarla dolduruldu (ADL, LOSO, SMV, TDA, TPE, PH, SVM, LR…).
- **SUMMARY (summary.tex):** Gerçek İngilizce kısa özet (~1 sayfa).
- **ÖZET (ozet.tex):** Genişletilmiş Türkçe özet (~3 sayfa) — İTÜ'nün İngilizce tez için zorunlu kıldığı tek Türkçe içerik.
- **Özgeçmiş (ozgecmis.tex):** Gerçek İngilizce CV (özgeçmiş PDF'inden: İst. Medeniyet B.Sc. Matematik 2023, Mapin Data, TÜBİTAK 119F279).
- **Ekler:** Boş Lorem ipsum temizlendi; ek bloğu tez.tex'te yorumlandı (gerçek ek yok).

### Yapısal bug düzeltmeleri
- **ch3 chapter şişmesi:** Methodology'de 13 `\chapter` vardı (birleştirme artığı) → numaralandırmayı bozuyordu (Results "17", Conclusion "A."). Başlıklar bir kademe indirildi: `\chapter{Methodology}` kaldı, 12 chapter→section, 46 section→subsection. Artık Methodology = §3.1–3.12.
- **Başıboş `\appendix`:** ch5 sonundaki `\appendix`+`\chapter*{Note on Figures}` TODO bloğu Conclusion'ı "A." yapıyordu → kaldırıldı.
- **ToC derinliği:** `\setcounter{secnumdepth}{3}` + `\setcounter{tocdepth}{3}` eklendi → paragraflar artık numarasız, "x.y.z.0.w" çirkinliği gitti.
- **Overfull tablolar:** ch3 (multline, align yeniden düzen) ve ch4 (footnotesize + dar sütun) düzeltildi.

### Ablation Studies (YENİ — Bölüm 5.5)
- `ch5.tex`'e "Ablation Study: Feature-Block Contributions" bölümü + tablo eklendi.
- Tez protokolüne **birebir** (General/LOSO, global grid-search, 15 tekrar, eşik, denek-üstü ortalama ± %95 CI) replikasyon: `tda_proje/ablation_run.py`.
- **Doğrulama:** Full (24-boyut) bloğu LR ve SVM'de üç veri kümesinde de tezin manşet F2'lerini yeniden üretti.
- **Bulgular:** (1) H0 ≫ H1, H1 neredeyse gereksiz; (2) topoloji vs sinyal katkısı veri kümesine bağlı (MobiFall'da topoloji baskın, SisFall/FAD'de sinyal tek başına güçlü); (3) Full her zaman en iyi; (4) sınıflandırıcıdan bağımsız (LR≈SVM sıralaması).

---

## 4.0 ✅ ÇÖZÜLEN SORUNLAR (2026-05-27)

### Sorun A — Sayfadan sayfaya hizalama kayması → ÇÖZÜLDÜ
**Belirtiydi:** Metin bloğu sayfadan sayfaya kayıyordu (bir sayfa sağa, diğeri sola).
**Kök neden:** itutez.cls `onluarkali`/twoside ciltleme payı → tek sayfa sol kenar 40mm, çift sayfa 26.4mm (13.6mm fark).
**Çözüm:** tez.tex'e `\AtBeginDocument{\setlength{\evensidemargin}{\oddsidemargin}}` eklendi → her sayfa 40mm sol kenar (ITU ciltleme payı korunarak, tek tip).
**Doğrulama:** Tek (s.32) ve çift (s.33) sayfanın sol metin kenarı artık piksel-piksel aynı.

### Sorun B — Atıflar parantezsiz → ÇÖZÜLDÜ
**Belirtiydi:** Atıflar düz metin olarak cümleye yapışıyordu ("…injuries World Health Organization, 2021.").
**Kök neden:** Tüm atıflar `\cite{}` ile yazılmıştı (60 adet); biblatex `apa`'da `\cite` parantez koymaz.
**Çözüm:** ch1–ch6'da `\cite{` → `\parencite{` (60 adet) değiştirildi.
**Doğrulama:** Artık "(World Health Organization, 2021)", "(Wild, Nayak, & Isaacs, 1981)" şeklinde parantezli.
**İnce ayar (opsiyonel, sonraki oturum):** Yazarın cümle öznesi olduğu birkaç yerde `\textcite{}` daha doğal okunabilir (ör. ch1 ~s.15-17 WHO narrative).

Derleme sonrası: 145 sayfa, hata/çözümlenmemiş referans yok.

---

## 4. AÇIK İŞLER (devam edilecek)

| Konu | Durum / Not |
|------|-------------|
| **Jüri üyeleri** | tez.tex'te hâlâ "Placeholder One/Two/Three" — savunma sonrası gerçek isimler girilecek |
| **Baseline karşılaştırması** | Threshold/istatistiksel/spektral yöntemlerle kıyas henüz yok (danışman istemişti). Konu netleşmedi. ch5'te "Limitation: absence of baseline comparison" olarak dürüstçe belirtiliyor |
| **Delay-embedding ablation** | Proposal'daki "gecikme gömmeyi tamamen kaldır" ablation'ı yapılmadı — gerektirdiği yeniden öznitelik çıkarımı UHeM cluster gerektiriyor. ch5 Ablation §5.5'te "scope" olarak belirtildi |
| **Genişletilmiş Türkçe özet** | ~3 sayfa yazıldı; İTÜ kuralı 3-5 sayfa istiyor. İstenirse genişletilebilir veya bölüm İngilizce-only kabul ediyorsa kaldırılabilir |
| **Önsöz** | Kısa; istenirse genişletilebilir |

---

## 5. Önemli Notlar / Uyarılar

- **Dil tercihi:** Tüm içerik İngilizce olmalı. Türkçe yazma (tek istisna: zorunlu genişletilmiş Türkçe ÖZET).
- **Kapak overfull uyarıları:** Log'da `line 118` (= `\begin{document}`) kaynaklı 473pt/422pt overfull uyarıları var — bunlar itutez kapak sayfalarının iç kutuları, **görsel etkisi yok**, normal.
- **Ablation tol=1e-3:** LR replikasyonunda liblinear `tol=1e-3` kullanıldı (varsayılan 1e-4 yerine). Tez F2'lerini CI içinde aynen ürettiği doğrulandı; sadece l1 hesabını hızlandırır (74s → 4s). Sonuçları değiştirmez.
- Bölümlerde subsubsection yok; en derin numaralı seviye subsection. Paragraf'lar bilinçli olarak numarasız.

---

## 6. Veri ve Sonuç Konumları (tezin dışında)

- **Sonuç veritabanı:** `/home/evo/YL_TEZ/tda_proje/Results_V15.db`
  - `features_<ds>_server`: optimal λ'da 24-boyut öznitelik matrisi + label + subject
  - `optuna_trials_<ds>_server`: 200 deneme (F2 + λ konfigürasyonu)
  - `metrics_<ds>_server`: denek-bazlı general/personal + threshold sweep
  - Öznitelik eşlemesi: feat_0–9 = H0 ist., feat_10–19 = H1 ist., feat_20–23 = sinyal (max, std, mean, range)
- **Ablation script & çıktıları:** `tda_proje/ablation_run.py`, `ablation_LogReg_final.json`, `ablation_SVM.json`
- **Pipeline:** `tda_proje/uhem_big_optuna_v13.py` (TDA çıkarım + protokol; cluster sürümü)
- **Bağlam notları:** `Move/Graduate_Sections/notes/` (TEZ_CONTEXT.md, ADVISOR_FEEDBACK.md, SESSION_LOG)

### Ablation'ı yeniden koşturma
```bash
cd /home/evo/YL_TEZ
python3 tda_proje/ablation_run.py LogReg   # ~birkaç dk
python3 tda_proje/ablation_run.py SVM      # ~30-45 dk (rbf+probability yavaş)
```

---

## 7. Veri Kümeleri (hızlı referans)

| Dataset | Denek | Öznitelik matrisi | Hz→50 | Sensör |
|---------|-------|-------------------|-------|--------|
| MobiFall | 9 | 1808×24 | 87.5→50 | Pantolon cebi (telefon) |
| SisFall | 24 | 7738×24 | 200→50 | Bel |
| FAD_40Hz | 13 | 9481×24 | 40→50 | Bel |

**General (LOSO) F2 (tez):** SisFall ~0.97, FAD ~0.94–0.96, MobiFall ~0.90 (recall hep >0.95).
