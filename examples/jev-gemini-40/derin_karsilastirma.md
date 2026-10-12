# Derin karşılaştırma — referans: Jev ücretsiz (BeatAPI, jev-1.13-free)

Koşu: `20261011-005911-jev-40 + 20261012-144545-gemini-40c` · veri: `destek_tr` (40 değerlendirme örneği, 10 puanlanmış model)

Bu belge ana raporu (`report.md`) tamamlar: aynı örnekler üzerinde modelleri **çiftler halinde ve örnek bazında** karşılaştırır. 40 örnekte tek bir örneğin ağırlığı 2,5 puandır; bu yüzden küçük farklar yorumlanmamalıdır.

## 1. Doğruluk sıralaması

| # | Model | Doğru / toplam | Doğruluk [%95 GA] |
|---:|:---|---:|---:|
| 1 | Jev ücretsiz (BeatAPI, jev-1.13-free) | 38 / 40 | %95,0 [%88–%100] |
| 2 | Gemini 3.5 Flash-Lite | 38 / 40 | %95,0 [%88–%100] |
| 3 | SetFit (MiniLM çok dilli, 8-shot) | 35 / 40 | %87,5 [%75–%98] |
| 4 | multilingual-e5-small + LR (eğitimli) | 33 / 40 | %82,5 [%70–%92] |
| 5 | TF-IDF + Lojistik Regresyon (eğitimli) | 31 / 40 | %77,5 [%65–%90] |
| 6 | GLiClass large v3.0 | 27 / 40 | %67,5 [%52–%80] |
| 7 | multilingual-e5-small (sıfır örnek, kosinüs) | 22 / 40 | %55,0 [%40–%70] |
| 8 | ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | 17 / 40 | %42,5 [%28–%57] |
| 9 | mDeBERTa-v3 çok dilli NLI (sıfır örnek) | 14 / 40 | %35,0 [%20–%50] |
| 10 | Laya multilingual (322M) | 13 / 40 | %32,5 [%18–%48] |

## 2. Jev ücretsiz (BeatAPI, jev-1.13-free) ile eşlenmiş McNemar testi

Yalnızca iki modelin **farklı** davrandığı örnekler bilgi taşır: *yalnız referans doğru* (b) ve *yalnız model doğru* (c). p değeri iki yönlü tam McNemar testidir; çok sayıda karşılaştırma yapıldığı için **Holm düzeltmeli** p ile karar verilir (α = 0,05). "Ayırt edilemez" = bu örneklemle fark kanıtlanamıyor, modellerin eşit olduğu anlamına gelmez.

| Model | Fark (puan) | İkisi doğru | Yalnız ref. doğru (b) | Yalnız model doğru (c) | İkisi de yanlış | p | p (Holm) | Sonuç |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| Gemini 3.5 Flash-Lite | +0,0 puan | 36 | 2 | 2 | 0 | 1,000 | 1,000 | ayırt edilemez |
| SetFit (MiniLM çok dilli, 8-shot) | -7,5 puan | 33 | 5 | 2 | 0 | 0,453 | 0,906 | ayırt edilemez |
| multilingual-e5-small + LR (eğitimli) | -12,5 puan | 31 | 7 | 2 | 0 | 0,180 | 0,539 | ayırt edilemez |
| TF-IDF + Lojistik Regresyon (eğitimli) | -17,5 puan | 29 | 9 | 2 | 0 | 0,065 | 0,262 | ayırt edilemez |
| GLiClass large v3.0 | -27,5 puan | 26 | 12 | 1 | 1 | 0,003 | 0,017 | referans daha iyi |
| multilingual-e5-small (sıfır örnek, kosinüs) | -40,0 puan | 22 | 16 | 0 | 2 | <0,001 | <0,001 | referans daha iyi |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | -52,5 puan | 17 | 21 | 0 | 2 | <0,001 | <0,001 | referans daha iyi |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | -60,0 puan | 13 | 25 | 1 | 1 | <0,001 | <0,001 | referans daha iyi |
| Laya multilingual (322M) | -62,5 puan | 13 | 25 | 0 | 2 | <0,001 | <0,001 | referans daha iyi |

## 3. Jev ücretsiz (BeatAPI, jev-1.13-free) hangi örneklerde yanıldı?

2 yanlış var. "Doğru bilen" sütunu, diğer 9 modelden kaçının bu örneği doğru bildiğidir. Etiketler gümüş etiketlidir: çok niyetli veya sınırda mesajlarda etiketin kendisi tartışmalı olabilir.

| Mesaj | Gerçek | Jev ücretsiz (BeatAPI, jev-1.13-free) | Güven | Doğru bilen |
|:---|:---|:---|---:|---:|
| Uygulama güzel ama kargo süreniz çok yavaş, bunu iyileştirmenizi öneririm. | sikayet_oneri | kargo_teslimat | 0,66 | 5 / 9 |
| Bildirimleriniz çok fazla, rahatsız edici olmaya başladı. | sikayet_oneri | teknik_destek | 0,82 | 5 / 9 |

**"Uygulama güzel ama kargo süreniz çok yavaş, bunu iyileştirmenizi öneririm."** — gerçek: `sikayet_oneri`

| Model | Tahmin | Sonuç |
|:---|:---|:---|
| Jev ücretsiz (BeatAPI, jev-1.13-free) | kargo_teslimat | yanlış |
| Laya multilingual (322M) | kargo_teslimat | yanlış |
| GLiClass large v3.0 | kargo_teslimat | yanlış |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | sikayet_oneri | doğru |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | kargo_teslimat | yanlış |
| multilingual-e5-small (sıfır örnek, kosinüs) | teknik_destek | yanlış |
| TF-IDF + Lojistik Regresyon (eğitimli) | sikayet_oneri | doğru |
| multilingual-e5-small + LR (eğitimli) | sikayet_oneri | doğru |
| SetFit (MiniLM çok dilli, 8-shot) | sikayet_oneri | doğru |
| Gemini 3.5 Flash-Lite | sikayet_oneri | doğru |

**"Bildirimleriniz çok fazla, rahatsız edici olmaya başladı."** — gerçek: `sikayet_oneri`

| Model | Tahmin | Sonuç |
|:---|:---|:---|
| Jev ücretsiz (BeatAPI, jev-1.13-free) | teknik_destek | yanlış |
| Laya multilingual (322M) | teknik_destek | yanlış |
| GLiClass large v3.0 | sikayet_oneri | doğru |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | diger | yanlış |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | iade_degisim | yanlış |
| multilingual-e5-small (sıfır örnek, kosinüs) | teknik_destek | yanlış |
| TF-IDF + Lojistik Regresyon (eğitimli) | sikayet_oneri | doğru |
| multilingual-e5-small + LR (eğitimli) | sikayet_oneri | doğru |
| SetFit (MiniLM çok dilli, 8-shot) | sikayet_oneri | doğru |
| Gemini 3.5 Flash-Lite | sikayet_oneri | doğru |

## 4. Örnek zorluğu

Her örneği kaç modelin doğru bildiği (10 modelden). Kimsenin bilemediği örnek, modelden çok **etiketin/örneğin** sorunlu olduğuna işaret eder.

| Doğru bilen model sayısı | Örnek sayısı |
|---:|---:|
| 2 | 1 |
| 3 | 1 |
| 4 | 4 |
| 5 | 7 |
| 6 | 4 |
| 7 | 10 |
| 8 | 3 |
| 9 | 6 |
| 10 | 4 |

Hiçbir modelin bilemediği örnek: **0**. Yalnızca referansın doğru bildiği örnek: **0**.

## 5. Tek sınıfa yığılma

Bir model tahminlerinin ≥%25'ini tek sınıfa veriyor ve bu pay o sınıfın gerçek payının ≥2 katıysa **yığılma** işaretlenir. Rastgele hata değil, sistematik bir sapma (ya da istem/etiket adı uyumsuzluğu) belirtisidir. "Hiç bilemediği sınıf": o sınıfın hiçbir örneğini doğru tahmin edememe.

| Model | En sık tahmin | Payı | Gerçek payı | Yığılma | Geçersiz | Hiç bilemediği sınıflar |
|:---|:---|---:|---:|:---|---:|:---|
| Laya multilingual (322M) | teknik_destek | %62 | %12 | **evet** | 0 | hesap_guvenlik, sikayet_oneri, diger |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | iade_degisim | %38 | %12 | **evet** | 0 | teknik_destek, diger |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | sikayet_oneri | %30 | %12 | **evet** | 0 | fatura_odeme, teknik_destek |
| multilingual-e5-small (sıfır örnek, kosinüs) | teknik_destek | %30 | %12 | **evet** | 0 | sikayet_oneri |
| GLiClass large v3.0 | satis_fiyat | %20 | %12 | hayır | 0 | — |
| TF-IDF + Lojistik Regresyon (eğitimli) | sikayet_oneri | %20 | %12 | hayır | 0 | — |
| Jev ücretsiz (BeatAPI, jev-1.13-free) | kargo_teslimat | %15 | %12 | hayır | 0 | — |
| multilingual-e5-small + LR (eğitimli) | iade_degisim | %15 | %12 | hayır | 0 | — |
| SetFit (MiniLM çok dilli, 8-shot) | kargo_teslimat | %15 | %12 | hayır | 0 | — |
| Gemini 3.5 Flash-Lite | kargo_teslimat | %15 | %12 | hayır | 0 | — |

## 6. Güven davranışı

Doğru ve yanlış yanıtlardaki ortalama güven. İyi bir modelde yanlış yanıtların güveni belirgin biçimde düşüktür; iki değer birbirine yakınsa model nerede yanıldığını bilmiyor demektir (seçici tahmin için işe yaramaz). Güven türleri aynı ölçekte değildir; yanlış sayısı çok az olduğunda ortalama güvenilir değildir.

| Model | Güven türü | Doğrularda ort. | Yanlışlarda ort. | Yanlış sayısı |
|:---|:---|---:|---:|---:|
| Jev ücretsiz (BeatAPI, jev-1.13-free) | yerel olasılık | 0,96 | 0,74 | 2 |
| Laya multilingual (322M) | yerel olasılık | 0,88 | 0,86 | 27 |
| GLiClass large v3.0 | türetilmiş, yerel olasılık | 1,00 | 1,00 | 13 |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | yerel olasılık | 0,59 | 0,49 | 26 |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | yerel olasılık | 0,40 | 0,32 | 23 |
| multilingual-e5-small (sıfır örnek, kosinüs) | türetilmiş | 0,23 | 0,20 | 18 |
| TF-IDF + Lojistik Regresyon (eğitimli) | yerel olasılık | 0,51 | 0,30 | 9 |
| multilingual-e5-small + LR (eğitimli) | yerel olasılık | 0,53 | 0,34 | 7 |
| SetFit (MiniLM çok dilli, 8-shot) | yerel olasılık | 0,70 | 0,51 | 5 |
| Gemini 3.5 Flash-Lite | sözel güven | 0,98 | 0,93 | 2 |

## Sınırlar

- **Örneklem:** 40 örnek. McNemar testinde yalnızca farklı davranılan örnekler sayılır; b+c küçükse test güçsüzdür.
- **Eşit veri bütçesi yok:** eğitimli modeller `train` bölümünden öğrenir, sıfır örnekli modeller (API/encoder) öğrenmez; bu grupları kıyaslarken doğruluk tek başına adil bir ölçüt değildir.
- **Gümüş etiket:** yerleşik veri insan etiketli altın küme değildir; sıralamayı değil, boru hattını ve eğilimleri gösterir.
- **Tek koşu:** farklı tohum/örneklemde sonuçlar değişebilir; karar için kendi insan etiketli verinizle yeniden çalıştırın.
