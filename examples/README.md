# Örnek koşular (gerçek)

| Klasör | Ne | İçerik |
|---|---|---|
| [`inception-151/`](inception-151/report.html) | **Tam test bölümü (151 örnek)**, seçenek sırası testi dahil | Inception API (Mercury Decide, Mercury 2.5) + 8 yerel model. Yayınlanabilir site bundan üretildi: [`site/`](../site/index.html) |
| [`jev-40/`](jev-40/report.html) | Dengeli 40 örnek | Ücretsiz Jev (BeatAPI) + 8 yerel model |

## inception-151

```powershell
python -m clfbench run --robustness --name inception-yerel
python -m clfbench site examples/inception-151        # -> site/
```

- **10 model puanlandı** (151 örnek): Mercury Decide %93,4 [%89–%97], Mercury 2.5 %96,7 [%93–%99]; eğitimli ve yerel modeller
  daha düşük. Mercury Decide ile Mercury 2.5 arasındaki fark bu örnek sayısında **ayırt edilemez** (eşlenmiş fark −3,3 puan, %95
  aralık [−8, +1]); gecikme ve maliyet farkı ise büyüktür (p50 ~290 ms ve ~$0,009 / 1.000 karar, ~2.000 ms ve ~$0,13).
- **15 model çalıştırılamadı** (anahtarları tanımlı değildi; `run.json > not_run` içinde kayıtlı). Sıralama yalnızca
  ölçülenler arasındadır, tüm piyasayı kapsamaz.
- Yerel modellerin gecikmesi bulut kapsayıcısının CPU'sunda, API gecikmesi ağ üzerinden ölçüldü: birbiriyle birebir kıyaslanamaz.
- Veri yerleşik `destek_tr` (gümüş etiketli, ~151 test örneği: güven aralığı ±6-7 puan). `jev-40` koşusuyla aynı yerel modeller
  benzer büyüklük sırasında çıktı (örn. Laya %33, mDeBERTa %30, e5+LR %81), yani sonuçlar kuruluma özgü değil.
- Mercury 2.5'te `max_tokens` 512/2048 iken bazı yanıtlar boş döndü (muhakeme tokenları bütçeyi tüketti); 4096'da 0 hata.

## jev-40 (gerçek koşu)

[`jev-40/report.html`](jev-40/report.html) dosyasını tarayıcıda aç: etkileşimli panel (tıklanınca model detayı,
karşılaştırma, filtreler, koyu tema). Aynı sonuçların Markdown sürümü: [`report.md`](jev-40/report.md).
Doğrudan bir modele bağlantı: `report.html#m=jev_beatapi`, koyu tema: `#theme=dark`.
Jev'i diğer modellerle örnek bazında karşılaştıran derin analiz: [`derin_karsilastirma.md`](jev-40/derin_karsilastirma.md)
(`python -m clfbench compare examples/jev-40 --reference jev_beatapi` ile yeniden üretilir).

**Yeni: [`jev-gemini-40/`](jev-gemini-40/report.html)** — aynı 40 örnek, `jev-40` + gerçek Gemini 3.5 Flash-Lite koşusu birleştirilmiş
(`python -m clfbench merge examples/jev-40 <gemini-koşusu> --out examples/jev-gemini-40 --reference jev_beatapi`). Gemini 40/40 geçerli
yanıtla %95,0 aldı (Jev ile aynı); eşlenmiş testte ayırt edilemez (b=2, c=2). Gemini aynı örnekler ve ayarlarla ayrı oturumda (Linux, bulut)
koşuldu; ücretsiz katman kotası (HTTP 429) yüzünden `--concurrency 1` ile birkaç denemede tamamlandı, başarısız istekler yeniden denendi
(model yanıtı değil, kota hatasıydı). Gecikmeler farklı makine/ağlardan geldiği için kıyaslanamaz.

Üretildiği komut:

```powershell
python -m clfbench run --limit 40 --concurrency 1 --name jev-40
```

**Bu bir model sıralaması değildir; boru hattının gerçek bir koşusudur.**

- **9 model puanlandı:** Jev ücretsiz (BeatAPI `jev-1.13-free`, gerçek API) + 8 yerel model (TF-IDF, e5 sıfır-örnek ve LR, SetFit,
  GLiClass, Laya, mDeBERTa NLI, ModernBERT-NLI).
- **Yalnızca 40 örnek (sınıf başına 5):** BeatAPI'nin ücretsiz katmanı hesap başına dakikada 1 başarılı isteğe izin veriyor
  (151 örnek ≈ 2,5 saat). Bu yüzden *tüm modeller aynı dengeli 40 örnekte* değerlendirildi. %95 güven aralıkları çok geniştir
  (örn. Jev %95,0 [%88–%100]); modeller arasında kesin sıralama çıkarılamaz.
- **Jev bu 40 örnekte öne çıktı** (doğruluk %95, kalibrasyon hatası 3,8 puan, p50 gecikme ~0,5 sn, örnek başına ~760 giriş tokenı).
  Gecikme API'ye uçtan uca istek süresidir, hız limiti beklemesi hariçtir. Daha fazla örnekle (bakiye yükleyince limit artıyor) doğrulanmalı.
- **8 API modeli puanlanamadı** ve nedeni panelde "Atlananlar" bölümünde görünür: Jev (jev-ai.pro, 402 bakiye yok), Instinct ×3
  (402 cüzdan boş), OpenRouter üzerindeki 4 model (402 kredi yok). Bunlar gerçek API yanıtlarıdır; anahtarlar geçerli, hesaplarda
  bakiye yok. Sistem boş istek atmadan modeli açık nedenle atlar.
- **14 kayıtlı model bu koşuya girmedi** (8'inin anahtarı yoktu, 6'sı varsayılan kapalı); panelde ve `report.md` içinde "Çalıştırılmayan
  modeller" bölümünde nedenleri ve nasıl açılacakları var. Not: bu koşu `not_run` / `free` alanları eklenmeden önce alındığı için,
  bu alanlar `run.json` ve `summary.json`'a kayıt defterinden **geriye dönük** eklendi (koşu yeniden çalıştırılmadı, sonuçlar değişmedi).
  Yeni koşular bu alanları kendileri yazar.
- Veri: yerleşik `destek_tr` (gümüş etiketli). Gecikmeler 4 iş parçacıklı bir dizüstü CPU'sunda ölçüldü.
- Eğitimli modeller `train` bölümünden öğrenir; sıfır örnekli modellerle veri bütçesi farklıdır, panelde ayrı grup olarak görünür.
- Laya multilingual bu Türkçe görevde zayıf çıktı; sonuç bu veri ve etiket adlarına özgüdür.
