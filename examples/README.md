# Örnek rapor (gerçek koşu)

[`jev-40/report.html`](jev-40/report.html) dosyasını tarayıcıda aç: etkileşimli panel (tıklanınca model detayı,
karşılaştırma, filtreler, koyu tema). Aynı sonuçların Markdown sürümü: [`report.md`](jev-40/report.md).
Doğrudan bir modele bağlantı: `report.html#m=jev_beatapi`, koyu tema: `#theme=dark`.
Jev'i diğer modellerle örnek bazında karşılaştıran derin analiz: [`derin_karsilastirma.md`](jev-40/derin_karsilastirma.md)
(`python -m clfbench compare examples/jev-40 --reference jev_beatapi` ile yeniden üretilir).

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
