# Örnek rapor (gerçek koşu)

[`tam-kosu/report.html`](tam-kosu/report.html) dosyasını tarayıcıda aç: etkileşimli panel (tıklanınca model detayı,
karşılaştırma, filtreler, koyu tema). Aynı sonuçların Markdown sürümü: [`report.md`](tam-kosu/report.md).
Doğrudan bir modele bağlantı örneği: `report.html#m=gliclass_large_v3`, koyu tema: `#theme=dark`.

Üretildiği komut:

```powershell
python -m clfbench run --robustness --name tam-kosu
```

**Bu bir model sıralaması değildir; boru hattının gerçek bir koşusudur.**

- **8 yerel model puanlandı** (TF-IDF, e5 sıfır-örnek ve LR, SetFit, GLiClass, Laya, mDeBERTa NLI, ModernBERT-NLI).
- **8 API modeli puanlanamadı** ve nedeni panelde "Atlananlar" bölümünde görünür: Jev (401, anahtar `api.typesafe.ai` tarafından
  reddedildi), Instinct ×3 (402, cüzdan boş), OpenRouter üzerindeki modeller (402, kredi yok). Bu durumlar gerçek API
  yanıtlarıdır; sistem boş istek atmadan modeli açık nedenle atlar.
- Veri: yerleşik `destek_tr` (gümüş etiketli, test = 151 örnek). Güven aralıkları geniştir (±6-8 puan).
- Gecikmeler 4 iş parçacıklı bir dizüstü CPU'sunda ölçüldü (NLI modelleri bu donanımda yavaştır).
- Eğitimli modeller `train` bölümünden öğrenir; sıfır örnekli modellerle veri bütçesi farklıdır, panelde ayrı grup olarak görünür.
- Laya multilingual bu Türkçe görevde zayıf çıktı (%33); sonuç bu veri ve etiket adlarına özgüdür.
