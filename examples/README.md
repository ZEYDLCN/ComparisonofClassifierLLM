# Örnek rapor

[`yerel-taban/report.html`](yerel-taban/report.html) (veya [`report.md`](yerel-taban/report.md)) altyapının gerçek bir koşusunun çıktısıdır:

```powershell
python -m clfbench run --models tfidf_lr,embedding_e5_lr,embedding_e5_zeroshot,nli_mdeberta_xnli,nli_modernbert_en --robustness --name yerel-taban
```

**Bu bir model sıralaması değildir; yalnızca raporun nasıl göründüğünü ve boru hattının çalıştığını gösterir.**

- Yalnızca **yerel** modeller var (API anahtarı gerektirmeyenler). Karar API'leri, Claude/Gemini vb. için anahtarlar
  ana README'deki tabloya göre `.env` dosyasına konunca aynı komut onları da ekler.
- Veri: yerleşik `destek_tr` (gümüş etiketli, test = 151 örnek). Güven aralıkları geniştir (±6-8 puan).
- Gecikmeler 4 iş parçacıklı bir dizüstü CPU'sunda ölçülmüştür (özellikle NLI modelleri bu donanımda yavaştır).
- Eğitimli modeller `train` bölümünden öğrenir; sıfır örnekli modellerle veri bütçesi farklıdır, raporda ayrı grupta gösterilir.
- Sıfır örnekli NLI modellerinin düşük puanı bu görev/etiket adları için geçerlidir (şablon değişikliği dev bölümünde
  sonucu iyileştirmedi); başka bir görevde farklı olabilir.
