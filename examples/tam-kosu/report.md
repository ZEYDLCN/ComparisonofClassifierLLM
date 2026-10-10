# Classifier LLM karşılaştırma raporu

Koşu: `20261011-000203-tam-kosu` · oluşturma: 2026-10-11T00:26:39

## Koşul özeti

**Veri:** `destek_tr` · değerlendirme bölümü: `test` (151 örnek) · eğitim bölümü: 128 örnek · parmak izi `9ac13de53ff85f2d`

**Sınıf dağılımı (değerlendirme):** fatura_odeme: 18, teknik_destek: 19, kargo_teslimat: 19, iade_degisim: 19, hesap_guvenlik: 19, satis_fiyat: 19, sikayet_oneri: 19, diger: 19

**Ayarlar:** tohum 0 · etiket başına few-shot 0 · istem sürümü `v1` · seçenek sırası testi: evet

**Ortam:** Python 3.13.14 · Windows-10-10.0.19045-SP0

**Not:** Gerçek anahtarlarla ilk koşu: Jev/Instinct/OpenRouter erişim hataları rapora yansır; yerel modeller 4 iş parçacıklı dizüstü CPU'da

## Öne çıkan bulgular

**En yüksek doğruluk:** multilingual-e5-small + LR (eğitimli) — %80,8 (%95 GA %74,2–%86,8), macro-F1 %80,7.

**İstatistiksel olarak ayırt edilemeyenler (multilingual-e5-small + LR (eğitimli) ile eşlenmiş bootstrap farkının %95 aralığı 0'ı içeriyor):** SetFit (MiniLM çok dilli, 8-shot). Bu modeller arasında bu örneklemle kesin sıralama yapılamaz.

**En düşük p50 gecikme:** TF-IDF + Lojistik Regresyon (eğitimli) — 1,4 ms (doğruluk %70,2).

**Doğruluk–gecikme Pareto sınırı:** multilingual-e5-small + LR (eğitimli), SetFit (MiniLM çok dilli, 8-shot), TF-IDF + Lojistik Regresyon (eğitimli).

![Öne çıkan bulgular](charts/accuracy.png)

## Ana sonuç tablosu

Doğruluk sıkıdır: geçersiz veya başarısız yanıtlar yanlış sayılır, oranı ayrıca verilir. Köşeli parantez: %95 bootstrap güven aralığı. ECE: kalibrasyon hatası (yüzde puan; düşük iyi). Gecikme: istemci tarafı uçtan uca ms. Maliyet: ölçülen token × tarife, 1.000 karar için USD; `~` token sayısının API'den gelmeyip tahmin edildiğini, `yerel` API ücreti olmadığını gösterir (altyapı maliyeti hariç).

| # | Model | Grup | Doğruluk [%95 GA] | Macro-F1 | Geçersiz | ECE | p50 ms | p95 ms | $/1.000 karar |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | multilingual-e5-small + LR (eğitimli) | Eğitimli | %80,8 [%74–%87] | %80,7 | %0,0 | 34,0 pp | 24,9 | 31,1 | yerel |
| 2 | SetFit (MiniLM çok dilli, 8-shot) | Eğitimli | %74,8 [%68–%81] | %75,1 | %0,0 | 11,7 pp | 20,0 | 25,5 | yerel |
| 3 | TF-IDF + Lojistik Regresyon (eğitimli) | Eğitimli | %70,2 [%62–%77] | %69,7 | %0,0 | 25,8 pp | 1,4 | 2,5 | yerel |
| 4 | GLiClass large v3.0 | Encoder | %60,3 [%52–%68] | %59,4 | %0,0 | 39,7 pp | 795 | 926 | yerel |
| 5 | multilingual-e5-small (sıfır örnek, kosinüs) | Encoder | %58,3 [%50–%66] | %55,6 | %0,0 | 37,3 pp | 24,6 | 29,5 | yerel |
| 6 | ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | Encoder | %38,4 [%31–%46] | %36,7 | %0,0 | 11,3 pp | 549 | 688 | yerel |
| 7 | Laya multilingual (322M) | Encoder | %33,1 [%26–%40] | %28,2 | %0,0 | 56,9 pp | 532 | 637 | yerel |
| 8 | mDeBERTa-v3 çok dilli NLI (sıfır örnek) | Encoder | %30,5 [%23–%38] | %32,1 | %0,0 | 17,4 pp | 5.755 | 6.818 | yerel |

## Kalibrasyon, hız ve kullanım ayrıntıları

Brier: çok sınıflı olasılık hatası (düşük iyi). **Güven türü** karşılaştırmayı sınırlar: üretken LLM'lerde 'sözel güven' modelin yazdığı sayıdır, yerel olasılık değildir; kalibrasyon sayıları bu yüzden doğrudan kıyaslanmamalıdır. Seçici doğruluk: modelin en emin olduğu %80 örnekteki doğruluk ve doğruluğu ≥%95 tutan en büyük kapsama. İlk istek (soğuk) gecikmesi ve kurulum/yükleme/eğitim süresi ayrı verilir; p50/p95 ısınmadan sonraki isteklerdir.

| Model | Brier | Güven türü | Doğruluk@%80 kapsama | Kapsama@%95 doğruluk | p99 ms | Soğuk ms | Kurulum | Eşzamanlılık | Ort. token (g/ç) |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| multilingual-e5-small + LR (eğitimli) | 0,429 | yerel olasılık | %86,0 | %28 | 34,0 | — | 18,4 sn | 1 | — |
| SetFit (MiniLM çok dilli, 8-shot) | 0,391 | yerel olasılık | %81,8 | %40 | 27,3 | 24,0 | 225,1 sn | 1 | — |
| TF-IDF + Lojistik Regresyon (eğitimli) | 0,504 | yerel olasılık | %76,9 | %25 | 3,6 | — | 0,4 sn | 1 | — |
| GLiClass large v3.0 | 0,795 | türetilmiş, yerel olasılık | %60,3 | %1 | 1.026 | 1.390 | 397,6 sn | 1 | — |
| multilingual-e5-small (sıfır örnek, kosinüs) | 0,760 | türetilmiş | %66,1 | %14 | 33,7 | — | 9,3 sn | 1 | — |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | 0,737 | yerel olasılık | %45,5 | %1 | 721 | — | 116,7 sn | 1 | — |
| Laya multilingual (322M) | 1,201 | yerel olasılık | %33,1 | %0 | 706 | 136.463 | 0,1 sn | 1 | — |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | 0,850 | yerel olasılık | %33,1 | %3 | 7.629 | — | 119,6 sn | 1 | — |

## Doğruluk ve gecikme

Gecikme ölçümleri farklı donanım, bağlantı ve eşzamanlılık koşullarında alınmıştır (API: bu makineden uçtan uca; yerel: bu makinenin CPU/GPU'su). Kontrollü bir donanım yarışı değildir.

![Doğruluk ve gecikme](charts/latency.png)

## Eşlenmiş karşılaştırma (referans: multilingual-e5-small + LR (eğitimli))

Aynı örnekler üzerinde doğruluk farkı (model − referans) ve %95 eşlenmiş bootstrap aralığı. Aralık 0'ı içeriyorsa bu örneklemle fark kanıtlanamaz.

| Model | Fark (puan) | %95 aralık | Sonuç |
|:---|---:|---:|:---|
| SetFit (MiniLM çok dilli, 8-shot) | -6,0 puan | -14,6 puan … +2,0 puan | ayırt edilemez |
| TF-IDF + Lojistik Regresyon (eğitimli) | -10,6 puan | -18,5 puan … -3,3 puan | daha kötü |
| GLiClass large v3.0 | -20,5 puan | -29,8 puan … -11,3 puan | daha kötü |
| multilingual-e5-small (sıfır örnek, kosinüs) | -22,5 puan | -31,1 puan … -13,9 puan | daha kötü |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | -42,4 puan | -51,7 puan … -33,1 puan | daha kötü |
| Laya multilingual (322M) | -47,7 puan | -57,0 puan … -38,4 puan | daha kötü |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | -50,3 puan | -60,3 puan … -40,4 puan | daha kötü |

## Sınıf bazında F1

Satır başına model, sütun başına sınıf. Destek sayısı küçük olduğundan tek sınıftaki farklar gürültülü olabilir.

| Model | fatura_odeme | teknik_destek | kargo_teslimat | iade_degisim | hesap_guvenlik | satis_fiyat | sikayet_oneri | diger |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| multilingual-e5-small + LR (eğitimli) | %78 | %83 | %82 | %80 | %81 | %84 | %83 | %74 |
| SetFit (MiniLM çok dilli, 8-shot) | %79 | %84 | %76 | %74 | %87 | %67 | %76 | %57 |
| TF-IDF + Lojistik Regresyon (eğitimli) | %65 | %73 | %68 | %76 | %75 | %72 | %69 | %60 |
| GLiClass large v3.0 | %62 | %56 | %75 | %80 | %51 | %61 | %30 | %61 |
| multilingual-e5-small (sıfır örnek, kosinüs) | %65 | %69 | %73 | %65 | %52 | %57 | %38 | %26 |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | %47 | %19 | %59 | %43 | %38 | %36 | %43 | %9 |
| Laya multilingual (322M) | %42 | %31 | %55 | %56 | %10 | %32 | %0 | %0 |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | %36 | %10 | %69 | %23 | %58 | %30 | %21 | %11 |

## Karışıklık matrisleri (en doğru 3 model)

Satırlar gerçek, sütunlar tahmin edilen sınıf; renk satır normalizasyonudur, hücrelerdeki sayılar örnek sayısıdır.

![Karışıklık matrisleri (en doğru 3 model)](charts/confusion.png)

## Zor örnek dilimleri

Doğruluk, örnek etiketine göre. En az 5 örnek içeren dilimler gösterilir; n küçük olduğundan yalnızca yön göstericidir.

| Model | Türkçe karaktersiz (ascii) | Çok niyetli | Çok kısa (≤4 sözcük) |
|:---|---:|---:|---:|
| multilingual-e5-small + LR (eğitimli) | %80 (n=5) | %40 (n=5) | %75 (n=12) |
| SetFit (MiniLM çok dilli, 8-shot) | %60 (n=5) | %60 (n=5) | %67 (n=12) |
| TF-IDF + Lojistik Regresyon (eğitimli) | %80 (n=5) | %40 (n=5) | %50 (n=12) |
| GLiClass large v3.0 | %80 (n=5) | %40 (n=5) | %58 (n=12) |
| multilingual-e5-small (sıfır örnek, kosinüs) | %80 (n=5) | %60 (n=5) | %50 (n=12) |
| ModernBERT-base NLI (yalnız İngilizce; negatif kontrol) | %60 (n=5) | %0 (n=5) | %25 (n=12) |
| Laya multilingual (322M) | %0 (n=5) | %40 (n=5) | %25 (n=12) |
| mDeBERTa-v3 çok dilli NLI (sıfır örnek) | %80 (n=5) | %40 (n=5) | %25 (n=12) |

## Seçenek sırası dayanıklılığı

Etiket listesi her örnek için rastgele sıralanıp model yeniden çalıştırıldı. Tutarlılık: iki sırada aynı etiketi seçme oranı. Sıradan etkilenmeyen modeller (TF-IDF, gömme, NLI) bu testten muaftır.

![Seçenek sırası dayanıklılığı](charts/robustness.png)

| Model | Doğruluk (varsayılan) | Doğruluk (karıştırılmış) | Fark | Tutarlılık |
|:---|---:|---:|---:|---:|
| GLiClass large v3.0 | %60,3 | %57,0 | -3,3 puan | %88,7 |
| Laya multilingual (322M) | %33,1 | %37,1 | +4,0 puan | %81,5 |

## Atlanan veya hatalı modeller

Bu modeller ya hiç puanlanamadı ya da bazı istekleri başarısız oldu. Başarısız istekler doğrulukta yanlış sayılır.

| Model | Durum | Neden / ilk hata |
|:---|:---|:---|
| Jev 1.13 (TypeSafe) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 401: {"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}). Anahtarı, bakiyeyi, model adını ve base_url/path'i kontrol edin: 'python  |
| Instinct 27B (ZooData) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"success":false,"error":{"code":"PAYMENT_REQUIRED","message":"wallet_not_provisioned"},"meta":{"requestId":"req_32e94b2317fd427c","timestamp":"2026-10-10T21:02:05.645944Z"}}). Anahtarı, bakiyeyi, model adını ve ba |
| Instinct tuned 4B (ZooData) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"success":false,"error":{"code":"PAYMENT_REQUIRED","message":"wallet_not_provisioned"},"meta":{"requestId":"req_d59998bedb9045d7","timestamp":"2026-10-10T21:02:05.872314Z"}}). Anahtarı, bakiyeyi, model adını ve ba |
| Instinct dual 4B (ZooData) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"success":false,"error":{"code":"PAYMENT_REQUIRED","message":"wallet_not_provisioned"},"meta":{"requestId":"req_3a0ae83d22504730","timestamp":"2026-10-10T21:02:06.059725Z"}}). Anahtarı, bakiyeyi, model adını ve ba |
| Mercury 2.5 (sohbet, OpenRouter) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settin). Anahtarı, bakiyeyi, m |
| GPT-6 Luna (sohbet, OpenRouter) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settin). Anahtarı, bakiyeyi, m |
| DeepSeek V4 Flash (OpenRouter) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settin). Anahtarı, bakiyeyi, m |
| Qwen3.8 Flash (OpenRouter) | atlandı | ProviderError: Erişim/ödeme hatası (HTTP 402: {"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settin). Anahtarı, bakiyeyi, m |

## Yöntem ve sınırlar

- **Veri:** Yerleşik `destek_tr` seti, bu çalışma için hazırlanmış *gümüş* etiketli bir örnek kümesidir; insan etiketli altın küme DEĞİLDİR. Bir satın alma kararı için kendi, insan etiketli Türkçe verinizle (`--dataset`) yeniden çalıştırın.

- **Bölme:** Değerlendirme yalnızca `test` (veya istenen) bölümünde yapılır; eğitimli modeller yalnızca `train` bölümünü görür; few-shot örnekleri de `train`'den alınır ve değerlendirmeden çıkarılır.

- **Aynı görev, doğal arayüz:** Karar API'leri kendi `choice` şemasıyla, üretken LLM'ler aynı Türkçe talimat + JSON şemasıyla, encoder modeller etiketlerin kısa Türkçe adlarıyla çağrılır. Bu, her modelin doğal arayüzünü adil kullanır ama modeller arasındaki farkın bir kısmı istem/arayüz farkından gelebilir.

- **Güven/olasılık:** Karar API'leri ve yerel modeller yerel olasılık verir; üretken LLM'ler sözel güven yazar. Brier/ECE bu yüzden aynı ölçekte değildir.

- **Gecikme:** İstemci tarafı uçtan uca süre. Konum, ağ, eşzamanlılık ve sunucu yükü sonucu etkiler; ısınma isteği hariç tutulur. Yerel gecikmeler bu makinenin donanımına aittir.

- **Maliyet:** Yanıttan okunan token kullanımı × tarife. API token vermiyorsa tahmin edilir ve `~` ile işaretlenir. Abonelik, minimum bakiye, yeniden deneme ve altyapı hariçtir.

- **İstatistik:** Güven aralıkları örnekler üzerinde bootstrap (2000) ile hesaplanır; küçük farklardan kazanan çıkarılamaz.

- **Tarifeler (USD / 1M giriş / çıkış):** Jev 1.13 (TypeSafe): 0,042 / 0,000 (TypeSafe liste tarifesi); Instinct 27B (ZooData): 0,030 / 0,000 (En az 20 USD bakiye yüklemesi gerekir); Instinct tuned 4B (ZooData): 0,010 / 0,000 (En az 20 USD bakiye yüklemesi gerekir); Instinct dual 4B (ZooData): 0,010 / 0,000 (En az 20 USD bakiye yüklemesi gerekir); Mercury 2.5 (sohbet, OpenRouter): 0,040 / 0,150; GPT-6 Luna (sohbet, OpenRouter): 0,100 / 0,500; DeepSeek V4 Flash (OpenRouter): 0,030 / 1,280 (Sağlayıcıya göre değişebilir; reasoning çıktı tokenı yükseltir); Qwen3.8 Flash (OpenRouter): 0,150 / 0,470

- **Sürüm:** clfbench 0.1.0 · git 8acd27b · paketler: requests 2.34.2, numpy 2.5.4, anthropic 1.13.0, scikit-learn 1.9.1, torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0, gliclass 0.1.20, laya 0.4.2, setfit 1.2.0
