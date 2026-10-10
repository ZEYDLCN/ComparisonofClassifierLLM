# ComparisonofClassifierLLM

Karar (classifier) modellerini, üretken LLM'leri, encoder modelleri ve eğitimli temel modelleri **aynı Türkçe veri
üzerinde, aynı metriklerle** karşılaştıran uçtan uca bir altyapı. Doğruluk, macro-F1, kalibrasyon, seçici doğruluk,
gecikme (p50/p95/p99), gerçek token kullanımından maliyet ve istatistiksel güven aralıkları tek komutla ölçülür;
çıktı Türkçe bir HTML/Markdown rapordur.

Projenin dayanağı: [classifier_llm_arastirmasi.md](classifier_llm_arastirmasi.md) — Jev, Laya, d1, Mercury Decide,
OpenAI Decisions, Perplexity/Quyet/H2O, GLiClass, SetFit vb. adayların yayımlanmış sonuçlarının araştırması.
Araştırmanın sonucu şuydu: *tek bir evrensel kazanan yok; Türkçe için ortak bir test yok.* Bu depo, o testi **kendi
verinle** yapmanı sağlar.

> ⚠️ **Dürüstlük notu.** Yerleşik `destek_tr` veri seti (327 örnek, 8 sınıf) bu çalışma için hazırlanmış, **gümüş
> etiketli** bir örnek kümesidir; insan etiketli altın küme değildir ve test bölümü ~150 örnektir (güven aralığı ±6-7 puan).
> Yöntemi ve raporu denemek, boru hattını doğrulamak için uygundur. **Satın alma/üretim kararı için** `--dataset` ile
> kendi insan etiketli Türkçe verini ver (aşağıda).

---

## Hızlı başlangıç

```powershell
git clone https://github.com/ZEYDLCN/ComparisonofClassifierLLM.git
cd ComparisonofClassifierLLM

python -m venv .venv
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -e ".[anthropic,local]"        # + isteğe bağlı: ,gliclass,laya,setfit,hf

copy .env.example .env                     # macOS/Linux: cp .env.example .env
# .env dosyasını aç, sahip olduğun servislerin anahtarlarını yaz (aşağıdaki tabloya bak)

python -m clfbench doctor                  # hangi paket/anahtar hazır, hangi model çalışacak?
python -m clfbench ping jev                # tek istek: anahtar ve uç nokta doğru mu?
python -m clfbench estimate --robustness   # çalıştırmadan ÖNCE tahmini API maliyeti
python -m clfbench run --robustness        # hepsini çalıştır → results/<tarih>/report.html
```

Anahtarı olmayan model **otomatik atlanır**; hiçbir anahtar olmadan bile yerel modellerle (TF-IDF, embedding, NLI)
tüm boru hattı çalışır. Rapor `results/<koşu>/report.html` (tek dosya, grafikler gömülü) ve `report.md` olarak üretilir.

Her şeyi (Laya, GLiClass, SetFit dahil) tek seferde kurmak için `pip install -e ".[all]"`. Bu kurulum temiz bir
sanal ortamda (Python 3.13, Windows) test edildi: 38 test geçti ve yerel sağlayıcıların hepsi gerçek modelle çalıştı.

### Disk ve süre notu (yerel modeller)

Yerel modeller ilk çalıştırmada Hugging Face'ten indirilir. Diskin dolu olduğu makinelerde `HF_HOME` ile önbelleği
başka sürücüye alabilirsin (örn. PowerShell: `setx HF_HOME D:\hf-cache`, yeni terminal aç).

| Model | İlk indirme | CPU'da hız (151 örnek) |
|---|---|---|
| TF-IDF + LR | yok | saniyeler |
| multilingual-e5-small (sıfır örnek / LR) | ~0,46 GB | ~25 ms/örnek |
| SetFit (MiniLM çok dilli) | ~0,45 GB | eğitim ~3-5 dk, sonra ~30 ms/örnek |
| Laya multilingual | ~0,6 GB | kısa; ilk çağrıda model indirilir |
| mDeBERTa-v3 NLI / ModernBERT-NLI | ~0,55 GB / ~0,56 GB | **~5 sn/örnek** (8 etiket × NLI) ≈ 12 dk |
| GLiClass large v3 | ~1,6 GB | ~1-2 sn/örnek |

NLI modelleri CPU'da yavaştır; yalnız API modellerini çalıştırmak için `--groups karar-api,uretken-llm`,
yalnız hızlı yerel modeller için `--models tfidf_lr,embedding_e5_lr,setfit_minilm` kullan.

### Sık kullanılan komutlar

| Komut | Ne yapar |
|---|---|
| `python -m clfbench doctor` | Paket, `.env` ve anahtar durumunu, çalışacak/atlanacak modelleri gösterir |
| `python -m clfbench keys --missing` | Eksik anahtarların nereden alınacağını anlatır |
| `python -m clfbench models --all` | Kayıtlı tüm modeller, fiyatlar, durum |
| `python -m clfbench ping <model>` | Tek modele tek istek (anahtar/uç nokta testi, gecikme, token) |
| `python -m clfbench estimate` | Koşu öncesi kaba maliyet tahmini |
| `python -m clfbench run --models jev,claude_haiku_5_5 --limit 40` | Hızlı deneme (40 örnek, iki model) |
| `python -m clfbench run --groups karar-api` | Yalnız karar API'leri |
| `python -m clfbench run --few-shot 2` | Üretken LLM'lere etiket başına 2 örnek ver |
| `python -m clfbench run --robustness` | Seçenek sırasını karıştırıp tekrarla (dayanıklılık) |
| `python -m clfbench report <klasör> --reference jev` | Raporu başka referans modelle yeniden üret |

**Hız limitli servisler:** bazı ücretsiz katmanlar çok düşük limit verir (örn. BeatAPI ücretsiz Jev: dakikada 1 istek).
`run` ve `estimate` bunu (`rpm_limit`) bilen modeller için çalıştırmadan önce tahmini süreyi uyarır. Tüm modelleri aynı dengeli
alt kümede kıyaslamak için `python -m clfbench run --limit 40` kullan (sınıf başına 5 örnek; yerel sonuçlar önbellekten gelir).

Maliyet güvencesi: tahmini maliyet `--max-cost` (varsayılan 2 USD) değerini aşarsa `--yes` vermeden çalışmaz.
Aynı koşu tekrarlanırsa başarılı yanıtlar `results/cache` içinden gelir (para harcanmaz); yarıda kesilen koşu kaldığı yerden sürer.
`--no-cache` ile sıfırdan çalıştırılır.

---

## API anahtarlarını nereden alırım?

Hepsi için kısa özet `python -m clfbench keys` ile de görülebilir. Anahtarı **`.env`** dosyasına yaz
(`.env` `.gitignore`'dadır; asla commit'lenmez, kimseyle paylaşma).

| `.env` değişkeni | Servis → modeller | Anahtar adresi | Not |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic → Claude Haiku/Sonnet 5.5 | <https://platform.claude.com/settings/keys> | Ön ödemeli kredi gerekir |
| `INCEPTION_API_KEY` | Inception → **Mercury Decide**, Mercury 2.5 | <https://platform.inceptionlabs.ai/dashboard/api-keys> | **Yeni kullanıcıya 100 milyon ücretsiz token, kart gerekmiyor** — başlamak için en kolayı |
| `LIQUID_API_KEY` | Liquid AI → **d1** | <https://console.liquid.ai> | Kaydol → organizasyona katıl → Dashboard → API Keys (`liquid_...`) |
| `OPENAI_API_KEY` | OpenAI → **Decisions** (gpt-6-luna) | <https://platform.openai.com/api-keys> | Decisions public beta; hesabında erişim yoksa 403/404 döner |
| `GEMINI_API_KEY` | Google → Gemini 3.5 Flash-Lite | <https://aistudio.google.com/apikey> | Ücretsiz katman olabilir (düşük hız limiti) |
| `OPENROUTER_API_KEY` | OpenRouter → GPT-6 Luna (sohbet), DeepSeek V4 Flash, Qwen3.8 Flash | <https://openrouter.ai/keys> | Tek anahtar, çok model; Credits yükle |
| `BEATAPI_API_KEY` | BeatAPI → **Jev ücretsiz** (`jev-1.13-free`) | <https://beatapi.io/dashboard/apikeys> | **Gerçek yanıtla doğrulandı: sıfır bakiyede çalışıyor, ama hesap başına dakikada 1 başarılı istek** (429 `rate_limit_exceeded`). Tam test (151 örnek) ~2,5 saat; `--limit 40` ≈ 40 dk. Bakiye yükleyenlere daha yüksek limit veriliyor |
| `JEV_AI_API_KEY` | Jev AI (jev-ai.pro) → **Jev** | <https://jev-ai.pro> | Anahtar `sk_jev-ai_` ile başlar. **Bakiye yoksa 402 döner**; yükleme: <https://jev-ai.pro/pricing?context=api&mode=top-up>. İstekte `model` alanı yok |
| `TYPESAFE_API_KEY` | TypeSafe → **Jev** | <https://docs.typesafe.ai> | TypeSafe konsolundaki *API Keys* sayfası. Üçüncü taraf kaynaklara göre yeni kayıtlar 22 Eylül 2026'da duraklatıldı. **Vercel AI Gateway / Zuplo gibi aracı servisten alınan anahtar `api.typesafe.ai`'de 401 verir**; o durumda `configs/models.yaml` → `jev` → `base_url`'i o servisin adresine çevir |
| `ZOODATA_API_KEY` | ZooData → **Instinct** 27B/4B | <https://zoodata.ai/en/api-keys> | **En az 20 USD bakiye** yüklemek gerekir; istemezsen bu modelleri atla |

Yerel modeller (TF-IDF, embedding, NLI, GLiClass, Laya, SetFit) **anahtar istemez**; ilk çalıştırmada modeli
Hugging Face'ten indirir (birkaç yüz MB–1 GB).

**En ucuz/hızlı başlangıç önerisi:** `INCEPTION_API_KEY` (ücretsiz token) + yerel modeller →
`python -m clfbench run --models mercury_decide,mercury_2_5,tfidf_lr,embedding_e5_lr,nli_mdeberta_xnli`.

---

## Ne karşılaştırılıyor?

[`configs/models.yaml`](configs/models.yaml) dosyasındaki adaylar (anahtarı/paketi olmayan atlanır):

| Grup | Modeller | Nasıl çağrılır |
|---|---|---|
| **karar-api** | Jev 1.13, Mercury Decide, Liquid d1, Instinct (27B / tuned 4B / dual 4B), OpenAI Decisions | Her API'nin kendi `choice` şeması: seçenek → olasılık dağılımı. (Jev/Mercury/d1/Instinct ortak "System One" protokolü; OpenAI Decisions ayrı şema.) |
| **uretken-llm** | Claude Haiku 5.5 / Sonnet 5.5, Gemini 3.5 Flash-Lite, Mercury 2.5, GPT-6 Luna, DeepSeek V4 Flash, Qwen3.8 Flash | Aynı Türkçe talimat + JSON şeması; Claude'da resmî SDK + yapılandırılmış çıktı |
| **encoder** | Laya multilingual, GLiClass, mDeBERTa çok dilli NLI, ModernBERT-NLI (İngilizce; negatif kontrol), e5-small sıfır örnek | Etiketlerin kısa Türkçe adlarıyla, yerelde |
| **egitimli** | TF-IDF + LR, e5-small + LR, SetFit (8-shot) | `train` bölümüyle eğitilir (etiketli veri bütçesi sıfır örnekli modellerden farklıdır, raporda ayrı gruptur) |

`enabled: false` olanlar (Claude Opus 5.5, Gemini 3.8 Flash, Qwen 27B, **Sage**, **Microsoft-Decision-1**) yalnızca
`--models` ile açıkça istenirse çalışır. Sage ve Microsoft-Decision-1 için kamuya açık istek şeması/uç nokta
doğrulanamadığından şablon olarak bırakıldı (`base_url` boş; docs'tan doldurman gerekir).

Fiyatlar 10 Ekim 2026 itibarıyla resmî dokümanlardan ve OpenRouter model listesinden alındı; değişebilir.
Raporun maliyeti = **yanıttan okunan gerçek token kullanımı × `models.yaml` tarifesi** (API token vermezse tahmin edilir ve `~` ile işaretlenir).

---

## Etkileşimli panel (`report.html`)

Her koşu `results/<koşu>/report.html` olarak **tek dosyalık, çevrimdışı çalışan etkileşimli bir panel** üretir (çift tıkla tarayıcıda aç):

- **Üstte özet kartları:** en doğru, en hızlı, en ucuz model; kaç model puanlandı/atlandı.
- **Sıralama grafiği:** ölçüt seçilebilir (doğruluk, macro-F1, ECE, gecikme, maliyet); %95 güven aralığı çizgisiyle. Çubuğa tıkla → model detayı.
- **Hız–maliyet dağılım grafiği:** doğruluk ↔ gecikme veya maliyet, Pareto sınırı, üzerine gelince ayrıntı.
- **Sıralanabilir tablo + filtreler:** grup (Karar API / Üretken LLM / Encoder / Eğitimli) ve arama; satıra tıkla → detay; iki modeli işaretle → **yan yana karşılaştırma** ve hangi örnekte biri doğru diğeri yanlış.
- **Model detayı (yan panel):** tüm ölçüler, sınıf bazında P/R/F1, **etkileşimli karışıklık matrisi** (hücreye tıkla → o hatalar listelenir), gecikme dağılımı, güven kalibrasyonu diyagramı, zor örnek dilimleri, seçenek sırası dayanıklılığı, **hatalı tahminlerin tam listesi** (arama/filtre), teknik notlar. `←` `→` ile modeller arasında gez, `Esc` ile kapat.
- **Atlanan modeller:** neden çalışmadığı (401 anahtar reddedildi, 402 bakiye yok, paket eksik...) ve çözümü.
- Koyu/açık tema, CSV dışa aktarma, klavye erişimi, mobil uyum. Bir modele doğrudan bağlantı: `report.html#m=<model_anahtarı>`.

Aynı sonuçların Markdown sürümü (`report.md`) ve PNG grafikleri de üretilir. Örnek: [examples/jev-40](examples/jev-40/report.html).

## Rapor neleri içerir?

- **Doğruluk** (geçersiz yanıt = yanlış; geçersiz oranı ayrıca), **macro-F1**, sınıf başına P/R/F1, karışıklık matrisi
- **%95 bootstrap güven aralığı** ve **eşlenmiş fark testi** (referans modele karşı; "ayırt edilemez" mi?)
- **Kalibrasyon:** Brier, ECE, güven türü (yerel olasılık / sözel güven / türetilmiş), kapsama–doğruluk eğrisi özetleri
- **Hız:** uçtan uca p50 / p95 / p99, ilk (soğuk) istek, kurulum/eğitim süresi, eşzamanlılık
- **Maliyet:** 1.000 karar başına USD; Pareto sınırları (doğruluk–gecikme, doğruluk–maliyet)
- **Dayanıklılık:** seçenek sırası karıştırma (`--robustness`), zor örnek dilimleri (ascii, yazım hatalı, kısa, çok niyetli)
- **Yöntem ve sınırlar** bölümü (neyin neyle kıyaslanabildiği, neyin kıyaslanamadığı)

Araştırma dosyasındaki uyarılar rapora aynen yansır: **tek kazanan ilan edilmez**, küçük farklar "ayırt edilemez" olarak işaretlenir.

---

## Kendi verinle çalıştır (önerilen)

Veriyi `jsonl` veya `csv` ver, yanına `<ad>.labels.json` koy:

```text
benim_veri.jsonl      {"text": "...", "label": "fatura", "split": "train|dev|test", "id": "isteğe bağlı", "tags": ["typo"]}
benim_veri.labels.json  {"instruction": "...", "labels": {"fatura": "açıklama", ...}, "short": {"fatura": "fatura ve ödeme", ...}}
```

```powershell
python -m clfbench run --dataset C:\veri\benim_veri.jsonl --split test --robustness
```

- Etiket açıklamaları (`labels`) karar API'lerinin `criteria` alanına ve LLM istemine gider: **iyi yazılmış açıklama sonucu ciddi etkiler.**
- `short`, encoder/NLI modellerinin gördüğü kısa etiket adlarıdır (isteğe bağlı).
- Makaledeki öneri: insan etiketli **1.000–3.000** Türkçe örnek; train/dev/test ayrı; istemi test setine göre ayarlama.
- Herkese açık veri: `--dataset hf:<ad>[:config]:<metin_sütunu>:<etiket_sütunu>` (`pip install -e ".[hf]"`).
- Yerleşik seti değiştirmek istersen `data/seed/*.txt` dosyalarını düzenleyip `python -m clfbench build-data` çalıştır.

---

## Yeni model eklemek

`configs/models.yaml` içine girdi eklemek yeterli; kod değişmez.

```yaml
- key: yeni_karar_api
  display: Yeni Karar API
  group: karar-api
  provider: systemone          # systemone | openai_decisions | chat_json | claude | gemini | ...
  base_url: https://api.ornek.com
  path: /v1/systemone
  model: model-adi
  api_key_env: ORNEK_API_KEY   # .env'ye ekle
  price: {input: 0.05, output: 0.0}
```

OpenAI uyumlu herhangi bir sohbet modeli için `provider: chat_json` + `base_url` + `model` yeterlidir.
Tamamen yeni bir arayüz için `src/clfbench/providers/` altına `Provider` alt sınıfı yazıp `providers/__init__.py` içine kaydet
(`predict(text, task, order) -> Prediction`).

---

## Yöntem ve sınırlar

- **Aynı görev, doğal arayüz.** Her model kendi doğal arayüzüyle çağrılır; farkın bir kısmı istem/arayüz farkından gelebilir.
- **Güven değerleri aynı ölçekte değil.** Karar API'leri ve yerel modeller olasılık verir; üretken LLM'ler *sözel* güven yazar.
- **Gecikme kontrollü bir donanım yarışı değil.** İstemci tarafı uçtan uca süre; konum, ağ, eşzamanlılık ve sunucu yükünden etkilenir. Yerel modeller bu makinenin CPU/GPU'sunda ölçülür. Isınma isteği hariç tutulur, ilk istek ayrıca raporlanır.
- **Maliyet.** Abonelik, minimum bakiye, yeniden deneme ve altyapı dahil değildir.
- **Claude notu.** Claude 5.x modellerinde zorunlu `tool_choice` ve örnekleme parametreleri (temperature vb.) reddedildiği için JSON-şema çıktısı kullanılır, örnekleme parametresi gönderilmez ve `effort: low` ayarlanır.
- **Eğitimli modeller** yalnızca `train` bölümünü görür; few-shot örnekleri de `train`'den gelir ve değerlendirmeden çıkarılır. `train` hiçbir zaman değerlendirilmez.
- **Başarısız istekler** sessizce atılmaz: sıkı doğrulukta yanlış sayılır, hata oranı ve ilk hata raporda görünür.

---

## Proje yapısı

```text
classifier_llm_arastirmasi.md   araştırma dosyası (projenin dayanağı)
configs/models.yaml             model kayıt defteri: uç nokta, kimlik, fiyat, seçenekler
data/seed/*.txt                 etiket başına örnek metinler  →  data/destek_tr.jsonl (build-data)
src/clfbench/
  providers/                    systemone, openai_decisions, chat_json, claude, gemini, local (tfidf/embedding/nli/gliclass/laya/setfit)
  runner.py                     paralel çalıştırma, önbellek/devam, ısınma, gecikme ölçümü
  metrics.py                    doğruluk, F1, Brier/ECE, seçici tahmin, bootstrap, Pareto
  bench.py, report.py, charts.py  koşu orkestrasyonu, Markdown/HTML rapor, grafikler
  cli.py, keys.py               komutlar ve API anahtar rehberi
tests/                          sahte API sunucusuyla protokol, metrik, veri ve uçtan uca testler
results/                        koşu çıktıları (git'e girmez)
```

Testler (gerçek anahtar gerekmez; protokolleri yerel sahte sunucuyla doğrular):

```powershell
pip install -e ".[dev]"
python -m pytest
```
