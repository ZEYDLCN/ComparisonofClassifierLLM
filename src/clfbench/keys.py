"""Her API anahtarının nereden ve nasıl alınacağı (clfbench keys komutu ve README için)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyGuide:
    env: str
    service: str
    url: str
    steps: str
    cost: str  # ücretsiz kredi / minimum bakiye bilgisi


GUIDES: list[KeyGuide] = [
    KeyGuide("ANTHROPIC_API_KEY", "Anthropic (Claude Haiku/Sonnet/Opus 5.5)",
             "https://platform.claude.com/settings/keys",
             "Giriş yap → Settings → API Keys → Create Key; anahtarı hemen kopyala (bir daha gösterilmez). "
             "Kullanım için Billing'den kredi yüklemek gerekir.",
             "Ön ödemeli kredi; bu veri setinde 150 istek birkaç sentten azdır."),
    KeyGuide("OPENAI_API_KEY", "OpenAI (Decisions API, gpt-6-luna)",
             "https://platform.openai.com/api-keys",
             "Giriş yap → API keys → Create new secret key. Decisions API public beta'dır; hesabında erişim yoksa 403/404 alırsın.",
             "Ön ödemeli kredi."),
    KeyGuide("GEMINI_API_KEY", "Google AI Studio (Gemini 3.5 Flash-Lite)",
             "https://aistudio.google.com/apikey",
             "Google hesabıyla giriş → Get API key → Create API key.",
             "Ücretsiz katman olabilir (sınırlı hız); ücretli katman için faturalandırma ekle."),
    KeyGuide("OPENROUTER_API_KEY", "OpenRouter (GPT-6 Luna sohbet, DeepSeek, Qwen)",
             "https://openrouter.ai/keys",
             "Giriş yap → Keys → Create Key. Tek anahtarla birçok sağlayıcıya erişirsin; Credits'ten bakiye yükle.",
             "Ön ödemeli kredi."),
    KeyGuide("INCEPTION_API_KEY", "Inception (Mercury Decide + Mercury 2.5)",
             "https://platform.inceptionlabs.ai/dashboard/api-keys",
             "https://platform.inceptionlabs.ai adresinden kaydol → Dashboard → API Keys → yeni anahtar oluştur.",
             "Yeni kullanıcıya 100 milyon ücretsiz token verildiği belirtiliyor, kart gerekmiyor."),
    KeyGuide("LIQUID_API_KEY", "Liquid AI (d1)",
             "https://console.liquid.ai",
             "Kaydol → bir organizasyona katıl → Dashboard → API Keys → yeni anahtar (liquid_ ile başlar).",
             "d1:free metin için ücretsiz katman sunuyor olabilir; model adını configs/models.yaml'da 'd1:free' yapabilirsin."),
    KeyGuide("JEV_AI_API_KEY", "Jev AI (jev-ai.pro) — Jev",
             "https://jev-ai.pro",
             "Sağlayıcı panelinden anahtar oluştur (sk_jev-ai_ ile başlar). Uç nokta: https://jev-ai.pro/api/v1/systemone; istekte model alanı yok. "
             "Bakiye yoksa API 402 döner; yükleme adresi: https://jev-ai.pro/pricing?context=api&mode=top-up",
             "Çağrı için token/kredi bakiyesi gerekir (bakiye yokken ücret kesilmez). Tarife bilinmiyor: models.yaml'da jev_ai fiyatını girersen rapor maliyeti hesaplar."),
    KeyGuide("BEATAPI_API_KEY", "BeatAPI (Jev ücretsiz: jev-1.13-free)",
             "https://beatapi.io/dashboard/apikeys",
             "beatapi.io'da hesap aç → Dashboard → API keys → anahtar oluştur. Uç nokta: https://api.beatapi.io/v1/systemone. "
             "Anahtarı sunucu tarafında (.env) tut. jev-ai.pro anahtarın buraya UYMAZ: her servisin anahtarı ayrıdır.",
             "Sıfır bakiyede ücretsiz çalışır ama hesap başına dakikada 1 başarılı istek (gerçek yanıtla doğrulandı). Tam test ~2,5 saat; --limit 40 kullan veya bakiye yükle."),
    KeyGuide("TYPESAFE_API_KEY", "TypeSafe (Jev)",
             "https://docs.typesafe.ai",
             "TypeSafe konsolunda API Keys sayfasından anahtar oluştur (docs.typesafe.ai). Üçüncü taraf kaynaklara göre yeni kayıtlar 22 Eylül 2026'da duraklatıldı. "
             "Anahtarı Vercel AI Gateway / Zuplo gibi bir aracıdan aldıysan api.typesafe.ai 401 verir: configs/models.yaml içinde jev girdisinin base_url'ini o servise çevir.",
             "Ücretsiz katman belirtilmemiş; tarife 0,042 USD / 1M giriş tokenı."),
    KeyGuide("ZOODATA_API_KEY", "ZooData (Instinct 27B / 4B)",
             "https://zoodata.ai/en/api-keys",
             "Giriş yap → önce https://zoodata.ai/en/pricing/models üzerinden bakiye yükle (en az 20 USD) → API Keys sayfasında anahtar oluştur (anahtar hms_ ile başlar).",
             "En az 20 USD ön yükleme zorunlu; para yatırmak istemiyorsan Instinct modellerini atla."),
]

BY_ENV = {g.env: g for g in GUIDES}

# Beklenen anahtar önekleri. hms_ (ZooData hata mesajı) ve liquid_ (Liquid dokümanı) doğrulandı; diğerleri yaygın biçimdir.
# Yalnızca bilgilendirici uyarı: doctor yanlış yapıştırmayı yakalamak için kullanır, çalışmayı engellemez.
PREFIX_HINT = {
    "ANTHROPIC_API_KEY": "sk-ant-",
    "OPENROUTER_API_KEY": "sk-or-",
    "OPENAI_API_KEY": "sk-",
    "LIQUID_API_KEY": "liquid_",
    "ZOODATA_API_KEY": "hms_",
    "GEMINI_API_KEY": ("AIza", "AQ."),  # AQ. önekli anahtar gerçek Gemini isteğiyle doğrulandı
    "JEV_AI_API_KEY": "sk_jev-ai_",
}


def prefix_warning(env: str, value: str) -> str | None:
    want = PREFIX_HINT.get(env)
    if want and value and not value.startswith(want):
        shown = " veya ".join(f"'{w}'" for w in ((want,) if isinstance(want, str) else want))
        return f"{env} {shown} ile başlamalı (yanlış anahtar yapıştırılmış olabilir)"
    return None


def render_guides(only: set[str] | None = None) -> str:
    out = []
    for g in GUIDES:
        if only is not None and g.env not in only:
            continue
        out += [f"● {g.env}  —  {g.service}", f"    Adres : {g.url}", f"    Adımlar: {g.steps}", f"    Maliyet: {g.cost}", ""]
    return "\n".join(out)


# Eksik Python modülü -> pip ek paketi (pyproject.toml [project.optional-dependencies])
PKG_EXTRA = {"torch": "local", "transformers": "local", "sentence_transformers": "local", "sklearn": "local",
             "gliclass": "gliclass", "laya": "laya", "setfit": "setfit", "datasets": "setfit", "anthropic": "anthropic"}


def fix_for(spec, reason: str) -> dict:
    """Çalıştırılmayan model için 'nasıl açılır' bilgisi: {short, steps, url, cost, cmd}.

    short: raporda tek satır; steps: panelde ayrıntılı adımlar."""
    cmd = f"python -m clfbench run --models {spec.key}"
    if "tanımlı değil" in reason and spec.api_key_env:
        g = BY_ENV.get(spec.api_key_env)
        url = g.url if g else ""
        return {"short": f"`.env` içine `{spec.api_key_env}=...` ekle" + (f" · anahtar: {url}" if url else ""),
                "steps": f".env dosyasına {spec.api_key_env}=... satırını ekle. " + (g.steps if g else ""),
                "url": url, "cost": g.cost if g else "", "cmd": cmd}
    if "paket eksik" in reason:
        extras = sorted({PKG_EXTRA.get(m.split(".")[0], "all") for m in spec.missing_requirements()})
        pip = f'pip install -e ".[{",".join(extras)}]"'
        return {"short": f"`{pip}`", "steps": f"{pip}  (hepsi için: pip install -e \".[all]\")", "url": "",
                "cost": "Paketler ücretsizdir.", "cmd": cmd}
    if "base_url" in reason:
        msg = "configs/models.yaml içinde bu girdinin base_url alanını sağlayıcı belgesinden doldur."
        return {"short": msg, "steps": msg, "url": "", "cost": "", "cmd": cmd}
    if "kapalı" in reason:
        extra = spec.verified if "DOĞRULANMADI" in spec.verified else ""
        url = BY_ENV[spec.api_key_env].url if spec.api_key_env in BY_ENV else ""
        return {"short": f"`--models {spec.key}` ile açıkça iste" + (" (şablon: uç nokta doğrulanmadı)" if extra else ""),
                "steps": "Girdi varsayılan kapalıdır; --models ile açıkça istenince çalışır. " + extra,
                "url": url, "cost": "", "cmd": cmd}
    return {"short": "", "steps": "", "url": "", "cost": "", "cmd": cmd}
