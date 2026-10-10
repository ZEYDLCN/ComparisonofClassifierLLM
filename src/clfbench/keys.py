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
    KeyGuide("TYPESAFE_API_KEY", "TypeSafe (Jev)",
             "https://docs.typesafe.ai",
             "Dokümanda anahtar paneli adresi verilmiyor: TypeSafe sitesinden hesap aç veya sales@typesafe.ai ile iletişime geç.",
             "Ücretsiz katman belirtilmemiş; tarife 0,042 USD / 1M giriş tokenı."),
    KeyGuide("ZOODATA_API_KEY", "ZooData (Instinct 27B / 4B)",
             "https://zoodata.ai/en/api-keys",
             "Giriş yap → önce https://zoodata.ai/en/pricing/models üzerinden bakiye yükle (en az 20 USD) → API Keys sayfasında anahtar oluştur.",
             "En az 20 USD ön yükleme zorunlu; para yatırmak istemiyorsan Instinct modellerini atla."),
]

BY_ENV = {g.env: g for g in GUIDES}


def render_guides(only: set[str] | None = None) -> str:
    out = []
    for g in GUIDES:
        if only is not None and g.env not in only:
            continue
        out += [f"● {g.env}  —  {g.service}", f"    Adres : {g.url}", f"    Adımlar: {g.steps}", f"    Maliyet: {g.cost}", ""]
    return "\n".join(out)
