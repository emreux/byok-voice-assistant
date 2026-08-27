# byok-voice-assistant

Windows için masaüstü sesli asistan: **senin** API anahtarınla, **senin** seçtiğin modelle çalışır ve **senin** dilinde konuşur.

> **Durum: Faz 0 — yalnızca tasarım.** Depoda henüz çalışan kod yok.
> Kurulum ve saglayici katmani hazir; `setup` ve `run` henuz bitmedi.

<!-- TODO(phase-1): 30 saniyelik demo videosu buraya, her şeyin üstüne. -->

## Neden bir sesli asistan daha

Açık kaynak sesli asistanların çoğu seni tek bir sağlayıcıya ve tek bir dile bağlar. Bu ikisini de yapmıyor.

- **Kendi anahtarını getir.** Üç adaptör 15+ sağlayıcıyı kapsıyor — Anthropic, OpenAI, Google Gemini, OpenRouter, Groq, DeepSeek, xAI, Mistral, Together, Ollama ve OpenAI uyumlu her uç nokta. Sağlayıcıyı ve modeli ilk açılış sihirbazında sen seçiyorsun; uygulama hiçbirini koda gömmüyor.
- **Senin dilinde konuşur.** Hangi dilde konuşursan o dilde cevap verir; konuşmanın ortasında dil değiştirirsen o da değiştirir. Arayüz dili kodda sabit değil, bir yapılandırma değeri — yeni dil eklemek tek bir TOML dosyası.
- **Önce yerel.** Yerel Whisper ile sesin makineden çıkmaz; buluta yalnızca metin gider. `assistant doctor` hangi verinin hangi sağlayıcıya gittiğini tek tek söyler.
- **Yapmadan önce sorar.** Her araç çağrısı, varsayılanı "onay iste" olan tek bir izin kapısından geçer. Mail gönderme, dosya silme ve takvim değiştirme sesli okunur ve senden "evet" bekler.

## Mimari

<!-- TODO(phase-1): mimari şeması (SVG) buraya. -->

Ses yerelde metne cevrilir, metin secilen modele gider, cevap sesli okunur. Saglayiciya
ozgu kod tek bir adaptorde kalir; ajan dongusu arkasinda hangi servisin oldugunu bilmez.

## Gecikme

Geliştirme makinesinde uçtan uca ölçülür, tahmin edilmez.

<!-- TODO(phase-0.3 onward): her faz sonunda scripts/bench_e2e.py çıktısıyla doldur. -->

| Tur tipi | p50 | p95 |
|---|---|---|
| *henüz ölçülmedi* | — | — |

## Gereksinimler

- Windows 10 veya 11
- Python 3.13+
- Desteklenen en az bir sağlayıcı için API anahtarı

<!-- TODO(phase-0.4): önerilen mikrofon, scripts/bench_mic.py kararıyla. -->

## Gizlilik

Notlar, konuşma geçmişi ve araç sonuçları `%LOCALAPPDATA%\assistant\` altında **düz metin** SQLite olarak durur; yalnızca senin Windows hesabın okuyabilir. Tam disk şifreleme (BitLocker) senin sorumluluğunda. `assistant purge --all` veritabanını ve saklanan tüm kimlik bilgilerini siler.

API anahtarları `keyring` üzerinden Windows Credential Manager'a gider, diske düz metin olarak asla yazılmaz.

## Kurulum

<!-- TODO(phase-1): paket derlenebilir olunca kurulum adımları. -->

Henüz kurulabilir değil.

## Lisans

MIT — bkz. [LICENSE](LICENSE).

---

English: [README.md](README.md)
