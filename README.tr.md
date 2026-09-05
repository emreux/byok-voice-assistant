# byok-voice-assistant

Windows için masaüstü sesli asistan: **senin** API anahtarınla, **senin** seçtiğin modelle
çalışır ve **senin** dilinde konuşur.

> **v0.1.0 — konuşma döngüsü.** Kısayolu basılı tut, bir şey söyle, modelin cevabını sesli
> duy. Yaptığı şey bu kadar, ama uçtan uca yapıyor: sesin kendi makinende metne çevriliyor,
> dışarı yalnız metin çıkıyor, cevabı Windows'un kendi sesi okuyor.

<!-- TODO(v0.1.0): 20 saniyelik ekran kaydı buraya, her şeyin üstüne. -->

```
    sen  Türkiye'nin başkenti neresi?
asistan  Türkiye'nin başkenti Ankara'dır.   300 girdi, 10 çıktı
● hazır    Konuşmak için Ctrl+Alt+Space basılı tut. Ctrl+C durdurur.
```

## İki şey bu projenin amacı

**Kendi anahtarını getir.** Sağlayıcıyı ve modeli sen seçiyorsun; uygulama hiçbirini koda
gömmüyor. Bugün tek adaptör var — Google Gemini — ama üstündeki hiçbir şey bunu bilmiyor:
ajan döngüsü arkasında hangi servisin olduğunu hiç öğrenmiyor ve tek bir sözleşme testi her
adaptörü aynı sorulardan geçiriyor, dolayısıyla bir sonrakini eklemek döngüde tek satır
değiştirmiyor. OpenAI uyumlu adaptör (OpenAI, OpenRouter, Groq, DeepSeek, Ollama ve
`base_url`'ü olan her uç nokta) v0.2.0'da, Anthropic v0.4.0'da geliyor.

**Kodun hiçbir yerinde dil sabiti yok.** Asistan hangi dilde konuşursan o dilde cevap
veriyor; konuşmanın ortasında dil değiştirirsen o da değiştiriyor. Dile bağlı geri kalan
her şey — konuşma tanıyıcıya verilen ipucu, Windows sesi, programın gösterdiği ve söylediği
her cümle — dil başına tek bir TOML dosyasında. Türkçe ve İngilizce eksiksiz geliyor;
üçüncüsünü eklemek bir şablonu kopyalayıp sağ tarafı çevirmek.

## Henüz yapmadıkları

| | Geldiği sürüm |
|---|---|
| Araçlar — "Spotify'ı aç", "şu sayfayı özetle" | v0.2.0 |
| Yapmadan önce soran izin kapısı | v0.2.0 |
| Cümle cümle konuşma — beklemeyi yaklaşık yarıya indirir | v0.2.0 |
| Uç-nokta tespiti — şimdilik ne zaman sustuğuna tuş karar veriyor | v0.2.0 |
| Web sayfası ve mail okuma | v0.3.0 |
| Notlar, hatırlatıcılar, tepsi ikonu | v0.4.0 |
| Uyandırma kelimesi ve bir pencere | v0.5.0 |

v0.1.0'da veritabanı yok ve söylediğin hiçbir şey diske yazılmıyor.

## Gereksinimler

- Windows 10 veya 11
- Python 3.13 ve [uv](https://docs.astral.sh/uv/)
- Bir mikrofon ve bir hoparlör
- [Google AI Studio](https://aistudio.google.com/apikey) anahtarı — ücretsiz katman yeter
- **GPU gerekmiyor.** Whisper `small` modelini int8 olarak dört CPU çekirdeğinde çalıştırır.

Dilin için kurulu bir Windows sesi, cevabın duyulabilir olmasıyla anlaşılabilir olması
arasındaki fark. Türkçe için *Microsoft Tolga* gerekiyor; Windows onu Ayarlar → Saat ve dil
→ Konuşma altından kuruyor.

## Kurulum

```bash
git clone https://github.com/emreux/byok-voice-assistant.git
cd byok-voice-assistant
uv sync
```

## Ayarlama

```bash
uv run assistant setup
```

Üç soru: asistanın konuşacağı dil, API anahtarın ve hangi modelin cevap vereceği. Anahtar,
hiçbir şey yazılmadan önce sağlayıcıya karşı sınanıyor ve **Windows Kimlik Bilgisi
Yöneticisi'ne** gidiyor — asla bir dosyaya değil. Ayarlar
`%APPDATA%\assistant\config.toml` dosyasına düşüyor; elle düzenlenebilir düz TOML.

## Konuşma

```bash
uv run assistant run
```

Satır `hazır` diyene kadar bekle, sonra **`Ctrl+Alt+Space`'i basılı tut, konuş ve bırak.**
Kısayol nerede olursan ol çalışıyor; terminalin önde olması gerekmiyor. Yaklaşık dört saniye
sonra cevabı duyuyorsun, konuşulanlar da durum satırının üstünden akıp gidiyor. `Ctrl+C`
durduruyor.

Bilmeye değer üç şey:

- **Basılı tutarken konuş.** Tuşu bırakmak kaydı bitiriyor. Saniyenin üçte birinden kısası,
  yanlışlıkla dokunulmuş bir tuş sayılıyor.
- **Sözünü kesmek için tekrar bas.** Asistan hâlâ konuşuyorken tuşa basarsan anında susuyor
  ve dinlemeye geçiyor — yoksa senin mikrofonuna konuşuyor olurdu.
- **Sessizlik cevaplanmıyor.** Konuşma tanıyıcı, boş bir kayda kendinden emin görünen
  kelimeler uyduruyor; o turlar modele gönderilmeden atılıyor.

Hangi mikrofonun dinleneceğini sen söylemedikçe sistem varsayılanı kullanılıyor.
`uv run python scripts/bench_mic.py --list-devices` gördüğü bütün aygıtları listeliyor;
seninkini satırındaki kelimelerle adlandır — `--device "Microphone Array WASAPI"` — kulaklıklı
bir akşam için `assistant run`'a, kalıcı olarak da `config.toml`'daki `[audio]` altına
`input_device` olarak. İndeks değil kelime: indeksler her Bluetooth aygıtı bağlandığında
kayıyor. Ölçüm betiği ile asistan aynı ayarı okuyor; ölçtüğün mikrofon, kullanılacak olan.

## Bir turun bedeli

Geliştirme makinesinde uçtan uca ölçüldü — dört çekirdekli, ayrık ekran kartı olmayan bir
dizüstü — tahmin edilmedi.

| | Türkçe | İngilizce |
|---|---|---|
| Yazıya çevirme, ~2.5 sn konuşma için | 2.8 sn | 2.1 sn |
| Modelin tam cevabı | 1.3 sn | 0.9 sn |
| **Tuşun kalkmasından ilk sese** | **4.1 sn** | **3.0 sn** |
| Tek cümlelik soru için token | 300 girdi / 10 çıktı | 297 girdi / 8 çıktı |

Whisper program açılırken bir kez yükleniyor, yaklaşık altı saniye, böylece ilk basış onu
beklemiyor. v0.2.0'daki cümle cümle konuşma, ilk sese kadar geçen süreyi kabaca yarıya
indiriyor.

## Verilerin nereye gidiyor

- **Sesin makineden çıkmıyor.** Whisper yerelde çalışıyor; sağlayıcıya yalnız metin gidiyor.
- **API anahtarın hiçbir dosyaya yazılmıyor.** `keyring` üzerinden Windows Kimlik Bilgisi
  Yöneticisi'nde duruyor.
- **Söylediğin hiçbir şey diske yazılmıyor.** Log, her turun ne kadara mal olduğunu —
  giren ve çıkan token — yazıyor, ne duyulduğunu ve ne cevaplandığını asla. Konuşma, son on
  iki turdan ibaret; bellekte duruyor ve program kapanınca gidiyor.
- Ayarlar: `%APPDATA%\assistant\config.toml`. Log:
  `%LOCALAPPDATA%\assistant\Logs\assistant.log`.

## Yeni bir dil eklemek

`src/assistant/locales/_template.toml` dosyasını `<kod>.toml` olarak kopyala — `de.toml`,
`es.toml`, `ja.toml` — ve sağ tarafı çevir. Prosedürün tamamı bu; kod değişmiyor. Boş
bıraktığın bir anahtar İngilizce cevaplanıyor, yani yarım kalmış bir paket bozuk değil
kullanılabilir oluyor.

## Yeni bir sağlayıcı eklemek

`src/assistant/defaults/providers.toml` dosyasına bir satır ekle, `LLMProvider` protokolünü
sağlayan bir adaptör yaz, adaptörünün test dosyasına bir `build` fonksiyonu ve
`tests/test_llm_adapters.py` içindeki `ADAPTERS` listesine bir satır koy. Sözleşme testi
bundan sonra diğerlerine sorduğu her soruyu seninkine de soruyor, hiç değişmeden. Projede
başka hiçbir şey değişmiyor — ve sözleşmeden geçirilmemiş bir adaptör kaydedilirse test
kırmızıya dönüyor.

## Geliştirme

```bash
uv run ruff check .            # lint
uv run ruff format .           # biçim
uv run mypy src tests --strict # tip denetimi
uv run pytest                  # testler
```

## Lisans

MIT — bkz. [LICENSE](LICENSE).

---

English: [README.md](README.md)
