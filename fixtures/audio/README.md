# Ses fikstürleri

Bu klasör konuşma tanımanın (STT) ölçüldüğü kayıtları tutar. **`.wav` ve `.mp3` dosyaları depoya girmez** (`.gitignore`) — kendi sesin kişisel veridir ve depoyu şişirir. Depoya giren yalnız **referans metinlerdir**: her kaydın yanında aynı adla bir `.txt`.

```
fixtures/audio/
  01-selamlama.wav      <- yoksayılır, senin makinende durur
  01-selamlama.txt      <- depoda: kayıtta ne söylendiğinin birebir metni
```

Referans metin, kelime hata oranı (WER) hesabının doğruluk tarafıdır: `scripts/bench_stt.py` her `.wav`'ı çevirir ve yanındaki `.txt` ile karşılaştırır. Metin yoksa kayıt ölçüme girmez ve betik atladığını söyler.

## Ne ölçülüyor

```bash
uv run python scripts/bench_stt.py                    # tiny, base, small, medium: süre ve WER
uv run python scripts/bench_stt.py --sizes small      # yalnız uygulamanın kullandığı boyut
uv run python scripts/bench_stt.py --provider gemini  # Google'ın tanıyıcısı, gerçek anahtarla (ses Google'a gider)
uv run python scripts/bench_e2e.py                    # kayıt → Whisper → model → ses: ilk sese kadar süre
```

`bench_stt.py` model boyutlarını süre (p50, p95) ve hata oranıyla (WER, CER) yan yana koyar; §8'deki karar kapısını (`small` p95 > 1.2 sn veya WER > %15) kendisi söyler. `bench_e2e.py` aynı kayıtları uygulamanın gerçek parçalarından geçirir — gerçek Whisper, gerçek sağlayıcı (anahtar Kimlik Bilgisi Yöneticisi'nden), gerçek kapı ve saat aracı, gerçek Windows sesi — ve her tur için üç süre yazar: yazıya dökme bitti, ilk ses, tur bitti. Ses çalınmaz; hiçbir uygulama açılmaz; her koşu token harcar.

## Kayıt nasıl alınır

```bash
uv run python scripts/bench_mic.py --takes              # fixtures/audio içine, config'deki mikrofonla
uv run python scripts/bench_mic.py --takes --seconds 4  # kısa cümleler için
```

Her kayıt için önce cümleyi **söyleyeceğin gibi** yazarsın, Enter'a basıp söylersin; betik
`.txt`'yi yazdığından üretir, kaydı dedektörün son konuşma duyduğu yerde keser ve
`NN-ilk-kelimeler.wav` adıyla yazar. Boş satır bitirir. Mikrofon `assistant run`'ın açtığı
mikrofondur (`config.toml [audio] input_device`) — Windows'un efekt yolundan alınan bir
kayıt asistanın hiç duymadığı bir sinyali ölçer.

15–20 kısa kayıt yeter. Alırken:

- **16 kHz, mono, 16 bit** — modelin beklediği biçim; başka bir örnekleme hızı girişte yeniden örneklenir ama ölçümü kirletir.
- Gerçek kullanım koşulunda kaydet: aynı mikrofon, aynı mesafe, aynı oda. Sessiz bir stüdyo kaydı iyimser bir sayı üretir, o sayıya güvenip yanılırsın. Sentezlenmiş ses (Windows'un kendi sesi) daha da iyimserdir: 10 Eyl 2026'daki ilk tablo öyle alındı ve ADR-001'de "taban, bütçe değil" diye işaretli.
- Çeşitlilik olsun: kısa komut ("dur", "saat kaç"), uzun cümle, uygulama adı geçen cümle ("Spotify'ı aç"), sayı ve tarih içeren cümle.
- Referans metni **söylediğin gibi** yaz — düzelterek değil. "yani şey" dediysen o da metne girer.

`01-`, `02-` gibi bir sıra öneki dosyaları bir arada tutar; ad İngilizce olmak zorunda değil, çünkü bu bir *değer* dosyasıdır, tanımlayıcı değil (§7.1).
