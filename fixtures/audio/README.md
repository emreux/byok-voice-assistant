# Ses fikstürleri

Bu klasör konuşma tanımanın (STT) ölçüldüğü kayıtları tutar. **`.wav` ve `.mp3` dosyaları depoya girmez** (`.gitignore`) — kendi sesin kişisel veridir ve depoyu şişirir. Depoya giren yalnız **referans metinlerdir**: her kaydın yanında aynı adla bir `.txt`.

```
fixtures/audio/
  01-selamlama.wav      <- yoksayılır, senin makinende durur
  01-selamlama.txt      <- depoda: kayıtta ne söylendiğinin birebir metni
```

Referans metin, kelime hata oranı (WER) hesabının doğruluk tarafıdır: `scripts/bench_stt.py` (Faz 2.8) her `.wav`'ı çevirir ve yanındaki `.txt` ile karşılaştırır. Metin yoksa kaydın ölçümde bir değeri olmaz.

## Kayıt nasıl alınır

Faz 2.8 geldiğinde 15–20 kısa kayıt gerekir; şimdilik iskelet duruyor. Alırken:

- **16 kHz, mono, 16 bit** — modelin beklediği biçim; başka bir örnekleme hızı yeniden örneklenir ve ölçümü kirletir.
- Gerçek kullanım koşulunda kaydet: aynı mikrofon, aynı mesafe, aynı oda. Sessiz bir stüdyo kaydı iyimser bir sayı üretir, o sayıya güvenip yanılırsın.
- Çeşitlilik olsun: kısa komut ("dur", "saat kaç"), uzun cümle, uygulama adı geçen cümle ("Spotify'ı aç"), sayı ve tarih içeren cümle.
- Referans metni **söylediğin gibi** yaz — düzelterek değil. "yani şey" dediysen o da metne girer.

`01-`, `02-` gibi bir sıra öneki dosyaları bir arada tutar; ad İngilizce olmak zorunda değil, çünkü bu bir *değer* dosyasıdır, tanımlayıcı değil (§7.1).
