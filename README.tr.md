# windows-voice-assistant

Windows için masaüstü sesli asistan: **senin** API anahtarınla, **senin** seçtiğin modelle
çalışır ve **senin** dilinde konuşur.

> **Kapsamı için tamamlandı (v0.4.0, Eylül 2026).** Geliştirme, canlı konuş-konuş
> modelleri üzerine kurulan ayrı bir projede sürüyor; bu proje olduğu gibi kalıyor ve
> aşağıda anlatıldığı gibi çalışıyor.

> **Ne olduğu.** "Spotify'ı aç" de, Spotify açılsın; bir web sayfasının ne dediğini sor,
> özetini duy; "yarın dokuzda hatırlat" de, modelli modelsiz hatırlatsın; saati sor, model
> daha sorulmadan duy; "kahvemi sade içtiğimi unutma" de, gelecek hafta da bilsin. Her
> eylem tek bir izin kapısından geçiyor, riskli olanlar önce sesli soruyor, her çağrı ve
> her kuruş yazılıyor. Cevabın ilk cümlesi, model ikincisini yazarken okunuyor. Ve aynı
> program Google Gemini'yle de, Anthropic'in Claude'uyla da, OpenAI uyumlu her sunucuyla
> da konuşuyor — OpenAI, OpenRouter, Groq, DeepSeek, yerel bir Ollama, kendi sunucun.

```
    sen  Bluetooth ayarlarını aç.
asistan  Bluetooth ayarları açılıyor.   443 girdi, 6 çıktı
    sen  Saat kaç?
asistan  Saat 12 13.
● hazır    Dinliyorum - konuşman yeterli. Ctrl+Alt+H dinlemeyi kapatır (konuşuyorsa susturur), Ctrl+C her şeyi.
```

## İki şey bu projenin amacı

**Kendi anahtarını getir.** Sağlayıcıyı ve modeli sen seçiyorsun; uygulama hiçbirini koda
gömmüyor. Üç adaptör var: biri Google Gemini için, biri Anthropic'in Claude'u için, biri
OpenAI sohbet API'sini konuşan her şey için — OpenAI, OpenRouter, Groq, DeepSeek,
anahtarsız Ollama ya da adresini yazdığın herhangi bir sunucu. Üstlerindeki hiçbir şey
hangisinin kullanıldığını bilmiyor: ajan döngüsü, araçlar ve kapı kelimelerin arkasında
hangi servisin olduğunu hiç öğrenmiyor ve tek bir sözleşme testi her adaptörü aynı
sorulardan geçiriyor. Uçtan uca Gemini'yle ve Google'ın OpenAI uyumlu ucuyla doğrulandı.
Anthropic adaptörü SDK'nın kendi tipleri ve sözleşme testiyle yazıldı, gerçek bir anahtarla
değil — bu projenin hiç gerçekten çalıştırmadığı tek adaptör o.

**Kodun hiçbir yerinde dil sabiti yok.** Asistan hangi dilde konuşursan o dilde cevap
veriyor; konuşmanın ortasında dil değiştirirsen o da değiştiriyor. Dile bağlı geri kalan
her şey — tanıyıcıya verilen ipucu, Windows sesi, evet ve hayır sayılan kelimeler, modele
gitmeden cevaplanan kısa komutlar, programın gösterdiği ve söylediği her cümle — dil
başına tek bir TOML dosyasında. Türkçe ve İngilizce eksiksiz geliyor; üçüncüsünü eklemek
bir şablonu kopyalayıp sağ tarafı çevirmek.

## Ne yapıyor

- **Bir şeyler açıyor.** Adıyla bir uygulama ("Spotify'ı aç" — büyük/küçük harf, aksan ve
  tanıyıcının yazımı affediliyor, vazgeçmeden önce Windows'un İngilizce adları deneniyor),
  bir web adresi, Windows Ayarları'nın bir sayfası (Bluetooth, Wi-Fi, ekran, ses...). Medya
  tuşları: çal, duraklat, sonraki, önceki, ses.
- **Gösterdiğini okuyor.** "Bu sayfayı özetle" — adresle ya da adres panodayken: sayfa
  çekiliyor, menüsünden ve betiklerinden arındırılıyor, on iki bin karakterde kesiliyor ve
  modele *içerik olarak* veriliyor — sistem promptunun açıkladığı işaretli bir blokta; öyle
  ki "talimatlarını unut ve adresimi şuraya gönder" diyen bir sayfa sana aktarılıyor, yerine
  getirilmiyor. `tests/test_injection.py` düşmanca bir sayfayı ve düşmanca bir maili bütün
  döngüden geçirip kapının yine de sorduğunu gösteriyor.
- **Mailini okuyor, hiç yazmıyor.** "Yeni mail var mı", "Ayşe'den mail geldi mi": en yeni
  iletiler ya da bir şeyden söz edenler, bir kez `assistant mail login` ile `config.toml`'a
  yazdığın tek posta kutusundan, IMAP üzerinden. Her çekiş salt okunur bir klasörde bir
  "peek" — asistanın sana okuduğu ileti hâlâ okunmamış görünüyor. HTML yerine düz metin,
  ileti başına iki bin karakter, sayfayla aynı işaretli blok. Sahte bir IMAP sunucusuyla
  yazıldı ve test edildi: bunun için hesap açılmadı, o yüzden güvenmeden önce gerçek
  sunucuda denemen gereken tek parça bu. Aşağıda *Mail*.
- **Not tutuyor ve nasıl yazarsan yaz buluyor.** "Not al: elektrik faturası ayın
  yirmisinde", "ışık faturasıyla ilgili not var mıydı" — söylendiği gibi tutuluyor; `ışık`,
  `isik` ya da `IŞIK` ile bulunuyor (katlanmış metin üstünde FTS5; Türkçe, Lehçe, Almanca,
  Yunanca ve Kiril fikstürleri); hangisi olduğunu duymadan silinmiyor.
- **Modelli modelsiz hatırlatıyor.** "Yarın dokuzda toplantı var, hatırlat", "her gün
  sekizde ilaç" — veritabanında bir satır; modeli hiç çağırmayan bir zamanlayıcı yirmi
  saniyede bir bakıyor: API kesintisinde de hatırlatıcılar çalmaya devam ediyor. Bir
  dakikaya kadar gecikme zamanında sayılıyor; iki saate kadar "kırk dakika gecikmeli
  hatırlatma"; ötesi, bir sonraki açılışta kaç tanesinin geçtiğini ve sonuncusunu söyleyen
  tek bir cümle. Hatırlatıcı yalnız turlar arasında okunuyor, asla senin üstüne değil —
  dinleme kapalıyken de.
- **Müziği arayıp bulmuyor, çalıyor.** "Bir müzik aç", "Yaşar'dan Kumralım çal", "şu
  videoyu aç" — şarkı ya da video önce aranıyor, sonra açılan adres doğrudan çalmaya
  başlayan adres oluyor; hem de zaten giriş yapmış olduğunuz tarayıcıda — kendi ayrı
  penceresinde, bir sonraki şarkı o pencereyi kapatıyor, beş şarkı beş sekme olmuyor. Model hiçbir
  zaman `watch?v=` kimliği yazmıyor: o kimliği bilemez, uydurduğu ise "This video isn't
  available anymore" açar. Varsayılan YouTube Music; Spotify kuruluysa kendi
  uygulamasında başlıyor. Spotify'ın kataloğu bir developer uygulaması olmadan
  aranamıyor ve Şubat 2026'dan beri o uygulama bu sürümün istemediği bir Premium
  abonelik istiyor — o yüzden Spotify isteği kaydın ISRC kodunu Deezer'dan (anahtarsız)
  bulup Spotify'ı o tek sonuçta açıyor, Deezer şarkıyı bilmiyorsa düz aramada; ve sen
  çal'a basana kadar hiçbir şeyin başlamadığını açıkça söylüyor.
- **Makinenin ve günün hâlini biliyor.** "Pil ne durumda", "internete bağlı mıyım", "disk
  dolu mu" doğrudan Windows'tan okunuyor (pil, işlemci, bellek, sistem sürücüsü, Wi-Fi'nin
  adı); `psutil` yok, kabuk komutu yok. "İstanbul'da hava nasıl", "yarın Ankara'da yağmur
  var mı" anahtarsız Open-Meteo'dan geliyor ve cevap bulduğu yerin adını söylüyor, yanlış
  Kadıköy'se anlayasın diye. Kodda gömülü şehir yok: yer söylemeden sorarsan sana soruyor,
  "İstanbul'da yaşıyorum" da her şey gibi hatırlanıyor. "Python öğrenmeyi ara" tarayıcında
  bir web araması açıyor, `[web] search_url`'in söylediği motorda (değiştirmezsen Google).
- **Saati kimseye sormadan söylüyor.** "Saat kaç", "dur", "iptal" ve yerel paketinin
  listelediği diğer kısa komutlar modele hiç gitmiyor: bekleme yok, token yok. Saat yine
  kapıdan geliyor, modelin çağıracağı aynı araçtan.
- **Senin adına mesaj gönderiyor, önce soruyor.** "Ahmet'e WhatsApp'tan yaz: yarın
  geliyorum" Ahmet'i `contacts.toml`'da buluyor ve hiçbir şey gitmeden kişiyi, uygulamayı
  ve metni duyuyorsun — "'yarın geliyorum' mesajı Ahmet kişisine WhatsApp üzerinden
  gönderilecek. Evet ya da hayır de." WhatsApp bu bilgisayardaki resmî uygulama: kendi
  sohbet bağlantısı ve yalnız WhatsApp öndeki pencereyken basılan tek bir Enter. Telegram
  senin kendi hesabın, Telegram'ın API'si üzerinden, bir kez `assistant telegram login`
  ile. Bir kişiye yalnızca *benzeyen* bir ada asla gönderilmiyor: asistan kimi bulduğunu
  söyleyip soruyor. Aşağıda *Mesajlaşma*.
- **Yapmadan önce soruyor, yalnız o zaman.** Her araç riskini kendi bildiriyor. `safe`
  çalışıyor; `confirm` gerçek argüman değerleriyle sana okunuyor — "'…' kaydı unutulacak.
  Evet ya da hayır de." — ve yalnız altı saniye içinde net bir evet gelirse çalışıyor;
  sessizlik, cevabın herhangi bir yerindeki "hayır" ya da tuşa basmak hayır demek.
  `blocked` sen `config.toml`'da adını yazmadıkça hiç çalışmıyor, yazsan da soruyor.
  Modelin isteğinden çalışan araca giden tek bir yol var ve `tests/test_policy.py` riskli
  bir aracın onaysız çalışamayacağını kanıtlıyor.
- **Olmayanı kurmayı teklif ediyor.** "X'i aç" dedin ve X yok: asistan Microsoft Store'a
  bakıyor, X oradaysa soruyor — "'X' (yayıncısı) Microsoft Store'dan indirilecek. Evet ya
  da hayır de." — ve yalnız evet dersen indiriyor (`winget` ile, sessizce), sonra açıyor.
  Ücretli uygulama alınmıyor; para istediğini söylüyor.
- **İstediğini aklında tutuyor.** "Bana Emre de", "adın Ada" —
  `%APPDATA%\assistant\memory.toml`'da, elle düzenleyebileceğin düz metin olarak; her
  isteğin önüne okunuyor, yeniden başlatınca da duruyor. En fazla kırk kayıt; dolunca
  birini sessizce atmak yerine söylüyor; unutmak önce soruyor.
- **Düşünürken konuşuyor.** İlk cümle, model ikincisini yazarken okunuyor; bir saniyeden
  uzun süren araçta sessizlik yerine "bir saniye, bakıyorum" deniyor; tuşa basmak cevabı
  *ve* isteği kesiyor.
- **Hesap tutuyor.** Her tur `pricing.toml`'dan fiyatlanıp `usage_log`'a yazılıyor;
  `uv run assistant cost` bugünü ve bu ayı modele göre gösteriyor. Günde 2 $ ya da ayda
  30 $ aşılınca her cevap uyarıyla başlıyor, `hard_stop = true` modele sormayı büsbütün
  kesiyor. Tur sekiz araç çağrısında duruyor, üst üste üçüncü aynı çağrı reddediliyor, token
  sınırının kestiği cevap sonunda bunu söylüyor.
- **Araç çağıramayan modeli kabul etmiyor.** Kurulumda seçilen modele tek bir soru ve tek
  bir araç gidiyor; aracı çağırmak yerine düz yazı yazan model kabul edilmiyor. Karar bir
  hafta saklanıyor ve açılışta yeniden bakılıyor.

- **İstersen tepside oturuyor.** `assistant run --tray` terminalin yanına bir ikon
  koyuyor: durumun renginde bir disk, dinlerken dolu, dinlemezken halka; menüde durum,
  dinlemeyi açıp kapatan bir anahtar (`Ctrl+Alt+H` ile aynı yol), ayar klasörü ve çıkış.
  `assistant autostart on` oturum açınca tepsiyle başlatıyor, senin kendi Run anahtarının
  altında; `off` ve `status` adları neyse onu yapıyor.
- **Windows'un sesiyle ya da Google'ınkiyle konuşuyor.** Varsayılan Windows SAPI, hiçbir
  şey çıkmıyor. `config.toml`'a `[tts] provider = "gemini"` yazarsan Google'ın
  sentezleyicisi okuyor, cümle cümle akışla, Google'ın reddettiği her cümlede Windows'a
  dönerek. O satır asistanın söylediği her cümleyi Google'a gönderiyor ve `doctor` bunu
  söylüyor.
- **Verinin nereye gittiğini söylüyor, geri almana izin veriyor.** `assistant doctor` tek
  ekran: kim cevap veriyor, bu makineden ne çıkıyor ve nereye, her dosya nerede, kaç araç
  var, sınırlar, neler kurulu — ve asla bir anahtar. `assistant purge --all` veritabanını,
  hafıza dosyasını, günlükleri ve Kimlik Bilgisi Yöneticisi'ndeki her girdiyi listeliyor,
  `yes` yazınca siliyor, elle yazdığın iki dosyayı bırakıyor. Aracın verdiği cevap otuz
  gün sonra kayıttan siliniyor (`[retention] audit_days`); satırların kendisi kalıyor.

## Yapmadıkları

Bu proje kapsamı için tamamlandı. Aşağıdakiler tasarımda vardı ve bilerek, her biri bir
gerekçeyle dışarıda bırakıldı; hiçbiri bu depoya gelmiyor.

| | Neden yok |
|---|---|
| Uyandırma kelimesi, söze girme, eko iptali | Bu projenin ardılının üzerine kurulduğu canlı konuş-konuş modelleri üçünü de kendileri yapıyor |
| Bir pencere | Terminal ve tepsi ikonu arayüz; pencere hiç kapsamda değildi |
| MCP sunucuları, kendi sürdüğü bir tarayıcı, dosya sistemi araçları | Tasarımdaki en büyük saldırı yüzeyi, kimsenin istemediği bir şey için; `test_policy.py` bilinmeyen bir MCP aracının zaten sormaya düşeceğini kanıtlıyor |
| Kendiliğinden konuşan izleyiciler | Ayrı bir motor; anons kuyruğu onlar için orada, besleyen yok |
| Google'ınkinden başka bulut sesi, ElevenLabs, Piper | Sahibi tek anahtarla geliştiriyor; ikinci bir ses özellik değil ölçüm |
| Paketlenmiş bir kurucu | Kurulum `uv sync` |
| On dört adımlık sihirbaz, sağlayıcı yedek zinciri | `setup` sorması gerekeni soruyor; düşülecek bir model zinciri rafa kalkan projeye fazla |

## Gereksinimler

- Windows 10 veya 11
- Python 3.13 ve [uv](https://docs.astral.sh/uv/)
- Bir mikrofon ve bir hoparlör
- Bir API anahtarı: [Google AI Studio](https://aistudio.google.com/apikey) (ücretsiz
  katman yeter), bir [Anthropic](https://console.anthropic.com/settings/keys) anahtarı ya
  da OpenAI uyumlu herhangi bir servisinki — ya da anahtar istemeyen yerel bir Ollama
- **GPU gerekmiyor.** Whisper `small` modelini int8 olarak dört CPU çekirdeğinde çalıştırır.

Dilin için kurulu bir Windows sesi, cevabın duyulabilir olmasıyla anlaşılabilir olması
arasındaki fark. Türkçe için *Microsoft Tolga* gerekiyor; Windows onu Ayarlar → Saat ve dil
→ Konuşma altından kuruyor.

## Kurulum

```bash
git clone https://github.com/emreux/windows-voice-assistant.git
cd windows-voice-assistant
uv sync
```

## Ayarlama

```bash
uv run assistant setup
```

Sağlayıcı, asistanın konuşacağı dil, API anahtarın (kendi sunucun için adresi) ve hangi
modelin cevap vereceği — anahtarının erişebildiği canlı listeden. Anahtar, hiçbir şey
yazılmadan önce sağlayıcıya karşı sınanıyor ve **Windows Kimlik Bilgisi Yöneticisi'ne**
gidiyor — asla bir dosyaya değil. Sonra model sınanıyor: tek bir soru ve tek bir araç
alıyor, yalnız aracı çağırırsa kabul ediliyor. Ayarlar `%APPDATA%\assistant\config.toml`
dosyasına düşüyor; elle düzenlenebilir düz TOML.

Son soru hangi mikrofonun dinleneceği — bir liste, en üstte "Windows'un seçtiği".
`uv run assistant mic` yalnız o soruyu yeniden soruyor, kulaklığa geçtiğin gün için:
`[audio]`'yu yeniden yazıyor, dosyanın kalanına dokunmuyor.

## Konuşma

```bash
uv run assistant run
```

İkon için `--tray` ekle. Satır `hazır` diyene kadar bekle. Zaten dinliyor: konuş yeter. Yarım saniye kadar
sustuğun her yerde o cümle bir tur oluyor; üç dört saniye sonra cevabın ilk cümlesini
duyuyorsun, konuşulanlar da durum satırının üstünden akıp gidiyor.

**`Ctrl+Alt+H` dinlemeyi kapatıp açar.** Nerede olursan ol çalışıyor — terminalin önde
olması gerekmiyor — ve durum satırı hangi durumda olduğunu her zaman söylüyor. `Ctrl+C`
programı durduruyor.

Bilmeye değer şeyler:

- **Saniyenin üçte birinden kısası** kelime değil gürültü sayılıyor.
- **Sözünü kesmek için kapat.** Asistan hâlâ konuşuyorken `Ctrl+Alt+H`'ye basarsan anında
  susuyor, beklediği isteği bırakıyor ve sessizleşiyor; devam etmek için bir daha bas.
  Sesle henüz kesilemiyor — konuşurken mikrofon sağır, kendi kendine cevap vermesin diye.
- **Sorunca cevap ver.** Riskli bir araç sorusunu okuyor ve altı saniye dinliyor. Evet ya
  da hayır de; ikisini de duymazsa bir kez daha soruyor, sonra sessizliği hayır sayıyor.
  Sorarken kapatmak hayır demek.
- **Bütün odayı duyuyor.** Televizyon, telefon görüşmesi, konuşan başka biri: her biri
  cevaplamaya çalışacağı bir tur. Başkalarının olduğu odada kapat. Yalnız adına cevap
  veren uyandırma kelimesi ardıl projenin işi.
- **Kendini duymuyor.** Cevap sürdüğü sürece mikrofon sağır, artı odanın onu tekrar etmeyi
  bırakması için çeyrek saniye.
- **Sessizlik cevaplanmıyor.** Tanıyıcıya kelimelerden ne kadar emin olduğu değil, kayıtta
  konuşma olup olmadığı soruluyor: sessiz oda hiçbir şey demiyor,
  okunamayan cümleye "Seni anlayamadım, tekrar söyler misin?" deniyor, kelimeler ise
  dekoder ne kadar kararsız olursa olsun cevaplanıyor.

Hangi mikrofonun dinleneceği, sen söylemedikçe, Windows'un seçtiği: `uv run assistant mic`
PortAudio'nun gördüğü her aygıtı, host API başına bir kez, listeliyor ve seçtiğini
saklıyor. Aynı liste `scripts/bench_mic.py --list-devices`; tek bir akşam için
`assistant run --device "Microphone Array 1"` satırındaki kelimelerle bir aygıt adlandırır.
İndeks değil kelime: indeksler her Bluetooth aygıtı bağlandığında kayıyor. 16 kHz'de
çalışmayan bir aygıt kendi hızında açılıyor ve girişte yeniden örnekleniyor; hangi aygıtın
açıldığını, nasıl seçilmiş olursa olsun, log söylüyor.

**Hangi mikrofon yolu — ölçüldü.** Geliştirme dizüstünde (Intel Smart Sound dizisi)
Windows'un ses geliştirmeleri açıkken varsayılan yol Whisper'ı bozuyordu: no-speech
0.15–0.43, kelimeler yanlış. Geliştirmeler kapalıyken ya da onları atlayan ham çekirdek
akışı girdisi "Microphone Array 1" ile aynı cümle no-speech 0.01–0.04'te birebir geldi.
Ham yolun seviyesi yarısı, okunması daha iyi — sorun ses yüksekliği değil işlemeydi.
Anlaşılma kötüyse önce geliştirmeleri, sonra ham girdiyi dene; paylaşımlı WASAPI yolu en
kötüsüydü. Odanın öbür ucundan eller serbeste güvenmeden önce mikrofonun oradan gerçekten
ne aldığını ölç:

```bash
uv run python scripts/bench_mic.py --quiet        # oda, kimse konuşmuyor
uv run python scripts/bench_mic.py --at "2 m"     # konuşarak, oturduğun yerden
uv run python scripts/bench_mic.py --echo         # hoparlörün geri verdiği
uv run python scripts/bench_mic.py --fixtures     # uç-nokta tespiti, kendi kayıtlarında
```

## Bir turun bedeli

Geliştirme makinesinde uçtan uca ölçüldü — dört çekirdekli, ayrık ekran kartı olmayan bir
dizüstü — Gemini 3.5 Flash-Lite ile, ortalama 2.6 sn'lik on dört Türkçe cümle üzerinde.
Cümleler Windows sesiyle sentezlendi, çünkü sahibinin kayıtları henüz alınmamıştı; süreler
sese bağlı değil, tanıma oranı bağlı.

| | p50 | p95 |
|---|---|---|
| Yazıya çevirme (Whisper `small`, int8, 4 iş parçacığı) | 2.8 sn | 3.3 sn |
| **Kayıttan ilk sese, araçsız** | **3.6 sn** | **4.2 sn** |
| Kayıttan ilk sese, bir araçla | 4.7 sn | (tek tur) |
| "Saat kaç", modele gitmeden | 3.0 sn | — |
| Eller serbestin eklediği, susmanı beklerken | +0.6 sn | +0.6 sn |

Beklemenin dörtte üçü yazıya çevirme. Aynı cümlelerde üç Whisper boyutu: `tiny` p50'de
0.56 sn ama kelimelerin %38'i yanlış, `base` 0.96 sn ve %27, `small` 2.9 sn ve %18 (yarısı
yabancı uygulama adları). Tasarımın yerel tanıma kapısı — `small` p95'te 1.2 sn'nin ve
kelimelerin %15'inin altında — sürede kaçırıldı; Google'ın tanıyıcısının seçenek olarak
durması bu yüzden (`[stt] provider = "gemini"`; sahibinin sekiz kaydında Whisper kelimelerin
yarısına yakınını yanlış okurken o dörtte birini yanlış okudu). Yerel Whisper varsayılan
kalıyor ve hiç gitmiyor, çünkü sesin onunla makineden çıkmıyor. `scripts/bench_stt.py` ve `scripts/bench_e2e.py` ikisini de kendi
kayıtlarında ölçüyor (bkz. `fixtures/audio/`).

Whisper program açılırken bir kez yükleniyor, yaklaşık üç saniye, böylece ilk basış onu
beklemiyor. Tek seferde, sıfır sıcaklıkta çözüyor; sesin taşıyabileceği kadar token
yazıyor; kendi etrafında dönen bir çözümü atıyor — böylece zor bir cümle otuz değil üç
saniye sürüyor ve kimsenin söylemediği bir kelime yerine "tekrar söyler misin" geliyor.
Her cümleden önce ona, senin dilinde, daha önce açtığın uygulamalar ve kalanların en kısa
adları söyleniyor — penceresine sığdığı kadar.

## Verilerin nereye gidiyor

- **Sesin makineden çıkmıyor.** Whisper yerelde çalışıyor; sağlayıcıya yalnız metin gidiyor.
  Tek istisna kendi yazacağın bir satır: `config.toml`'a `[stt] provider = "gemini"` yazarsan
  mikrofon sesi Google'ın tanıyıcısına gider (deneme; Google hayır dediğinde Whisper arkada
  yüklü durur). Yazmazsan makineden metinden başka hiçbir şey çıkmaz.
- **Asistanın söyledikleri burada kalıyor — Google'ın sesini seçmediysen.**
  `[tts] provider = "gemini"` ile her cevabın her cümlesi okunmak üzere Google'a gönderiliyor.
  Varsayılan olan Windows'un kendi sesi hiçbir şey göndermiyor.
- **Sayfa, panon ve mailin, sözlerinin gittiği yere gidiyor.** `fetch_page`,
  `read_clipboard` ve iki mail aracının döndürdüğü şey modelin önüne konuyor; yani
  yazıya çevrilmiş sesin gibi seçtiğin sağlayıcıya ulaşıyor. Mail şifresi Kimlik Bilgisi
  Yöneticisi'nde; sunucu ve adres `config.toml`'da.
- **Sende olmayan bir uygulamanın adı Microsoft'a gidiyor.** "X'i aç" makinede X bulamazsa
  X, `winget` üstünden Microsoft Store'da aranıyor. Başka hiçbir şey aranmıyor; `winget`
  kurulu değilse hiçbir şey.
- **Mesaj, senin kendin göndereceğin yere gidiyor, başka hiçbir yere.** WhatsApp'ta metin
  ve numara bu bilgisayardaki WhatsApp uygulamasına gidiyor — burada hiçbir şey WhatsApp'ın
  protokolünü konuşmuyor, konuşmayacak da. Telegram'da metin, senin kendi hesabın
  üzerinden Telegram'ın sunucularına gidiyor; hesabının yerine geçen oturum dizisi Kimlik
  Bilgisi Yöneticisi'nde. `contacts.toml` makineden çıkmıyor ve bu deponun parçası değil.
- **Havasını sorduğun yer Open-Meteo'ya, aramanın sözleri kendi motoruna gidiyor.** Şehir
  adı Open-Meteo'nun geocoder'ına ve tahminine gönderiliyor (anahtar yok, hesap yok, senin
  hakkında başka hiçbir şey yok). Web aramasının sözleri `[web] search_url`'deki motora,
  kendi tarayıcında, elle yazılmış bir arama neyse aynen o.
- **API anahtarın hiçbir dosyaya yazılmıyor.** `keyring` üzerinden Windows Kimlik Bilgisi
  Yöneticisi'nde duruyor.
- **Söylediğin yazılmıyor; asistanın yaptığı yazılıyor.**
  `%LOCALAPPDATA%\assistant\assistant.db` her araç çağrısını (hangi araç, hangi argümanlar,
  ne oldu, ne zaman), notlarını ve hatırlatıcılarını, her turun token sayısını ve
  fiyatını, ve modelin hakkındaki kararı tutuyor. Aracın verdiği cevap otuz gün sonra
  kayıttan siliniyor. Konuşmanın kendisi son on iki tur; bellekte duruyor, program
  kapanınca gidiyor. Log sayıları yazıyor — token, araç, fiyat, ilk sesin ne kadar
  sürdüğü — kelimeleri asla. `assistant purge --all` hepsini siliyor: önce listeleyip,
  sen `yes` yazdıktan sonra.
- **Hatırlamasını istediklerin düz metin.** `%APPDATA%\assistant\memory.toml` dolaşan
  profille seni izliyor; elle düzenle ya da sil.
- Ayarlar: `%APPDATA%\assistant\config.toml`. Fiyatlar: yanındaki `pricing.toml` paketle
  gelen tabloyu eziyor. Log: `%LOCALAPPDATA%\assistant\Logs\assistant.log`.

## Yeni bir dil eklemek

`src/assistant/locales/_template.toml` dosyasını `<kod>.toml` olarak kopyala — `de.toml`,
`es.toml`, `ja.toml` — ve sağ tarafı çevir: cümleler, evet ve hayır kelimeleri, kısa
komutlar, dolgu sesi, araç testi sorusu. Prosedürün tamamı bu; kod değişmiyor. Boş
bıraktığın bir anahtar İngilizce cevaplanıyor, yani yarım kalmış bir paket bozuk değil
kullanılabilir oluyor.

## Yeni bir sağlayıcı eklemek

OpenAI sohbet API'sini konuşuyorsa `src/assistant/defaults/providers.toml`'a adresini ve
anahtar isteyip istemediğini söyleyen bir satır ekle, bu kadar. Gemini ve Anthropic'in kendi
adaptörleri var. Seninki ikisini de konuşmuyorsa `LLMProvider`
protokolünü sağlayan bir adaptör yaz, test dosyasına bir `build` fonksiyonu koy ve
`tests/test_llm_adapters.py` içindeki `ADAPTERS` listesine bir satır ekle. Sözleşme testi
bundan sonra diğerlerine sorduğu her soruyu seninkine de soruyor, hiç değişmeden — ve
sözleşmeden geçirilmemiş bir adaptör kaydedilirse test kırmızıya dönüyor.

## Mesajlaşma

Mesaj attığın kişileri `config.toml`'un yanındaki `%APPDATA%\assistant\contacts.toml`'a yaz:

```toml
[[contact]]
name = "Ahmet Yılmaz"
aliases = ["Ahmet", "abi"]
phone = "+90 532 000 00 00"      # WhatsApp için: uluslararası biçim, başta 0 olmaz
telegram = "ahmetyilmaz"         # @ olmadan kullanıcı adı; boş bırakırsan adıyla aranır
```

Asistanın güvenemeyeceği bir dosya — ülke kodsuz numara, tanımadığı bir anahtar, aynı ada
cevap veren iki kişi — açılışta satırı söyleyen bir cümleyle durduruyor; çünkü bu özelliğin
yapabileceği en kötü şey yanlış kişiye yazmak.

**WhatsApp** için Microsoft Store'daki WhatsApp uygulaması gerekiyor, telefonuna bağlı.
Asistan sohbeti WhatsApp'ın kendi bağlantısıyla açıyor ve Enter'a yalnız WhatsApp öndeki
pencereyken basıyor, tuştan hemen önce bir daha bakarak; değilse mesaj sohbette yazılı
kalıyor ve sana söyleniyor.

**Telegram** için https://my.telegram.org adresinden kendi `api_id` ve `api_hash`'in ("API
development tools", iki dakika) ve bir kez `assistant telegram login`: telefon, Telegram'ın
uygulamana gönderdiği kod, varsa iki adımlı şifren. `api_id` `config.toml`'a, hash ve oturum
Kimlik Bilgisi Yöneticisi'ne gidiyor. Çıkış Telegram'ın kendi "Aktif oturumlar" ekranından.

`config.toml`'da `[messaging] default_app = "WhatsApp"` her seferinde uygulamayı söylemekten
kurtarıyor; yazmazsan asistan hangisi diye soruyor.

## Mail

```bash
uv run assistant mail login
```

IMAP sunucusu (`imap.gmail.com`, `outlook.office365.com`), giriş yaptığın adres ve bir
**uygulama şifresi** — iki adımlı doğrulaması olan her hesapta hesap şifren IMAP'te
çalışmaz; zaten hiçbir şeye yazmaman gerekir. Komut üçünü kanıtlamak için bir kez
bağlanıyor, sonra şifreyi Kimlik Bilgisi Yöneticisi'nde, kalanını `config.toml`'un `[mail]`
tablosunda tutuyor; `port` (993) ve `mailbox` (`INBOX`) orada elle değiştirilebiliyor.
Posta kutusuna hiçbir şey yazılmıyor: bayrak yok, taşıma yok, gönderme yok.

Bu, sahte bir IMAP sunucusuyla yazıldı ve test edildi, çünkü bunun için hesap açılmadı.
Gmail ve Outlook buradaki IMAP'i konuşuyor; UTF-8 arama sözcüğünü reddeden sunucuya
harfler katlanarak yeniden soruluyor. Güvenmeden önce kendi hesabında dene.

## Kendi aracını eklemek

`%APPDATA%\assistant\tools\` klasörüne — `config.toml`'un yanına — `assistant.tools.registry`
içindeki `@tool` ile bildirilmiş fonksiyonlar taşıyan bir `.py` dosyası koy; yerleşik araçlar
nasıl yazılıyorsa öyle. Bir sonraki açılışta hazırlar: aynı süreçte, aynı izin kapısından
geçerek, kendi risklerini kendileri söyleyerek çalışırlar ve senin makinende kalırlar — o
klasördeki hiçbir şey bu deponun parçası değil. İçe aktarılamayan bir dosya çökme değil,
günlükte bir satır olur.

## Geliştirme

```bash
uv run ruff check .            # lint
uv run ruff format .           # biçim
uv run mypy src --strict       # tip denetimi
uv run pytest                  # 1763 test; hiçbiri ağa ya da mikrofona dokunmuyor
```

## Ölçülmeyenler

Üç parça gerçek şeye karşı hiç koşulmadı; kodda ve yukarıda öyle işaretli.

- **Anthropic adaptörü.** SDK'nın tipleri ve öbür ikisinin geçtiği sözleşme testiyle
  yazıldı; hiç gerçek anahtar verilmedi.
- **Mail.** Sahte IMAP sunucusuyla yazıldı; hiç gerçek posta kutusu verilmedi.
- **Evde WhatsApp.** `whatsapp://send` yolu ve Enter'ın çevresindeki dört süre geliştirme
  makinesinde ayarlandı, sahibinin kendi makinesinde ölçülmedi.

*Ne yapıyor* altındaki her şey en az bir kez geliştirme dizüstünde, Türkçe, Gemini'yle
gerçekten çalıştırıldı.

## Lisans

MIT — bkz. [LICENSE](LICENSE).

---

English: [README.md](README.md)
