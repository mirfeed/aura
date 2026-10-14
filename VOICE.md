# Голос Aura (Day 4 + Day 5)

## Установка (Windows, Python 3.10)

```
pip install -U -r requirements.txt
```

Для распознавания на видеокарте NVIDIA нужны cuBLAS и cuDNN 9 (CUDA 12). Если CUDA Toolkit не стоит:

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

Без них Whisper сам переключится на процессор (в логе будет `Whisper failed to load on device`, потом `Whisper loaded ... device: cpu`).

FFmpeg не нужен. Opus уже входит в discord.py для Windows.

## Настройка

Скопируйте нужные строки из `voice.env.example` в `.env`. Значения по умолчанию уже рабочие.

В Discord Developer Portal у бота должен быть включён Message Content Intent (он уже включён). На сервере боту нужны права **Connect** и **Speak** в голосовых каналах.

## Как пользоваться

- «Аура, зайди в войс»: бот заходит в канал, где сидит автор сообщения.
- «Аура, выйди из войса»: бот выходит.
- В войсе бот отвечает голосом, если в фразе прозвучало «Аура», или на любую фразу, если в канале с ним один человек.
- Бот сам выходит, когда в канале не остаётся людей.

При первом запуске скачиваются модели (Whisper small около 0,5 ГБ, Silero около 0,1 ГБ) в `data/models/`.

## Что смотреть в logs/bot.log

| событие | значит |
|---|---|
| `DAVE receive patch applied` | патч для расшифровки входящего звука включён |
| `Whisper loaded` / `Silero TTS loaded` | модели готовы, видно устройство и время загрузки |
| `voice_transcript` | что распознано, обращались ли к боту |
| `voice_reply_sent` | сколько секунд ушло на ответ модели (llm) и озвучку (tts) |

Если в распознанном тексте мусор вместо слов, значит не расшифровывается DAVE. Включите `LOG_LEVEL=DEBUG` и ищите `DAVE decrypt failed`.

## Fish Speech вместо Silero

Fish Speech (модель OpenAudio S1-mini) звучит живее: перед каждым предложением бот ставит маркер эмоции по своему настроению, например `(joyful)`, `(curious)`, `(frustrated)` или `(soft tone)`, а на восклицаниях `(excited)`. Если `TTS_EMOTIONAL=false`, маркеров нет.

Fish работает отдельным HTTP-сервером (ему нужны Python 3.12 и torch с CUDA, бот остаётся на своём окружении). Бот шлёт фразы на `FISH_API_URL/v1/tts`.

1. Поднять сервер fish-speech с моделью S1-mini на порту 8080 (лучше в WSL2: там работает `--compile`, и синтез заметно быстрее).
2. Положить образец голоса на сервер: `references/aura/sample.wav` (10–30 секунд чистой речи) и `references/aura/sample.lab` (её текст). Без образца голос каждый раз будет случайным.
3. В `.env` поставить `TTS_ENGINE=fish`.

Silero грузится и в этом режиме: если сервер Fish не ответил, фразу озвучит Silero, а Fish бот попробует снова через `FISH_RETRY_AFTER_S` секунд. В логе это `fish_tts_failed`.

| событие | значит |
|---|---|
| `Fish TTS ready` | сервер ответил, видно адрес, голос и время прогрева |
| `fish_tts_failed — Silero speaks instead` | сервер не ответил, временно говорит Silero |

## Ограничения

- discord-ext-voice-recv официально не поддерживает DAVE, поэтому входящий звук расшифровывает наш патч `voice/dave_patch.py`.
- Модели Silero распространяются по лицензии CC BY-NC: только некоммерческое использование.
- OpenAudio S1-mini: CC BY-NC-SA 4.0, тоже только некоммерческое использование.
