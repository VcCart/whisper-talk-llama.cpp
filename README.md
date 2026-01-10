# whisper-talk-llama.cpp fork whisper.cpp

## Добавлено:
     
    UTF8 and Russian — UTF8 и русский язык  
    Доработано: потоковая генерация, потоковый XTTS, агрессивный VAD  
    Голосовые команды: Google, stop, regenerate, delete, reset, call  
    Прерывание генерации/синтеза речи при разговоре пользователя
    Добавлены возможности библиотеки SDL2
	  Добавлена возможность последующей озвучки в XTTSv2 

## Дополнительно может использоваться: 
- XTTSv2 server in streaming-mode
- langchain google-serper

## Новости
- [2025.11.27] Собственные доработки, которых очень много.
- [2025.11.01] Все изменеия из https://github.com/Mozer/talk-llama-fast/


## Заметки
   -  В talk-llama.cpp был изменен сдвиг контекста под whisper.cpp > 1.8.0., и изменена работа с кэшем. В версии 1.76 это реализовывалось по другому.
   -  Диалог с talk-llama может вестись почти бесконечно — модель остаётся адекватной, серьёзных зацикливаний или повсеместных проблем не наблюдается. 
   -  Llama запоминает начальный промпт и последние N токенов контекста, но всё, что находится между ними, теряется. 
   -  Дополнительная видеопамять (VRAM) ,больше той что уже занялась при запуске не расходуется — вы можете вести практически бесконечный диалог без потери скорости.  
   -  talk-llama.cpp тестировался на llm модели saiga_yandexgpt_8b_Q4_K_S.gguf и Whisper модели whisper-ggml-large-v3-q4.bin
   -  В качестве тестовой видеокарты использовалсась карта GTX1070 ti всего 8 ГБ на архитектуре Pascal. Лёгкую квантованную версию llama вполне нормально загружает.
   -  Далее была попытка запуска скомпилированных файлов на RTX 3060 12GB на архитектуре Ampere и файлы оказались несовместимы, так что проект придется перекомпилировать.
   -  Процессор желателен с AVX2 инструкциями, но и здесь можно обойти ограничение, скомпилировав проект без них; 
   -  XTTS можно запустить с флагом --lowvram или даже на CPU вместо GPU (-d=cpu, но это будет медленно), лучше сэкономить  GPU на llm, так как может llama приемлемо  работает на мощном CPU процессоре, а tts уже не справляется.
   -  Для использования с колонками (а не наушниками): Вы можете попробовать отключить прерывание речи бота из-за шума, установив --vad_start_thold 0.  
   -  Опционально: есть команда «пробуждения» — --wake-command "Эмма," (запятая после имени обязательна). Теперь только фразы, начинающиеся например, с имени «Эмма», будут отправляться в чат. Это частично поможет при работе с колонками или в шумном помещении, но лучше придумать как отключать микрофон вручную, или использовать наушники.

## Языки
Программа Мультиязычная, но зависит от подгуженных моделей Whisper и LLM.

## Примерные системные требования
- Windows 10/11 x64
- python, cuda 11, 12
- Recommended: nvidia GPU with 8 GB vram. Minimum: nvidia with 6 GB. 
- Для AMD, macos, linux - Не собиралось, не тестировалось и неизвестно заработает ли. Скорее всего нет...  

## Установка
### For Windows 10/11 x64 with CUDA.
- CUDA для разработки на Nvidia:
https://developer.nvidia.com/cudnn-archive
https://developer.nvidia.com/cuda-toolkit-archive
Проверить версию: nvcc --version в командной строке.
Рекомендую испоьзовать CUDA до версии 12.9x. Выше наблюдаются проблемы несовместимости.
- Загрузите [release](https://github.com/VcCart/whisper-talk-llama.cpp) или скомпилируйте самостоятельно. 
Распакуйте в папку c:\DATA\ .
- Загрузите модель whisper в папку c:\DATA\ с whisper-talk-llama.exe: Для Русского языка может подойти [ggml-large-v3-q4_k.bin](https://huggingface.co/adriabama06/whisper-large-v3-ggml) Или другой квантизации, в зависимости от объема VRAM.
- Загрузите LLM в ту же папку [saiga_yandexgpt_8b_Q4_K_S.gguf](https://huggingface.co/IlyaGusev/saiga_yandexgpt_8b_gguf/tree/main) Вы можете попробовать Q4_K_S или Q3_K_S, если у вас под llm запланировано мало VRAM.

Теперь установим xtts-api-server и TTS От Mozer (Ссылки на свои форки я поправлю позже, если опубликую). 
Примечание: XTTS с DeepSpeed требует PyTorch 2.1, но некоторые пакеты DeepSpeed требуют PyTorch 2.2 и выше, поэтому Depspeed придется компилировать или искать готовый whl
Все представленные здесь компоненты тестировались на Python 3.11 с разными версиями PyTorch. 
Установка окружения состоит в основном из:  Git, Python 3.11, XTTS сервера, langchain_community для google поиска и прочих модулей.

Для установки и запуска Xtts-Api-Server нужно Python окружение.
Подойдет Python 3.10 - 3.12 в зависимости от версии coqui-tts
В оригинальном Xtts-Api-Server используется coqui-tts 0.24.1
В Xtts-Api-Server от Mozer используется coqui-tts 0.22.0 c небольшими доработками.

Откройте папку \Data, куда вы положили основные файлы с talk-llama. В этой папке откройте командную строку (cmd) и выполняйте команды построчно:

```
git clone https://github.com/Mozer/xtts-api-server 
cd c:\DATA\xtts-api-server\
```
Создайте окружение Python в той же папке:
```
python -m venv venv
```
Активируйте в windows так:
```
venv\Scripts\activate
```
Далее можно устанвливать:
```
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/Mozer/tts

```
Запуск xtts сервера можно настроить в start.bat примерно такого содержания:
```
call venv/scripts/activate
python -m xtts_api_server  --deepspeed --stream-play-sync --streaming-mode-improve --lowvram -d=cuda
pause
```
Как более упрощенный вариант, можно использовать --streaming-mode вместо --streaming-mode-improve, чтобы не загружалась более тяжелая NLP Stanza.
Тогда обработка стрима будет на более легком Ntlk.

- Если при установке xtts-api-server возникают ошибки, сверьтесь с инструкциями (в них показана установка оригинального XTTS, а не модифицированный, без управления VAD остановкой, когда пользователь говорит): [xtts-api-server](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#installation)

При первой установке xtts-api-server система может запросить установку [visual-cpp-build-tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/). Стандартная страница загрузки от Microsoft может измениться, поэтому можно поискать установщик через поисковик в интернете.

Возможно, потребуется установить в windows библиотеки [VisualCppRedist](https://gitlab.com/stdout12/vcredist). Иногда это даже обязательно. Установщики vcredist разных версий еще можно найти на сайте Microsoft, если не доверяете сборщику.

Установка ffmpeg скорее всего не понадобится, так как у нас tall-llama без wavtolip для экономии ресурсов, но инструкция будет ниже:
- Download [ffmpeg full](https://www.gyan.dev/ffmpeg/builds/), 
Добавьте его в переменную PATH в вашем Windows (как это сделать: https://phoenixnap.com/kb/ffmpeg-windows). 
Если нужна установка dll библиотеки кодека h264 свежую версию можно взять по ссылке https://github.com/cisco/openh264/releases и положить в /system32 или /ffmpeg/bin . 
Последняя на данный момент: openh264-2.6.0-win64.dll.
Это может пригодиться для других модулей ассистенка.

## Запуск
- Сервер XTTS запускать в папке C:\DATA\xtts-api-server через start.bat, который мы создали по инструкции выше.

ПРИМЕЧАНИЕ: При первом запуске XTTS скачает DeepSpeed с GitHub. Если загрузка DeepSpeed завершится ошибкой вида «Warning: Retrying (Retry... ReadTimeoutError...)», включите VPN для загрузки DeepSpeed (27 МБ) и чекпоинта XTTS (1,8 ГБ), после чего можно отключить VPN. 

Чекпоинт (Модели) XTTS можно скачать и без VPN. Однако если вы прервёте загрузку, чекпоинт будет повреждён — в этом случае вам нужно вручную удалить папку \xtts_models\ и перезапустить XTTS.
**ПРИМЕЧАНИЕ:** если в имени `.bat`-файла есть кириллические (русские) символы, сохраните его в кодировке **OEM 866** (Notepad++ поддерживает эту кодировку).

- Запустите talk-emma.bat для старта консоли whisper-talk-llama.exe

Примерное содержание файла:
```
@echo off
chcp 866
whisper-talk-llama.exe -mw ggml-large-v3-q4_k.bin -ml saiga_yandexgpt_8b_Q5_K.gguf --language ru -p "Друг" --speak speak --vad-last-ms 320 --vad-start-thold 0.000150  --bot-name "Эмма" --xtts-voice "Эмма" --prompt-file talk_emma_inst.txt  --xtts-url http://localhost:8020/  --instruct-preset ChatML --temp 0.20 --min_p 0.10  -ngl 30  -n 256 --ctx_size 2560  --threads 20  --allow-newline --sleep-before-xtts 1100 --flash-attn
```

### Настройки для видеопамяти 6 и 8 ГБ

- **Использовать CPU вместо GPU** (будет медленнее, ~5-6 сек):  
  В файле `talk-llama` найдите параметр `ngl` и измените его на `-ngl 0` (это даст максимальную скорость при работе на CPU).

- **Уменьшите контекст для Llama**:  
  Добавьте или измените параметр запуска на `--ctx_size 512`.

- **Включите экономный режим видеопамяти для XTTS**:  
  В файле `xtts_start.bat` добавьте флаг `--lowvram`.  
  *Примечание:* это будет перемещать модель XTTS из видеопамяти (VRAM) в оперативную память (RAM) после каждого запроса, что замедлит работу, но значительно снизит потребление VRAM.

### Дополнительно
- Помещайте новые голоса XTTS в папку `\xtts\speakers\`. Рекомендуется использовать монофонические WAV-файлы с глубиной 16 бит, частотой дискретизации 22050 Гц и длительностью около 10 секунд, без шумов и музыки.
С опцией командной строки `--multi-chars` в режиме чата будет передавать имя нового персонажа в XTTS, командой "Позови Алису", "Позови Олега".
Если XTTS не найдёт соответствующий голос — будет использован голос по умолчанию: "default.wav".

- Поместите описание персонажа и несколько примеров его реплик в файл talk_emma_inst.txt для того чтобы задать свой базовый промпт ассистента.
- В C:\DATA\instruct_presets должен быть ChatML.json с инструкциями для соответсвующей модели.

- Голоса персонажей хранятсяв виде .wav-файлов в папке c:\DATA\xtts\speakers\. Вы также можете создать копии аудио с разными именами (например, Алиса или Олег). Теперь вы сможете обращаться к ним по имени.


#### Опционально плагин гугл поиска
- search_server.py - выложу позже после доработки.
- **Скачайте** [search_server.py]
- **Установите**: `pip install langchain`
- Зарегистрируйтесь на сайте https://serper.dev/api-key Сервис бесплатный и быстрый, предоставляет 2500 бесплатных поисковых запросов. Получите API-ключ и вставьте его в файл search_server.py на строке 13: `os.environ["SERPER_API_KEY"] = "your_key"`
- Запустите сервер поиска search_server.py создав батник под свой python
- Теперь вы можете использовать голосовые команды, например:
: `Пожалуйста, погугли, кто такой Джон Доу` или `Пожалуйста погугли погоду в Москве`.


## Сборка и компиляция
- Для NVIDIA и Windows. Для других систем — не тестировалось, поэтому мануалы ищите в исходных репозиториях, там есть как собрать под cpu например. 
- скачайте https://www.libsdl.org/release/ https://www.libsdl.org/release/SDL2-2.32.10-win32-x64.zip распакуйте в C:\DATA\whisper-talk-llama.cpp/SDL2/ 
- Также установите libcurl из пакета vcpkg:
```
cd c:\DATA\
git clone https://github.com/Microsoft/vcpkg.git
cd c:\DATA\vcpkg
bootstrap-vcpkg
vcpkg integrate install
vcpkg install curl[tool]
vcpkg install pkgconf
```
- Измените путь к `c:\\DATA\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake` ниже — в папку, куда вы установили vcpkg. Затем соберите проект.
Сделать это можно примерно так:
```
set VCPKG_ROOT=c:\DATA\vcpkg\
set PATH=%VCPKG_ROOT%;%PATH%
```
Далее клонируем репозиторий с исходниками:
```
cd c:\DATA\
git clone https://github.com/VcCart/talk-llama.cpp.git
cd talk-llama.cpp
set SDL2_DIR=SDL2\cmake

cmake.exe -DWHISPER_SDL2=ON  -DGGML_CUDA=1 -DCMAKE_TOOLCHAIN_FILE=C:/DATA/vcpkg/scripts/buildsystems/vcpkg.cmake -B build

cmake.exe --build build -j --config release --target clean
cmake.exe --build build -j --config release --parallel 8

for old CPU's without AVX2 / для процессоров без AVX2: 

cmake.exe -DWHISPER_NO_AVX2=1 -DWHISPER_SDL2=ON -DWHISPER_CUBLAS=0 -DGGML_CUDA=1 cmake.exe -DWHISPER_SDL2=ON  -DGGML_CUDA=1 DCMAKE_TOOLCHAIN_FILE=C:/DATA/vcpkg/scripts/buildsystems/vcpkg.cmake -B build
Потом повторите две команды по очистке и сборке, как выше

Компиляция может длиться около 10 мин и больше, в зависимости от вашего компьютерного железа.

```
## whisper-talk-llama.exe params / Параметры командной строки для bat файла
```
  -h,       --help           [default] show this help message and exit
  -t N,     --threads N      [4      ] number of threads to use during computation
  -vms N,   --voice-ms N     [10000  ] voice duration in milliseconds
  -c ID,    --capture ID     [-1     ] capture device ID
  -mt N,    --max-tokens N   [32     ] maximum number of tokens per audio chunk
  -ac N,    --audio-ctx N    [0      ] audio context size (0 - all)
  -ngl N,   --n-gpu-layers N [999    ] number of layers to store in VRAM
  -vth N,   --vad-thold N    [0.60   ] voice avg activity detection threshold
  -vths N,  --vad-start-thold N [0.000270] vad min level to stop tts, 0: off, 0.000270: default
  -vlm N,   --vad-last-ms N  [0      ] vad min silence after speech, ms
  -fth N,   --freq-thold N   [100.00 ] high-pass frequency cutoff
  -su,      --speed-up       [false  ] speed up audio by x2 (not working)
  -tr,      --translate      [false  ] translate from source language to english
  -ps,      --print-special  [false  ] print special tokens
  -pe,      --print-energy   [false  ] print sound energy (for debugging)
  --debug                    [false  ] print debug info
  -vp,      --verbose-prompt [false  ] print prompt at start
  --verbose                  [false  ] print speed
  -ng,      --no-gpu         [false  ] disable GPU
  -fa,      --flash-attn     [false  ] flash attention
  -p NAME,  --person NAME    [Georgi ] person name (for prompt selection)
  -bn NAME, --bot-name NAME  [LLaMA  ] bot name (to display)
  -w TEXT,  --wake-command T [       ] wake-up command to listen for
  -ho TEXT, --heard-ok TEXT  [       ] said by TTS before generating reply
  -l LANG,  --language LANG  [en     ] spoken language
  -mw FILE, --model-whisper  [models/ggml-base.en.bin] whisper model file
  -ml FILE, --model-llama    [models/ggml-llama-7B.bin] llama model file
  -s FILE,  --speak TEXT     [./examples/talk-llama/speak] command for TTS
  -sf FILE, --speak-file     [./examples/talk-llama/to_speak.txt] file to pass to TTS
  --prompt-file FNAME        [       ] file with custom prompt to start dialog
  --instruct-preset TEXT     [       ] instruct preset to use without .json
  --session FNAME                   file to cache model state in (may be large!) (default: none)
  -f FNAME, --file FNAME     [       ] text output file name
   --ctx_size N              [2048   ] Size of the prompt context
  -b N,     --batch-size N   [64     ] Size of input batch size
  -n N,     --n_predict N    [64     ] Max number of tokens to predict
  --temp N                   [0.90   ] Temperature
  --top_k N                  [40.00  ] top_k
  --top_p N                  [1.00   ] top_p
  --min_p N                  [0.00   ] min_p
  --repeat_penalty N         [1.10   ] repeat_penalty
  --repeat_last_n N          [256    ] repeat_last_n
  --n_keep N                 [128    ] keep first n_tokens after context_shift
  --main-gpu N               [0      ] main GPU id, starting from 0
  --split-mode NAME          [none   ] GPU split mode: 'none' or 'layer'
  --tensor-split NAME        [(null) ] Tensor split, list of floats: 0.5,0.5
  --xtts-voice NAME          [emma_1 ] xtts voice without .wav
  --xtts-url TEXT            [http://localhost:8020/] xtts/silero server URL, with trailing slash
  --xtts-control-path FNAME  [c:\DATA\LLM\xtts\xtts_play_allowed.txt] not used anymore
  --xtts-intro               [false  ] xtts instant short random intro like Hmmm.
  --sleep-before-xtts        [0      ] sleep llama inference before xtts, ms.
  --google-url TEXT          [http://localhost:8003/] langchain google-serper server URL, with /
  --allow-newline            [false  ] allow new line in llama output
  --multi-chars              [false  ] xtts will use same wav name as in llama output
  --push-to-talk             [false  ] hold Alt to speak
  --seqrep                   [false  ] sequence repetition penalty, search last 20 in 300
  --split-after N            [0      ] split after first n tokens for tts
  --min-tokens N             [0      ] min new tokens to output
  --stop-words TEXT          [       ] llama stop w: separated by ;
```

## Голосовые команды:
Полный список команд и их вариаций находится в `talk-llama.cpp`, search `user_command`.
- Stop (остановись, Ctrl+Space)
- Regenerate (переделай, , Ctrl+Right) - will regenerate llama answer
- Delete (удали, Ctrl+Delete) - will delete user question and llama answer.
- Delete 3 messages (удали 3 сообщениия)
- Reset (удали все, Ctrl+R) - will delete all context except for a initial prompt
- Google something (погугли что-то)
- Сall NAME (позови Алису)

## Licenses / Лицензии
- whisper-talk-llama - MIT License - OK for commercial use
- whisper.cpp - MIT License - OK for commercial use
- whisper - MIT License - OK for commercial use
- TTS(xtts) - Mozilla Public License 2.0 - OK for commercial use
- xtts-api-server - MIT License - OK for commercial use