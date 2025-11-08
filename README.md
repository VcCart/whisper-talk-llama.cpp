# talk-llama.cpp fork whisper.cpp

## Добавлено:
    XTTSv2 support — добавлена поддержка XTTSv2  
    UTF8 and Russian — UTF8 и русский язык  
    Доработано: потоковая генерация, потоковый XTTS, агрессивный VAD  
    Голосовые команды: Google, stop, regenerate, delete, reset, call  
    Прерывание генерации/синтеза речи при разговоре пользователя
    Add SDL2 / добавлена библиотека SDL2

## Дополнительно может использоваться: 
- XTTSv2 server in streaming-mode
- langchain google-serper

## Новости
- [2025.11.01] Собственные доработки, которых очень много.
- [2025.10.31] Все изменеия из https://github.com/Mozer/talk-llama-fast/
- [2025.10.31] initial commit

## Заметки
   -  В talk-llama.cpp был изменен сдвиг контекста под whisper.cpp > 1.8.0., и изменена работа с кэшем. Диалог может вестись почти бесконечно — модель остаётся адекватной, серьёзных зацикливаний или повсеместных проблем не наблюдается. 
   -  Llama запоминает начальный промпт и последние N токенов контекста, но всё, что находится между ними, теряется. 
   -  Дополнительная видеопамять (VRAM) не расходуется — вы можете вести практически бесконечный диалог без потери скорости.  
   -  talk-llama.cpp тестировался на llm модели saiga_yandexgpt_8b_Q4_K_S.gguf и Whisper модели whisper-ggml-large-v3-q4.bin
   -  В качестве тестовой видеокарты использовалсась карта всего 8 ГБ на архитектуре Pascal. Лёгкую квантованную версию llama вполне нормально загружает. Процессор желателен с AVX2 инструкциями; 
   -  XTTS можно запустить с флагом --lowvram или даже на CPU вместо GPU (-d=cpu, но это будет медленно). 
   -  Для использования с колонками (а не наушниками): Вы можете попробовать отключить прерывание речи бота из-за шума, установив --vad_start_thold 0.  
   -  Опционально: есть команда «пробуждения» — --wake-command "Эмма," (запятая после имени обязательна). Теперь только фразы, начинающиеся с имени «Эмма», будут отправляться в чат от вашего имени. Это частично поможет при работе с колонками или в шумном помещении.

## Языки
Программа Мультиязычная, зависит от подгуженных моделей Whisper и LLM.

## Примерные системные требования
- Windows 10/11 x64
- python, cuda
- Recomended 12-16 GB RAM
- Recommended: nvidia GPU with 8 GB vram. Minimum: nvidia with 6 GB. 
- Для AMD, macos, linux - Не собиралось, не тестировалось и неизвестно заработает ли.  

## Установка
### For Windows 10/11 x64 with CUDA.
- CUDA для разработки на Nvidia:
https://developer.nvidia.com/cudnn-archive
https://developer.nvidia.com/cuda-toolkit-archive
Проверить версию: nvcc --version в командной строке.
- Загрузите [release](https://github.com/VcCart/whisper-talk-llama.cpp) in zip. Распакуйте например в папку c:\DATA\ .
- Загрузите модель whisper в папку c:\DATA\ с talk-llama.exe: Для Русского языка может подойти [large-v3-q4_0.bin](https://huggingface.co/Ftfyhh/whisper-ggml-q4_0-models/blob/main/whisper-ggml-large-v3-q4_0.bin) .
- Загрузите LLM в ту же папку [saiga_yandexgpt_8b_Q4_K_S.gguf](https://huggingface.co/IlyaGusev/saiga_yandexgpt_8b_gguf/resolve/main/saiga_yandexgpt_8b.Q4_K_S.gguf?download=true) Вы можете попробовать q4_K_S или q3, если у вас не так много видеопамяти VRAM.
Теперь установим xtts-api-server и TTS (ССылки я поправлю позже). Примечание: XTTS с DeepSpeed требует PyTorch 2.1, но некоторые пакеты в требуют PyTorch 2.2. 
Все компоненты тестировались на Python 3.11 в окружении Miniforge3 с разными версиями PyTorch. Установка состоит в основном из: XTTS и Git.


Для установки и запуска Xtts-Api-Server нужно Python окружение.
Подойдет Python 3.10 - 3.12 в зависимости от версии xtts2 

Установите [miniforge](https://github.com/conda-forge/miniforge). Во время установки обязательно отметьте галочкой пункт («Добавить Miniconda3/Miniforge3 в переменную среды PATH») — это важно.


Откройте папку \xtts, куда вы распаковали архив whisper-talk-llama-1.82.zip. В этой папке откройте командную строку (cmd) и выполняйте команды построчно:
```
conda create -n xtts
conda activate xtts
conda install python=3.11
conda install git

pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/Mozer/xtts-api-server pydub
pip install git+https://github.com/Mozer/tts
pip install https://github.com/S95Sedan/Deepspeed-Windows/releases/download/v14.0%2Bpy311/deepspeed-0.14.0+ce78a63-cp311-cp311-win_amd64.whl
conda deactivate
```
- Если при установке xtts-api-server возникают ошибки, сверьтесь с инструкциями (в них устанавливается оригинальный XTTS, а не модифицированный): [xtts-api-server](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#installation)
При первой установке xtts-api-server система может запросить установку [visual-cpp-build-tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/). Стандартная Страница загрузки от Microsoft может измениться, поэтому можно поискать установщик через поисковик в интернете.
Воозможно, потребуется установить [VisualCppRedist](https://github.com/abbodi1406/vcredist).
- Download [ffmpeg full](https://www.gyan.dev/ffmpeg/builds/), put into your PATH environment (how to: https://phoenixnap.com/kb/ffmpeg-windows). Then download h264 codec .dll of required version from https://github.com/cisco/openh264/releases and put to /system32 or /ffmpeg/bin dir. In my case for Windows 11 it was openh264-1.8.0-win64.dll. 

## Запуск
- В папке /xtts/ дважды щёлкните по файлу xtts_start.bat, чтобы запустить сервер XTTS.
ПРИМЕЧАНИЕ: При первом запуске XTTS скачает DeepSpeed с GitHub. Если загрузка DeepSpeed завершится ошибкой вида «Warning: Retrying (Retry... ReadTimeoutError...)», включите VPN для загрузки DeepSpeed (27 МБ) и чекпоинта XTTS (1,8 ГБ), после чего можно отключить VPN. Чекпоинт XTTS можно скачать и без VPN. Однако если вы прервёте загрузку, чекпоинт будет повреждён — в этом случае вам нужно вручную удалить папку \xtts_models\ и перезапустить XTTS.
**ПРИМЕЧАНИЕ:** если в имени `.bat`-файла есть кириллические (русские) символы, сохраните его в кодировке **OEM 866** (Notepad++ поддерживает эту кодировку).
- Запустите talk-emma.bat для старта консоли talk-llama

### Tweaks for 6 and 8 GB vram / Твики для 6 и 8 GB vram
- use CPU instead of GPU, it will be a bit slower (5-6 s): in talk-llama find and change ngl to `-ngl 0` (find best speed)
- set smaller context for llama: `--ctx_size 512`
- set `--lowvram` in xtts_start.bat, that will move xtts model from GPU to RAM after each xtts request (but it will be slower)

### Optional
- Помещайте новые голоса XTTS в папку `\xtts\speakers\`. Рекомендуется использовать монофонические WAV-файлы с глубиной 16 бит, частотой дискретизации 22050 Гц и длительностью около 10 секунд, без шумов и музыки.
С опцией командной строки `--multi-chars` будет передавать имя нового персонажа в XTTS, даже если этот персонаж не указан ни в .bat-файле, ни в начальном промпте. Если XTTS не найдёт соответствующий голос — будет использован голос по умолчанию.
- Поместите описание персонажа и несколько примеров его реплик в файл talk_emma_inst.txt. 
- Переименуйте имя персонажа для .wav-файла в папке c:\DATA\xtts\speakers\. Вы также можете создать копии аудио с разными именами (например, Алиса или Олег). Теперь вы сможете обращаться к ним по имени.


#### Optional, google search plugin
- download [search_server.py]
- install langchain: `pip install langchain`
- sign up at https://serper.dev/api-key it is free and fast, it will give you 2500 free searches. Get an API key, paste it to search_server.py at line 13 `os.environ["SERPER_API_KEY"] = "your_key"`
- start search server by double clicking search_server.py. Now you can use voice commands like these: `Please google who is Barack Obama` or `Пожалуйста погугли погоду в Москве`.


## Building, optional
- for nvidia and Windows. Other systems - try yourself.
- download https://www.libsdl.org/release/ https://www.libsdl.org/release/SDL2-2.32.10-win32-x64.zip from extract to /whispertalk-llama.cpp/SDL2/ folder
- install libcurl using vcpkg:
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

Компиляция может длиться более 10 мин в зависимости от вашего компьютерного железа.
```
## talk-llama.exe params / Параметры командной строки для bat файла
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

## Voice commands / Голосовые команды:
Полный список команд и их вариаций находится в `talk-llama.cpp`, search `user_command`.
- Stop (остановись, Ctrl+Space)
- Regenerate (переделай, , Ctrl+Right) - will regenerate llama answer
- Delete (удали, Ctrl+Delete) - will delete user question and llama answer.
- Delete 3 messages (удали 3 сообщениия)
- Reset (удали все, Ctrl+R) - will delete all context except for a initial prompt
- Google something (погугли что-то)
- Сall NAME (позови Алису)

## Licenses / Лицензии
- talk-llama-fast - MIT License - OK for commercial use
- whisper.cpp - MIT License - OK for commercial use
- whisper - MIT License - OK for commercial use
- TTS(xtts) - Mozilla Public License 2.0 - OK for commercial use
- xtts-api-server - MIT License - OK for commercial use