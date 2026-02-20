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

<<<<<<< HEAD
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
=======
## Memory usage

| Model  | Disk    | Mem     |
| ------ | ------- | ------- |
| tiny   | 75 MiB  | ~273 MB |
| base   | 142 MiB | ~388 MB |
| small  | 466 MiB | ~852 MB |
| medium | 1.5 GiB | ~2.1 GB |
| large  | 2.9 GiB | ~3.9 GB |

## POWER VSX Intrinsics

`whisper.cpp` supports POWER architectures and includes code which
significantly speeds operation on Linux running on POWER9/10, making it
capable of faster-than-realtime transcription on underclocked Raptor
Talos II. Ensure you have a BLAS package installed, and replace the
standard cmake setup with:

```bash
# build with GGML_BLAS defined
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
./build/bin/whisper-cli [ .. etc .. ]
```

## Quantization

`whisper.cpp` supports integer quantization of the Whisper `ggml` models.
Quantized models require less memory and disk space and depending on the hardware can be processed more efficiently.

Here are the steps for creating and using a quantized model:

```bash
# quantize a model with Q5_0 method
cmake -B build
cmake --build build -j --config Release
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q5_0.bin q5_0

# run the examples as usual, specifying the quantized model file
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin ./samples/gb0.wav
```

## Core ML support

On Apple Silicon devices, the Encoder inference can be executed on the Apple Neural Engine (ANE) via Core ML. This can result in significant
speed-up - more than x3 faster compared with CPU-only execution. Here are the instructions for generating a Core ML model and using it with `whisper.cpp`:

- Install Python dependencies needed for the creation of the Core ML model:

  ```bash
  pip install ane_transformers
  pip install openai-whisper
  pip install coremltools
  ```

  - To ensure `coremltools` operates correctly, please confirm that [Xcode](https://developer.apple.com/xcode/) is installed and execute `xcode-select --install` to install the command-line tools.
  - Python 3.11 is recommended.
  - MacOS Sonoma (version 14) or newer is recommended, as older versions of MacOS might experience issues with transcription hallucination.
  - [OPTIONAL] It is recommended to utilize a Python version management system, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this step:
    - To create an environment, use: `conda create -n py311-whisper python=3.11 -y`
    - To activate the environment, use: `conda activate py311-whisper`

- Generate a Core ML model. For example, to generate a `base.en` model, use:

  ```bash
  ./models/generate-coreml-model.sh base.en
  ```

  This will generate the folder `models/ggml-base.en-encoder.mlmodelc`

- Build `whisper.cpp` with Core ML support:

  ```bash
  # using CMake
  cmake -B build -DWHISPER_COREML=1
  cmake --build build -j --config Release
  ```

- Run the examples as usual. For example:

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_init_state: loading Core ML model from 'models/ggml-base.en-encoder.mlmodelc'
  whisper_init_state: first run on a device may take a while ...
  whisper_init_state: Core ML model loaded

  system_info: n_threads = 4 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 | COREML = 1 |

  ...
  ```

  The first run on a device is slow, since the ANE service compiles the Core ML model to some device-specific format.
  Next runs are faster.

For more information about the Core ML implementation please refer to PR [#566](https://github.com/ggml-org/whisper.cpp/pull/566).

## OpenVINO support

On platforms that support [OpenVINO](https://github.com/openvinotoolkit/openvino), the Encoder inference can be executed
on OpenVINO-supported devices including x86 CPUs and Intel GPUs (integrated & discrete).

This can result in significant speedup in encoder performance. Here are the instructions for generating the OpenVINO model and using it with `whisper.cpp`:

- First, setup python virtual env. and install python dependencies. Python 3.10 is recommended.

  Windows:

  ```powershell
  cd models
  python -m venv openvino_conv_env
  openvino_conv_env\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

  Linux and macOS:

  ```bash
  cd models
  python3 -m venv openvino_conv_env
  source openvino_conv_env/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

- Generate an OpenVINO encoder model. For example, to generate a `base.en` model, use:

  ```
  python convert-whisper-to-openvino.py --model base.en
  ```

  This will produce ggml-base.en-encoder-openvino.xml/.bin IR model files. It's recommended to relocate these to the same folder as `ggml` models, as that
  is the default location that the OpenVINO extension will search at runtime.

- Build `whisper.cpp` with OpenVINO support:

  Download OpenVINO package from [release page](https://github.com/openvinotoolkit/openvino/releases). The recommended version to use is [2024.6.0](https://github.com/openvinotoolkit/openvino/releases/tag/2024.6.0). Ready to use Binaries of the required libraries can be found in the [OpenVino Archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/)

  After downloading & extracting package onto your development system, set up required environment by sourcing setupvars script. For example:

  Linux:

  ```bash
  source /path/to/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64/setupvars.sh
  ```

  Windows (cmd):

  ```powershell
  C:\Path\To\w_openvino_toolkit_windows_2023.0.0.10926.b4452d56304_x86_64\setupvars.bat
  ```

  And then build the project using cmake:

  ```bash
  cmake -B build -DWHISPER_OPENVINO=1
  cmake --build build -j --config Release
  ```

- Run the examples as usual. For example:

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_ctx_init_openvino_encoder: loading OpenVINO model from 'models/ggml-base.en-encoder-openvino.xml'
  whisper_ctx_init_openvino_encoder: first run on a device may take a while ...
  whisper_openvino_init: path_model = models/ggml-base.en-encoder-openvino.xml, device = GPU, cache_dir = models/ggml-base.en-encoder-openvino-cache
  whisper_ctx_init_openvino_encoder: OpenVINO model loaded

  system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | COREML = 0 | OPENVINO = 1 |

  ...
  ```

  The first time run on an OpenVINO device is slow, since the OpenVINO framework will compile the IR (Intermediate Representation) model to a device-specific 'blob'. This device-specific blob will get
  cached for the next run.

For more information about the OpenVINO implementation please refer to PR [#1037](https://github.com/ggml-org/whisper.cpp/pull/1037).

## NVIDIA GPU support

With NVIDIA cards the processing of the models is done efficiently on the GPU via cuBLAS and custom CUDA kernels.
First, make sure you have installed `cuda`: https://developer.nvidia.com/cuda-downloads

Now build `whisper.cpp` with CUDA support:

```
cmake -B build -DGGML_CUDA=1
cmake --build build -j --config Release
```

or for newer NVIDIA GPU's (RTX 5000 series):
```
cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build -j --config Release
```

## Vulkan GPU support
Cross-vendor solution which allows you to accelerate workload on your GPU.
First, make sure your graphics card driver provides support for Vulkan API.

Now build `whisper.cpp` with Vulkan support:
```
cmake -B build -DGGML_VULKAN=1
cmake --build build -j --config Release
```

## BLAS CPU support via OpenBLAS

Encoder processing can be accelerated on the CPU via OpenBLAS.
First, make sure you have installed `openblas`: https://www.openblas.net/

Now build `whisper.cpp` with OpenBLAS support:

```
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
```

## Ascend NPU support

Ascend NPU provides inference acceleration via [`CANN`](https://www.hiascend.com/en/software/cann) and AI cores.

First, check if your Ascend NPU device is supported:

**Verified devices**
| Ascend NPU                    | Status  |
|:-----------------------------:|:-------:|
| Atlas 300T A2                 | Support |
| Atlas 300I Duo                | Support |

Then, make sure you have installed [`CANN toolkit`](https://www.hiascend.com/en/software/cann/community) . The lasted version of CANN is recommanded.

Now build `whisper.cpp` with CANN support:

```
cmake -B build -DGGML_CANN=1
cmake --build build -j --config Release
```

Run the inference examples as usual, for example:

```
./build/bin/whisper-cli -f samples/jfk.wav -m models/ggml-base.en.bin -t 8
```

*Notes:*

- If you have trouble with Ascend NPU device, please create a issue with **[CANN]** prefix/tag.
- If you run successfully with your Ascend NPU device, please help update the table `Verified devices`.

## Moore Threads GPU support

With Moore Threads cards the processing of the models is done efficiently on the GPU via muBLAS and custom MUSA kernels.
First, make sure you have installed `MUSA SDK rc4.2.0`: https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=4.2.0

Now build `whisper.cpp` with MUSA support:

```
cmake -B build -DGGML_MUSA=1
cmake --build build -j --config Release
```

or specify the architecture for your Moore Threads GPU. For example, if you have a MTT S80 GPU, you can specify the architecture as follows:

```
cmake -B build -DGGML_MUSA=1 -DMUSA_ARCHITECTURES="21"
cmake --build build -j --config Release
```

## FFmpeg support (Linux only)

If you want to support more audio formats (such as Opus and AAC), you can turn on the `WHISPER_FFMPEG` build flag to enable FFmpeg integration.

First, you need to install required libraries:

```bash
# Debian/Ubuntu
sudo apt install libavcodec-dev libavformat-dev libavutil-dev

# RHEL/Fedora
sudo dnf install libavcodec-free-devel libavformat-free-devel libavutil-free-devel
```

Then you can build the project as follows:

```bash
cmake -B build -D WHISPER_FFMPEG=yes
cmake --build build
```

Run the following example to confirm it's working:

```bash
# Convert an audio file to Opus format
ffmpeg -i samples/jfk.wav jfk.opus

# Transcribe the audio file
./build/bin/whisper-cli --model models/ggml-base.en.bin --file jfk.opus
```

## Docker

### Prerequisites

- Docker must be installed and running on your system.
- Create a folder to store big models & intermediate files (ex. /whisper/models)

### Images

We have multiple Docker images available for this project:

1. `ghcr.io/ggml-org/whisper.cpp:main`: This image includes the main executable file as well as `curl` and `ffmpeg`. (platforms: `linux/amd64`, `linux/arm64`)
2. `ghcr.io/ggml-org/whisper.cpp:main-cuda`: Same as `main` but compiled with CUDA support. (platforms: `linux/amd64`)
3. `ghcr.io/ggml-org/whisper.cpp:main-musa`: Same as `main` but compiled with MUSA support. (platforms: `linux/amd64`)
4. `ghcr.io/ggml-org/whisper.cpp:main-vulkan`: Same as `main` but compiled with Vulkan support. (platforms: `linux/amd64`)

### Usage

```shell
# download model and persist it in a local folder
docker run -it --rm \
  -v path/to/models:/models \
  whisper.cpp:main "./models/download-ggml-model.sh base /models"

# transcribe an audio file
docker run -it --rm \
  -v path/to/models:/models \
  -v path/to/audios:/audios \
  whisper.cpp:main "whisper-cli -m /models/ggml-base.bin -f /audios/jfk.wav"

# transcribe an audio file in samples folder
docker run -it --rm \
  -v path/to/models:/models \
  whisper.cpp:main "whisper-cli -m /models/ggml-base.bin -f ./samples/jfk.wav"

# run the web server
docker run -it --rm -p "8080:8080" \
  -v path/to/models:/models \
  whisper.cpp:main "whisper-server --host 127.0.0.1 -m /models/ggml-base.bin"
  
# run the bench too on the small.en model using 4 threads
docker run -it --rm \
  -v path/to/models:/models \
  whisper.cpp:main "whisper-bench -m /models/ggml-small.en.bin -t 4"
```

## Installing with Conan

You can install pre-built binaries for whisper.cpp or build it from source using [Conan](https://conan.io/). Use the following command:

```
conan install --requires="whisper-cpp/[*]" --build=missing
```

For detailed instructions on how to use Conan, please refer to the [Conan documentation](https://docs.conan.io/2/).

## Limitations

- Inference only

## Real-time audio input example

This is a naive example of performing real-time inference on audio from your microphone.
The [stream](examples/stream) tool samples the audio every half a second and runs the transcription continuously.
More info is available in [issue #10](https://github.com/ggml-org/whisper.cpp/issues/10).
You will need to have [sdl2](https://wiki.libsdl.org/SDL2/Installation) installed for it to work properly.

```bash
cmake -B build -DWHISPER_SDL2=ON
cmake --build build -j --config Release
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## Confidence color-coding

Adding the `--print-colors` argument will print the transcribed text using an experimental color coding strategy
to highlight words with high or low confidence:

```bash
./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/gb0.wav --print-colors
```

<img width="965" alt="image" src="https://user-images.githubusercontent.com/1991296/197356445-311c8643-9397-4e5e-b46e-0b4b4daa2530.png">

## Controlling the length of the generated text segments (experimental)

For example, to limit the line length to a maximum of 16 characters, simply add `-ml 16`:

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 16

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.850]   And so my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:04.140]   Americans, ask
[00:00:04.140 --> 00:00:05.660]   not what your
[00:00:05.660 --> 00:00:06.840]   country can do
[00:00:06.840 --> 00:00:08.430]   for you, ask
[00:00:08.430 --> 00:00:09.440]   what you can do
[00:00:09.440 --> 00:00:10.020]   for your
[00:00:10.020 --> 00:00:11.000]   country.
```

## Word-level timestamp (experimental)

The `--max-len` argument can be used to obtain word-level timestamps. Simply use `-ml 1`:

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 1

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.320]
[00:00:00.320 --> 00:00:00.370]   And
[00:00:00.370 --> 00:00:00.690]   so
[00:00:00.690 --> 00:00:00.850]   my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:02.850]   Americans
[00:00:02.850 --> 00:00:03.300]  ,
[00:00:03.300 --> 00:00:04.140]   ask
[00:00:04.140 --> 00:00:04.990]   not
[00:00:04.990 --> 00:00:05.410]   what
[00:00:05.410 --> 00:00:05.660]   your
[00:00:05.660 --> 00:00:06.260]   country
[00:00:06.260 --> 00:00:06.600]   can
[00:00:06.600 --> 00:00:06.840]   do
[00:00:06.840 --> 00:00:07.010]   for
[00:00:07.010 --> 00:00:08.170]   you
[00:00:08.170 --> 00:00:08.190]  ,
[00:00:08.190 --> 00:00:08.430]   ask
[00:00:08.430 --> 00:00:08.910]   what
[00:00:08.910 --> 00:00:09.040]   you
[00:00:09.040 --> 00:00:09.320]   can
[00:00:09.320 --> 00:00:09.440]   do
[00:00:09.440 --> 00:00:09.760]   for
[00:00:09.760 --> 00:00:10.020]   your
[00:00:10.020 --> 00:00:10.510]   country
[00:00:10.510 --> 00:00:11.000]  .
```

## Speaker segmentation via tinydiarize (experimental)

More information about this approach is available here: https://github.com/ggml-org/whisper.cpp/pull/1058

Sample usage:

```py
# download a tinydiarize compatible model
./models/download-ggml-model.sh small.en-tdrz

# run as usual, adding the "-tdrz" command-line argument
./build/bin/whisper-cli -f ./samples/a13.wav -m ./models/ggml-small.en-tdrz.bin -tdrz
...
main: processing './samples/a13.wav' (480000 samples, 30.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, tdrz = 1, timestamps = 1 ...
...
[00:00:00.000 --> 00:00:03.800]   Okay Houston, we've had a problem here. [SPEAKER_TURN]
[00:00:03.800 --> 00:00:06.200]   This is Houston. Say again please. [SPEAKER_TURN]
[00:00:06.200 --> 00:00:08.260]   Uh Houston we've had a problem.
[00:00:08.260 --> 00:00:11.320]   We've had a main beam up on a volt. [SPEAKER_TURN]
[00:00:11.320 --> 00:00:13.820]   Roger main beam interval. [SPEAKER_TURN]
[00:00:13.820 --> 00:00:15.100]   Uh uh [SPEAKER_TURN]
[00:00:15.100 --> 00:00:18.020]   So okay stand, by thirteen we're looking at it. [SPEAKER_TURN]
[00:00:18.020 --> 00:00:25.740]   Okay uh right now uh Houston the uh voltage is uh is looking good um.
[00:00:27.620 --> 00:00:29.940]   And we had a a pretty large bank or so.
```

## Karaoke-style movie generation (experimental)

The [whisper-cli](examples/cli) example provides support for output of karaoke-style movies, where the
currently pronounced word is highlighted. Use the `-owts` argument and run the generated bash script.
This requires to have `ffmpeg` installed.

Here are a few _"typical"_ examples:

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -owts
source ./samples/jfk.wav.wts
ffplay ./samples/jfk.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337465-dbee4b5e-9aeb-48a3-b1c6-323ac4db5b2c.mp4

---

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/mm0.wav -owts
source ./samples/mm0.wav.wts
ffplay ./samples/mm0.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337504-cc8fd233-0cb7-4920-95f9-4227de3570aa.mp4

---

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/gb0.wav -owts
source ./samples/gb0.wav.wts
ffplay ./samples/gb0.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337538-b7b0c7a3-2753-4a88-a0cd-f28a317987ba.mp4

---

## Video comparison of different models

Use the [scripts/bench-wts.sh](https://github.com/ggml-org/whisper.cpp/blob/master/scripts/bench-wts.sh) script to generate a video in the following format:

```bash
./scripts/bench-wts.sh samples/jfk.wav
ffplay ./samples/jfk.wav.all.mp4
```

https://user-images.githubusercontent.com/1991296/223206245-2d36d903-cf8e-4f09-8c3b-eb9f9c39d6fc.mp4

---

## Benchmarks

In order to have an objective comparison of the performance of the inference across different system configurations,
use the [whisper-bench](examples/bench) tool. The tool simply runs the Encoder part of the model and prints how much time it
took to execute it. The results are summarized in the following Github issue:

[Benchmark results](https://github.com/ggml-org/whisper.cpp/issues/89)

Additionally a script to run whisper.cpp with different models and audio files is provided [bench.py](scripts/bench.py).

You can run it with the following command, by default it will run against any standard model in the models folder.

```bash
python3 scripts/bench.py -f samples/jfk.wav -t 2,4,8 -p 1,2
```

It is written in python with the intention of being easy to modify and extend for your benchmarking use case.

It outputs a csv file with the results of the benchmarking.

## `ggml` format

The original models are converted to a custom binary format. This allows to pack everything needed into a single file:

- model parameters
- mel filters
- vocabulary
- weights

You can download the converted models using the [models/download-ggml-model.sh](models/download-ggml-model.sh) script
or manually from here:

- https://huggingface.co/ggerganov/whisper.cpp

For more details, see the conversion script [models/convert-pt-to-ggml.py](models/convert-pt-to-ggml.py) or [models/README.md](models/README.md).

## [Bindings](https://github.com/ggml-org/whisper.cpp/discussions/categories/bindings)

- [x] Rust: [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) | [#310](https://github.com/ggml-org/whisper.cpp/discussions/310)
- [x] JavaScript: [bindings/javascript](bindings/javascript) | [#309](https://github.com/ggml-org/whisper.cpp/discussions/309)
  - React Native (iOS / Android): [whisper.rn](https://github.com/mybigday/whisper.rn)
- [x] Go: [bindings/go](bindings/go) | [#312](https://github.com/ggml-org/whisper.cpp/discussions/312)
- [x] Java:
  - [GiviMAD/whisper-jni](https://github.com/GiviMAD/whisper-jni)
- [x] Ruby: [bindings/ruby](bindings/ruby) | [#507](https://github.com/ggml-org/whisper.cpp/discussions/507)
- [x] Objective-C / Swift: [ggml-org/whisper.spm](https://github.com/ggml-org/whisper.spm) | [#313](https://github.com/ggml-org/whisper.cpp/discussions/313)
  - [exPHAT/SwiftWhisper](https://github.com/exPHAT/SwiftWhisper)
- [x] .NET: | [#422](https://github.com/ggml-org/whisper.cpp/discussions/422)
  - [sandrohanea/whisper.net](https://github.com/sandrohanea/whisper.net)
  - [NickDarvey/whisper](https://github.com/NickDarvey/whisper)
- [x] Python: | [#9](https://github.com/ggml-org/whisper.cpp/issues/9)
  - [stlukey/whispercpp.py](https://github.com/stlukey/whispercpp.py) (Cython)
  - [AIWintermuteAI/whispercpp](https://github.com/AIWintermuteAI/whispercpp) (Updated fork of aarnphm/whispercpp)
  - [aarnphm/whispercpp](https://github.com/aarnphm/whispercpp) (Pybind11)
  - [abdeladim-s/pywhispercpp](https://github.com/abdeladim-s/pywhispercpp) (Pybind11)
- [x] R: [bnosac/audio.whisper](https://github.com/bnosac/audio.whisper)
- [x] Unity: [macoron/whisper.unity](https://github.com/Macoron/whisper.unity)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example, the v1.7.5 version of the XCFramework can be
used as follows:

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Whisper",
    targets: [
        .executableTarget(
            name: "Whisper",
            dependencies: [
                "WhisperFramework"
            ]),
        .binaryTarget(
            name: "WhisperFramework",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/v1.7.5/whisper-v1.7.5-xcframework.zip",
            checksum: "c7faeb328620d6012e130f3d705c51a6ea6c995605f2df50f6e1ad68c59c6c4a"
        )
    ]
)
```

## Voice Activity Detection (VAD)
Support for Voice Activity Detection (VAD) can be enabled using the `--vad`
argument to `whisper-cli`. In addition to this option a VAD model is also
required.

The way this works is that first the audio samples are passed through
the VAD model which will detect speech segments. Using this information the
only the speech segments that are detected are extracted from the original audio
input and passed to whisper for processing. This reduces the amount of audio
data that needs to be processed by whisper and can significantly speed up the
transcription process.

The following VAD models are currently supported:

### Silero-VAD
[Silero-vad](https://github.com/snakers4/silero-vad) is a lightweight VAD model
written in Python that is fast and accurate.

Models can be downloaded by running the following command on Linux or MacOS:
```console
$ ./models/download-vad-model.sh silero-v6.2.0
Downloading ggml model silero-v6.2.0 from 'https://huggingface.co/ggml-org/whisper-vad' ...
ggml-silero-v6.2.0.bin        100%[==============================================>] 864.35K  --.-KB/s    in 0.04s
Done! Model 'silero-v6.2.0' saved in '/path/models/ggml-silero-v6.2.0.bin'
You can now use it like this:

  $ ./build/bin/whisper-cli -vm /path/models/ggml-silero-v6.2.0.bin --vad -f samples/jfk.wav -m models/ggml-base.en.bin

```
And the following command on Windows:
```console
> .\models\download-vad-model.cmd silero-v6.2.0
Downloading vad model silero-v6.2.0...
Done! Model silero-v6.2.0 saved in C:\Users\danie\work\ai\whisper.cpp\ggml-silero-v6.2.0.bin
You can now use it like this:

C:\path\build\bin\Release\whisper-cli.exe -vm C:\path\ggml-silero-v6.2.0.bin --vad -m models/ggml-base.en.bin -f samples\jfk.wav

```

To see a list of all available models, run the above commands without any
arguments.

This model can be also be converted manually to ggml using the following command:
```console
$ python3 -m venv venv && source venv/bin/activate
$ (venv) pip install silero-vad
$ (venv) $ python models/convert-silero-vad-to-ggml.py --output models/silero.bin
Saving GGML Silero-VAD model to models/silero-v6.2.0-ggml.bin
```
And it can then be used with whisper as follows:
```console
$ ./build/bin/whisper-cli \
   --file ./samples/jfk.wav \
   --model ./models/ggml-base.en.bin \
   --vad \
   --vad-model ./models/silero-v6.2.0-ggml.bin
```

### VAD Options

* --vad-threshold: Threshold probability for speech detection. A probability
for a speech segment/frame above this threshold will be considered as speech.

* --vad-min-speech-duration-ms: Minimum speech duration in milliseconds. Speech
segments shorter than this value will be discarded to filter out brief noise or
false positives.

* --vad-min-silence-duration-ms: Minimum silence duration in milliseconds. Silence
periods must be at least this long to end a speech segment. Shorter silence
periods will be ignored and included as part of the speech.

* --vad-max-speech-duration-s: Maximum speech duration in seconds. Speech segments
longer than this will be automatically split into multiple segments at silence
points exceeding 98ms to prevent excessively long segments.

* --vad-speech-pad-ms: Speech padding in milliseconds. Adds this amount of padding
before and after each detected speech segment to avoid cutting off speech edges.

* --vad-samples-overlap: Amount of audio to extend from each speech segment into
the next one, in seconds (e.g., 0.10 = 100ms overlap). This ensures speech isn't
cut off abruptly between segments when they're concatenated together.

## Examples

There are various examples of using the library for different projects in the [examples](examples) folder.
Some of the examples are even ported to run in the browser using WebAssembly. Check them out!

| Example                                             | Web                                   | Description                                                                                                                     |
| --------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [whisper-cli](examples/cli)                         | [whisper.wasm](examples/whisper.wasm) | Tool for translating and transcribing audio using Whisper                                                                       |
| [whisper-bench](examples/bench)                     | [bench.wasm](examples/bench.wasm)     | Benchmark the performance of Whisper on your machine                                                                            |
| [whisper-stream](examples/stream)                   | [stream.wasm](examples/stream.wasm)   | Real-time transcription of raw microphone capture                                                                               |
| [whisper-command](examples/command)                 | [command.wasm](examples/command.wasm) | Basic voice assistant example for receiving voice commands from the mic                                                         |
| [whisper-server](examples/server)                   |                                       | HTTP transcription server with OAI-like API                                                                                     |
| [whisper-talk-llama](examples/talk-llama)           |                                       | Talk with a LLaMA bot                                                                                                           |
| [whisper.objc](examples/whisper.objc)               |                                       | iOS mobile application using whisper.cpp                                                                                        |
| [whisper.swiftui](examples/whisper.swiftui)         |                                       | SwiftUI iOS / macOS application using whisper.cpp                                                                               |
| [whisper.android](examples/whisper.android)         |                                       | Android mobile application using whisper.cpp                                                                                    |
| [whisper.nvim](examples/whisper.nvim)               |                                       | Speech-to-text plugin for Neovim                                                                                                |
| [generate-karaoke.sh](examples/generate-karaoke.sh) |                                       | Helper script to easily [generate a karaoke video](https://youtu.be/uj7hVta4blM) of raw audio capture                           |
| [livestream.sh](examples/livestream.sh)             |                                       | [Livestream audio transcription](https://github.com/ggml-org/whisper.cpp/issues/185)                                            |
| [yt-wsp.sh](examples/yt-wsp.sh)                     |                                       | Download + transcribe and/or translate any VOD [(original)](https://gist.github.com/DaniruKun/96f763ec1a037cc92fe1a059b643b818) |
| [wchess](examples/wchess)                           | [wchess.wasm](examples/wchess)        | Voice-controlled chess                                                                                                          |

## [Discussions](https://github.com/ggml-org/whisper.cpp/discussions)

If you have any kind of feedback about this project feel free to use the Discussions section and open a new topic.
You can use the [Show and tell](https://github.com/ggml-org/whisper.cpp/discussions/categories/show-and-tell) category
to share your own projects that use `whisper.cpp`. If you have a question, make sure to check the
[Frequently asked questions (#126)](https://github.com/ggml-org/whisper.cpp/discussions/126) discussion.
>>>>>>> master
