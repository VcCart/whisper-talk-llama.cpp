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
- [2025.11.01] Все изменения из https://github.com/Mozer/talk-llama-fast/
Stable: [v1.9.2](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.2) / [Roadmap](https://github.com/orgs/ggml-org/projects/4/)


## Заметки
   -  В talk-llama.cpp был изменен сдвиг контекста под whisper.cpp > 1.8.0., и изменена работа с кэшем. В версии 1.76 это реализовывалось по другому.
   -  Диалог с talk-llama может вестись почти бесконечно — модель остаётся адекватной, серьёзных зацикливаний или повсеместных проблем не наблюдается. 
   -  Llama запоминает начальный промпт и последние N токенов контекста, но всё, что находится между ними, теряется. 
   -  Дополнительная видеопамять (VRAM), больше той что уже занята, при запуске не расходуется — вы можете вести практически бесконечный диалог без потери скорости.  
   -  talk-llama.cpp тестировался на llm модели saiga_yandexgpt_8b_Q4_K_S.gguf и Whisper модели whisper-ggml-large-v3-q4.bin
   -  В качестве тестовой видеокарты использовалась карта GTX1070 ti всего 8 ГБ на архитектуре Pascal. Лёгкую квантованную версию llama вполне нормально загружает.
   -  Далее была попытка запуска скомпилированных файлов на RTX 3060 12GB на архитектуре Ampere и файлы оказались несовместимы, так что проект придется перекомпилировать.
   -  Процессор желателен с AVX2 инструкциями, но и здесь можно обойти ограничение, скомпилировав проект без них; 
   -  XTTS можно запустить с флагом --lowvram или даже на CPU вместо GPU (-d=cpu, но это будет медленно), лучше сэкономить  GPU на llm, так как может llama приемлемо  работает на мощном CPU процессоре, а tts уже не справляется.
   -  Для использования с колонками (а не наушниками): Вы можете попробовать отключить прерывание речи бота из-за шума, установив --vad_start_thold 0.  
   -  Опционально: есть команда «пробуждения» — --wake-command "Эмма," (запятая после имени обязательна). Теперь только фразы, начинающиеся например, с имени «Эмма», будут отправляться в чат. Это частично поможет при работе с колонками или в шумном помещении, но лучше придумать как отключать микрофон вручную, или использовать наушники.

## Языки
Программа Мультиязычная, но зависит от подгруженных моделей Whisper и LLM.

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
Рекомендую использовать CUDA до версии 12.9x. Выше наблюдаются проблемы несовместимости.
- Загрузите [release](https://github.com/VcCart/whisper-talk-llama.cpp) или скомпилируйте самостоятельно. 
Распакуйте в папку c:\DATA\ .
- Загрузите модель whisper в папку c:\DATA\ с whisper-talk-llama.exe: Для Русского языка может подойти [ggml-large-v3-q4_k.bin](https://huggingface.co/adriabama06/whisper-large-v3-ggml) Или другой квантизации, в зависимости от объема VRAM.
- Загрузите LLM в ту же папку [saiga_yandexgpt_8b_Q4_K_S.gguf](https://huggingface.co/IlyaGusev/saiga_yandexgpt_8b_gguf/tree/main) Вы можете попробовать Q4_K_S или Q3_K_S, если у вас под llm запланировано мало VRAM.

Теперь установим xtts-api-server и TTS От Mozer (Ссылки на свои форки я поправлю позже, если опубликую). 
Примечание: XTTS с DeepSpeed требует PyTorch 2.1, cu118 или cu121, но некоторые пакеты DeepSpeed требуют PyTorch 2.2 и выше,
поэтому Depspeed придется компилировать или искать готовый whl
Все представленные здесь компоненты тестировались на Python 3.11 с разными версиями PyTorch. 
Установка окружения состоит в основном из:  Git, Python 3.11, XTTS сервера, langchain_community для google поиска и прочих модулей.

Для установки и запуска Xtts-Api-Server нужно Python окружение.
Подойдет Python 3.10 - 3.12 в зависимости от версии coqui-tts
В оригинальном Xtts-Api-Server используется coqui-tts 0.24.1
В Xtts-Api-Server от Mozer используется coqui-tts 0.22.0 c небольшими доработками.

Откройте папку \DATA\, куда вы положили основные файлы с talk-llama. В этой папке откройте командную строку (cmd) и выполняйте команды построчно:

```
git clone https://github.com/Mozer/xtts-api-server 
cd c:\DATA\
```
Переименуем xtts-api-server в xtts
```
ren xtts-api-server xtts
```
В папке xtts
```
cd c:\DATA\xtts\
```
Создайте окружение Python в той же папке:
```
python -m venv venv
```
Активируйте в windows так:
```
venv\Scripts\activate
```
Далее можно устанавливать:
```
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```
Запуск xtts сервера можно настроить в xtts_start.bat примерно такого содержания:
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
whisper-talk-llama.exe ^
  --model-whisper "ggml-large-v3-q4_k.bin" ^
  --model-llama "saiga_yandexgpt_8b_Q5_K.gguf" ^
  --language ru ^
  --person Друг ^
  --bot-name Эмма ^
  --xtts-voice Эмма ^
  --xtts-url http://localhost:8020/ ^
  --prompt-file "prompt_talk_emma_instruct.txt" ^
  --instruct-preset ChatML ^
  --temp 0.70 ^
  --top_k 40 ^
  --top_p 0.95 ^
  --repeat_penalty 1.15 ^
  --repeat_last_n 256 ^
  --n-gpu-layers -1 ^
  --threads 32 ^
  --batch-size 512 ^
  --ctx_size 4096 ^
  --n_predict 256 ^
  --min-tokens 20 ^
  --flash-attn ^
  --sleep-before-xtts 300 ^
  --vad-thold 0.006 ^
  --vad-start-thold 0.00027 ^
  --vad-last-ms 250
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

- Голоса персонажей хранятся в виде .wav-файлов в папке c:\DATA\xtts\speakers\. Вы также можете создать копии аудио с разными именами (например, Алиса или Олег). Теперь вы сможете обращаться к ним по имени.


#### Опционально плагин гугл поиска
- search_server.py - выложу позже после доработки.
- **Скачайте** [search_server.py]
- **Установите**: `pip install langchain`
- Зарегистрируйтесь на сайте https://serper.dev Сервис бесплатный и быстрый, предоставляет 2500 бесплатных поисковых запросов. Получите API-ключ и вставьте его в файл search_server.py на строке 13: `os.environ["SERPER_API_KEY"] = "your_key"`
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

Для процессоров без AVX2, надо будет билдить с ключом: -DWHISPER_NO_AVX2=1

Потом две команды по очистке и сборке, ниже (вводить по отдельности)

cmake.exe --build build -j --config release --target clean
cmake.exe --build build -j --config release --parallel 8

Компиляция может длиться около 10 мин и больше, в зависимости от вашего компьютерного железа.

```
## whisper-talk-llama.exe params / Параметры командной строки для bat файла
```
Базовые параметры
  -h,       --help                [default] показать это сообщение и выйти
  -t N,     --threads N           [4      ] количество потоков для вычислений
  -vms N,   --voice-ms N          [10000  ] длительность голоса в миллисекундах
  --interrupt-check-ms N          [200    ] как часто проверять микрофон во время генерации (мс)
  --interrupt-threshold-ms N      [250    ] сколько мс речи нужно для прерывания генерации
  -c ID,    --capture ID          [-1     ] ID устройства захвата звука
  -mt N,    --max-tokens N        [64     ] максимальное количество токенов на аудио-фрагмент
  -ac N,    --audio-ctx N         [0      ] размер аудио-контекста (0 - весь)
  -ngl N,   --n-gpu-layers N      [999    ] количество слоёв для хранения в VRAM
  -vth N,   --vad-thold N         [0.0005 ] порог обнаружения голосовой активности
  -vths N,  --vad-start-thold N   [0.0003 ] мин. уровень VAD для остановки TTS (0: выкл)
  -vlm N,   --vad-last-ms N       [1500.00] мин. тишина после речи для VAD, мс
  -fth N,   --freq-thold N        [90.00  ] частота среза высокочастотного фильтра
  -su,      --speed-up            [false  ] ускорить аудио в 2 раза (не работает)
  -tr,      --translate           [false  ] перевести с исходного языка на английский
  -ps,      --print-special       [false  ] печатать специальные токены
  -pe,      --print-energy        [false  ] печатать энергию звука (для отладки)
  --debug                         [false  ] печатать отладочную информацию
  -vp,      --verbose-prompt      [false  ] печатать промпт при запуске
  --verbose                       [false  ] печатать скорость
  -ng,      --no-gpu              [false  ] отключить GPU
  -fa,      --flash-attn          [false  ] использовать flash attention

Персонажи и промпты
  -p NAME,  --person NAME         [Друг   ] имя пользователя (для выбора промпта)
  -bn NAME, --bot-name NAME       [Эмма   ] имя бота (для отображения)
  -w TEXT,  --wake-command TEXT   [       ] команда пробуждения для прослушивания
  -ho TEXT, --heard-ok TEXT       [       ] текст, озвучиваемый TTS перед генерацией ответа
  -l LANG,  --language LANG       [ru     ] язык общения
  --prompt-file FNAME             [       ] файл с пользовательским промптом для начала диалога
  --instruct-preset TEXT          [       ] preset для инструкций (без .json)

Модели и файлы
  -mw FILE, --model-whisper       [whisper-ggml-medium-q4_0.bin]        файл модели Whisper
  -ml FILE, --model-llama         [saiga_yandexgpt_8b_Q4_K_S.gguf]      файл модели LLaMA
  --session FNAME                 [       ] файл для кэширования состояния модели
  -f FNAME, --file FNAME          [       ] имя файла для вывода текста

Параметры генерации LLaMA
  --ctx_size N                    [2048   ] размер контекста промпта
  -b N,     --batch-size N        [64     ] размер входного батча
  -n N,     --n_predict N         [64     ] максимальное количество токенов для предсказания
  --temp N                        [0.90   ] температура
  --top_k N                       [40.00  ] top_k
  --top_p N                       [1.00   ] top_p
  --min_p N                       [0.00   ] min_p
  --repeat_penalty N              [1.10   ] штраф за повторения
  --repeat_last_n N               [256    ] количество последних токенов для штрафа
  --n_keep N                      [128    ] сохранять первые N токенов после сдвига контекста
  --min-tokens N                  [0      ] минимальное количество новых токенов на вывод
  --stop-words TEXT               [       ] стоп-слова LLaMA (разделяются ;)

GPU и распределение
  --main-gpu N                    [0      ] ID основной GPU (начиная с 0)
  --split-mode NAME               [none   ] режим разделения GPU: 'none' или 'layer'
  --tensor-split LIST             [       ] разделение тензоров (список float: 0.5,0.5)

XTTS (озвучка)
  -s FILE,  --speak TEXT          [speak  ] команда для TTS
  -sf FILE, --speak-file          [to_speak.txt] файл для передачи в TTS
  --xtts-voice NAME               [Emma   ] голос XTTS (без расширения .wav)
  --xtts-url TEXT                 [http://localhost:8020/]     URL сервера XTTS/Silero (с / на конце)
  --xtts-control-path FNAME       [xtts_play_allowed.txt]      больше не используется
  --xtts-intro                    [false  ] короткое случайное вступление XTTS (например "Хмм")
  --sleep-before-xtts N           [0      ] пауза перед XTTS после инференса LLaMA (мс)
  --allow-newline                 [false  ] разрешить новые строки в выводе LLaMA
  --multi-chars                   [false  ] XTTS использует имя из вывода LLaMA для голоса
  --seqrep                        [false  ] штраф за повторения последовательностей
  --split-after N                 [0      ] разделять текст для TTS после N токенов

Сеть и интеграции
  --google-url TEXT               [http://localhost:8003/]     URL сервера Google Search

Управление
  --push-to-talk                  [false  ] зажимать Alt для разговора
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
