// Talk with AI
// =============================================================================
// talk-llama.cpp - Голосовой ассистент на базе Whisper + LLaMA + XTTS
//
// 📌 ОПИСАНИЕ:
//   Программа реализует голосового ассистента с полным циклом:
//   1. VAD (Voice Activity Detection) - обнаружение речи в реальном времени
//   2. Whisper - распознавание речи в текст
//   3. LLaMA - генерация ответа на основе распознанного текста
//   4. XTTS - озвучивание сгенерированного ответа
//
// 🔄 ОСНОВНОЙ ЦИКЛ РАБОТЫ (упрощённая, но надёжная архитектура):
//   [Микрофон] → [VAD анализ] → [Whisper] → [Обработка команд] → [LLaMA] → [XTTS]
//                      ↓               ↓               ↓               ↓
//                 управление       распознан.     спец.команды     озвучка
//                 TTS через файл    текст           (call,google)    ответа
//
// 🚦 УПРАВЛЕНИЕ TTS ЧЕРЕЗ ФАЙЛ:
//   При начале речи:   xtts_play_allowed.txt = 0 (блокировка)
//   При тишине:        xtts_play_allowed.txt = 1 (разрешение)
//   XTTS сервер читает этот файл перед воспроизведением
//
// ⚡ ВАЖНО:
//   - Все отладочные выводы (>>>) появляются ТОЛЬКО при --verbose
//   - В обычном режиме только диалог и ошибки
//   - VAD использует сглаживание для устойчивости к шумам
//
// =============================================================================
// 1. ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
// -----------------------------------------------------------------------------
// 1.1 Внешние библиотеки ИИ (Whisper и LLaMA)
// -----------------------------------------------------------------------------
#include "common-sdl.h"        // Общие функции SDL для работы с аудио
#include "common.h"            // Общие вспомогательные функции проекта
#include "common-whisper.h"    // Общие функции для интеграции с Whisper
#include "whisper.h"           // Основная библиотека Whisper для распознавания речи
#include "llama.h"             // Основная библиотека LLaMA для генерации текста
// СИСТЕМНЫЕ БИБЛИОТЕКИ C++
#include <chrono>              // Работа со временем и таймерами
#include <cstdio>              // Стандартный ввод/вывод C (printf, fprintf)
#include <cassert>             // Проверки при отладке (удаляются в релизе)
#include <fstream>             // Работа с файлами
#include <regex>               // Регулярные выражения для парсинга текста
#include <sstream>             // Строковые потоки для форматирования
#include <functional>          // Функциональные объекты и лямбда-выражения
#include <string>              // Строковый класс std::string
#include <thread>              // Многопоточность и управление потоками
#include <vector>              // Динамические массивы
#include <stdexcept>           // Исключения стандартной библиотеки
#include <mutex>               // Мьютексы для потокобезопасного доступа
#include <atomic>              // Атомарные переменные для синхронизации
#include <iostream>            // Стандартный ввод/вывод C++ (std::cin, std::cout)
#include <algorithm>           // Алгоритмы STL (sort, find, transform)
#include <cctype>              // Функции для работы с символами (isalpha, isspace)
#include <locale>              // Локализация и региональные настройки
#include <clocale>             // Управление локалью C
#include <codecvt>             // Преобразование между кодировками
#include <queue>               // Очереди FIFO
#include <unordered_set>       // Хэш-множества для быстрого поиска
#include <ctype.h>             // С-стиль функции для работы с символами
#include <map>                 // Ассоциативные массивы (ключ-значение)
#include <iterator>            // Итераторы STL
#include <ctime>               // Работа с системным временем
#include <filesystem>
#include <random>              // Современный генератор случайных чисел
// ПОЛЬЗОВАТЕЛЬСКИЕ МОДУЛИ
#include "console.h"           // Заголовочный файл консольных функций
#include "console.cpp"         // Реализация консольных функций
// СЕТЕВЫЕ БИБЛИОТЕКИ
#include <curl/curl.h>         // Библиотека libcurl для HTTP запросов
#include "json.hpp"            // Библиотека nlohmann/json для работы с JSON
// ЗАГОЛОВКИ ОС (Windows)
#ifdef _WIN32
#include <Windows.h>           // Windows API (работа с окнами, клавиатурой)
#endif

// ============================================================
// ФУНКЦИИ ДЛЯ ЦВЕТНОГО ВЫВОДА (кроссплатформенные)
// ============================================================
#ifdef _WIN32
static HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
static WORD originalColors = 0;

static void init_console_colors() {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hConsole, &csbi)) {
        originalColors = csbi.wAttributes;
    }
}

static void set_console_color(WORD color) {
    SetConsoleTextAttribute(hConsole, color);
}

static void reset_console_color() {
    SetConsoleTextAttribute(hConsole, originalColors);
}
#else
// Для Linux/macOS используем ANSI
#define set_console_color(x)
#define reset_console_color() printf("\033[0m")
static inline void init_console_colors() {}
#endif

// ГЛОБАЛЬНЫЕ МЬЮТЕКСЫ
std::atomic<bool> g_is_interrupted{false};          // Флаг прерывания для сетевых запросов (curl)
std::atomic<int>  llama_interrupted{0};             // Флаг прерывания генерации LLaMA (связь с озвучкой)
std::atomic<bool> g_shutting_down{false};

std::queue<std::string> input_queue;                // Очередь ввода с клавиатуры
std::mutex              input_mutex;                // Мьютекс для защиты input_queue

std::atomic<bool> keyboard_input_running{true};     // Флаг работы потока ввода с клавиатуры

std::string g_hotkey_pressed = "";                  // Последняя нажатая горячая клавиша
std::mutex  g_hotkey_pressed_mutex;                 // Мьютекс для защиты g_hotkey_pressed

std::mutex g_threads_mutex;                         // Мьютекс для защиты вектора потоков threads

std::string g_last_tts_text = "";                   // Последний текст, отправленный в TTS (для regenerate)
std::mutex  g_last_tts_mutex;                       // Мьютекс для защиты g_last_tts_text
std::atomic<bool> g_shortcut_thread_running{true};  // Флаг работы потока горячих клавиш
std::mutex  g_llama_mutex;                          // Мьютекс для защиты ctx_llama

// ФУНКЦИЯ ТОКЕНИЗАЦИИ ТЕКСТА
// Преобразует текст в последовательность токенов модели LLaMA

static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Начальный размер с запасом (как в старой версии)
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);

    // Токенизация с реальным буфером
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);

    if (n_tokens < 0) {
        // Если буфер мал, увеличиваем до нужного размера
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        if (check != -n_tokens) {
            fprintf(stderr, "Warning: token count mismatch after resize\n");
        }
    } else {
        // Обрезаем до реального размера
        result.resize(n_tokens);
    }

    return result;
}

// ФУНКЦИЯ ПРЕОБРАЗОВАНИЯ ТОКЕНА В СТРОКУ
// Преобразует токен обратно в текстовое представление
static std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<char> result(8); // Начальный буфер из 8 символов

    // Преобразуем токен в текст
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) { // Если буфер мал, увеличиваем его
        result.resize(-n_tokens);
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens); // Проверяем корректность
    } else {
        result.resize(n_tokens); // Устанавливаем точный размер
    }

    return std::string(result.data(), result.size());
}

/**
 * @brief Парсит строку с числами с плавающей точкой, разделёнными запятыми.
 *        Используется для параметров командной строки, таких как --tensor-split.
 */
std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> result;
    if (s.empty()) {
        std::cerr << "Error: Empty input string for float list." << std::endl;
        return result;
    }
    std::stringstream ss(s);
    std::string item;

    try {
        // Разделяем строку по запятым
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                // Удаляем лишние пробелы
                item.erase(0, item.find_first_not_of(' '));
                item.erase(item.find_last_not_of(' ') + 1);

                if (!item.empty()) {
                    // Преобразуем подстроку в float
                    result.push_back(std::stof(item));
                }
            }
        }

        // Проверка: если не нашли ни одного числа
        if (result.empty()) {
            std::cerr << "Warning: No valid float numbers found in string: '" << s << "'" << std::endl;
        }

    } catch (const std::exception& e) {
        // Если в строке не float или другая ошибка преобразования
        std::cerr << "Error parsing float list from '" << s << "': " << e.what() << '\n';
        result.clear();
    }

    return result;
}

// command-line parameters
struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms     = 10000;
    int32_t capture_id   = -1;
    int32_t max_tokens   = 64;
    int32_t audio_ctx    = 0;
    int32_t n_gpu_layers = 999;

	float vad_thold        = 0.6f;           //  VAD
    float vad_start_thold  = 0.000270f;       // Снижен для более быстрого определения начала речи
    float vad_last_ms      = 1250;           // Уменьшена пауза между фразами для русской речи
    float freq_thold       = 90.0f;

    bool speed_up       = false;
    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool debug          = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool verbose        = false;
    bool use_gpu        = true;
	bool flash_attn     = false;
    bool allow_newline  = false;
    bool multi_chars    = false;
    bool xtts_intro     = false;
    bool seqrep         = false;
    bool push_to_talk   = false;
    int split_after     = 0;
    int sleep_before_xtts = 0; // in ms
    int main_gpu = 0;
    // Параметры прерывания генерации
    int32_t interrupt_check_ms   = 200;    // Как часто проверять микрофон (мс)
    int32_t interrupt_threshold_ms = 250;   // Сколько мс речи нужно для прерывания
	std::string person      = "Друг";
    std::string bot_name    = "Эмма";
    std::string xtts_voice  = "Emma";
    std::string wake_cmd    = "";      // Команда пробуждения (например "Эмма,")
    std::string heard_ok    = "";
    std::string language    = "ru";
    std::string model_wsp   = "whisper-ggml-medium-q4_0.bin";
    std::string model_llama = "saiga_yandexgpt_8b_Q4_K_S.gguf";
    std::string speak       = "speak";
	std::string speak_file  = "to_speak.txt"; // not used
    std::string xtts_control_path = "xtts_play_allowed.txt";
    std::string xtts_url = "http://localhost:8020/";
    std::string google_url = "http://localhost:8003/";
    std::string prompt      = "";
	std::string instruct_preset = "";
	std::string split_mode = "none";
    std::vector<float> tensor_split;
	std::map<std::string, std::string> instruct_preset_data = {
		{"system_prompt_prefix", ""},
		{"system_prompt_suffix", ""},
		{"user_message_prefix", ""},
		{"user_message_suffix", ""},
		{"bot_message_prefix", ""},
		{"bot_message_suffix", ""},
		{"stop_sequence", ""}
	};
    std::string fname_out;
    std::string path_session = "";
    std::string stop_words = "";
    int32_t ctx_size = 2048;
    int32_t batch_size = 64;
    int32_t n_predict = 64;
    int32_t min_tokens = 0;
    float temp = 0.9;
    int32_t top_k = 40;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repeat_penalty = 1.10;
    int repeat_last_n = 256;
    int n_keep = 128;
    bool safe_context_shift = true;     // Включить расширенную защиту (всегда true в этом патче)
};

// ### ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ###
void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

// Улучшенный код: добавлены блоки try-catch для обработки ошибок, добавлены проверки на выход за пределы argv.
bool whisper_params_parse(int argc, char **argv, whisper_params &params) {

    params.tensor_split.clear();
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        try {
            if (arg == "-h" || arg == "--help") {
                whisper_print_usage(argc, argv, params);
                return false;
            }
            else if (arg == "-t" || arg == "--threads") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: missing value after " << arg << std::endl;
                    return false;
                }
                params.n_threads = std::stoi(argv[++i]);
            }

            else if (arg == "-vms" || arg == "--voice-ms") {
                params.voice_ms = std::stoi(argv[++i]);
            }
            else if (arg == "-c" || arg == "--capture") {
                params.capture_id = std::stoi(argv[++i]);
            }
            else if (arg == "--interrupt-check-ms") {
                params.interrupt_check_ms = std::stoi(argv[++i]);
            }
            else if (arg == "--interrupt-threshold-ms") {
                params.interrupt_threshold_ms = std::stoi(argv[++i]);
            }
            else if (arg == "-mt" || arg == "--max-tokens") {
                params.max_tokens = std::stoi(argv[++i]);
            }
            else if (arg == "-ac" || arg == "--audio-ctx") {
                params.audio_ctx = std::stoi(argv[++i]);
            }
            else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                params.n_gpu_layers = std::stoi(argv[++i]);
            }
            else if (arg == "-vth" || arg == "--vad-thold") {
                params.vad_thold = std::stof(argv[++i]);
            }
            else if (arg == "-vths" || arg == "--vad-start-thold") {
                params.vad_start_thold = std::stof(argv[++i]);
            }
            else if (arg == "-vlm" || arg == "--vad-last-ms") {
                params.vad_last_ms = std::stoi(argv[++i]);
            }
            else if (arg == "-fth" || arg == "--freq-thold") {
                params.freq_thold = std::stof(argv[++i]);
            }
            else if (arg == "-su" || arg == "--speed-up") {
                params.speed_up = true;
            }
            else if (arg == "-tr" || arg == "--translate") {
                params.translate = true;
            }
            else if (arg == "-ps" || arg == "--print-special") {
                params.print_special = true;
            }
            else if (arg == "-pe" || arg == "--print-energy") {
                params.print_energy = true;
            }
            else if (arg == "--debug") {
                params.debug = true;
            }
            else if (arg == "-vp" || arg == "--verbose-prompt") {
                params.verbose_prompt = true;
            }
            else if (arg == "--verbose") {
                params.verbose = true;
            }
            else if (arg == "-ng" || arg == "--no-gpu") {
                params.use_gpu = false;
            }
            else if (arg == "-fa" || arg == "--flash-attn") {
                params.flash_attn = true;
            }
            else if (arg == "-p" || arg == "--person") {
                params.person = argv[++i];
            }
            else if (arg == "-bn" || arg == "--bot-name") {
                params.bot_name = argv[++i];
            }
            else if (arg == "--session") {
                params.path_session = argv[++i];
            }
            else if (arg == "-w" || arg == "--wake-command") {
                params.wake_cmd = argv[++i];
            }
            else if (arg == "-ho" || arg == "--heard-ok") {
                params.heard_ok = argv[++i];
            }
            else if (arg == "-l" || arg == "--language") {
                params.language = argv[++i];
            }
            else if (arg == "-mw" || arg == "--model-whisper") {
                params.model_wsp = argv[++i];
            }
            else if (arg == "-ml" || arg == "--model-llama") {
                params.model_llama = argv[++i];
            }
            else if (arg == "-s" || arg == "--speak") {
                params.speak = argv[++i];
            }
            else if (arg == "-sf" || arg == "--speak-file") {
                params.speak_file = argv[++i];
            }
            else if (arg == "--ctx_size") {
                params.ctx_size = std::stoi(argv[++i]);
            }
            else if (arg == "-b" || arg == "--batch-size") {
                params.batch_size = std::stoi(argv[++i]);
            }
            else if (arg == "-n" || arg == "--n_predict") {
                params.n_predict = std::stoi(argv[++i]);
            }
            else if (arg == "--temp") {
                params.temp = std::stof(argv[++i]);
            }
            else if (arg == "--top_k") {
                params.top_k = std::stoi(argv[++i]);
            }
            else if (arg == "--top_p") {
                params.top_p = std::stof(argv[++i]);
            }
            else if (arg == "--min_p") {
                params.min_p = std::stof(argv[++i]);
            }
            else if (arg == "--repeat_penalty") {
                params.repeat_penalty = std::stof(argv[++i]);
            }
            else if (arg == "--repeat_last_n") {
                params.repeat_last_n = std::stoi(argv[++i]);
            }
            else if (arg == "--n_keep") {
                params.n_keep = std::stoi(argv[++i]);
            }
            else if (arg == "--main-gpu") {
                params.main_gpu = std::stoi(argv[++i]);
            }
            else if (arg == "--split-mode") {
                params.split_mode = argv[++i];
            }

            else if (arg == "--tensor-split") {
                // Безопасная обработка tensor-split аргумента
                if (i + 1 >= argc) {
                    std::cerr << "Error: missing value after " << arg << std::endl;
                    return false;
                }
                std::string tensor_split_str = argv[++i];

                // Проверка на пустую строку
                if (tensor_split_str.empty()) {
                    std::cerr << "Error: empty tensor-split list" << std::endl;
                    return false;
                }

                // Парсинг списка float
                params.tensor_split = parse_float_list(tensor_split_str);

                // Проверка результата парсинга
                if (params.tensor_split.empty()) {
                    std::cerr << "Error: failed to parse tensor-split list: '" << tensor_split_str << "'" << std::endl;
                    return false;
                }

                // Дополнительная валидация значений
                float sum = 0.0f;
                for (float val : params.tensor_split) {
                    if (val < 0.0f || val > 1.0f) {
                        std::cerr << "Error: tensor-split values must be between 0.0 and 1.0, got: " << val << std::endl;
                        return false;
                    }
                    sum += val;
                }

                // Проверка суммы (должна быть близка к 1.0 для распределения между GPU)
                if (fabs(sum - 1.0f) > 0.001f) {
                    std::cerr << "Warning: tensor-split values sum to " << sum << " (expected ~1.0)" << std::endl;
                }
            }

            else if (arg == "--xtts-voice") {
                params.xtts_voice = argv[++i];
            }
            else if (arg == "--xtts-url") {
                params.xtts_url = argv[++i];
            }
            else if (arg == "--google-url") {
                params.google_url = argv[++i];
            }
            else if (arg == "--xtts-control-path") {
                params.xtts_control_path = argv[++i];
            }
            else if (arg == "--allow-newline") {
                params.allow_newline = true;
            }
            else if (arg == "--multi-chars") {
                params.multi_chars = true;
            }
            else if (arg == "--xtts-intro") {
                params.xtts_intro = true;
            }
            else if (arg == "--sleep-before-xtts") {
                params.sleep_before_xtts = std::stoi(argv[++i]);
            }
            else if (arg == "--seqrep") {
                params.seqrep = true;
            }
            else if (arg == "--push-to-talk") {
                params.push_to_talk = true;
            }
            else if (arg == "--split-after") {
                params.split_after = std::stoi(argv[++i]);
            }
            else if (arg == "--min-tokens") {
                params.min_tokens = std::stoi(argv[++i]);
            }
            else if (arg == "--stop-words") {
                params.stop_words = argv[++i];
            }
            else if (arg == "--instruct-preset") {
                params.instruct_preset = argv[++i];
            }
            else if (arg == "--prompt-file") {
                if (i + 1 >= argc) { // Проверяем, есть ли аргумент после --prompt-file
                    std::cerr << "Error: --prompt-file requires a filename." << std::endl;
                    whisper_print_usage(argc, argv, params);
                    return false;
                }
                std::ifstream file(argv[++i]); // i увеличен ТОЛЬКО после проверки
                if (!file.is_open()) {
                    std::cerr << "Failed to open prompt file: " << argv[i] << std::endl;
                    return false; // завершаем работу при ошибке
                }
                std::copy(std::istreambuf_iterator<char>(file),
                          std::istreambuf_iterator<char>(),
                          std::back_inserter(params.prompt));
                if (!params.prompt.empty() && params.prompt.back() == '\n') {
                    params.prompt.pop_back();
                }
            }
            else if (arg == "-f" || arg == "--file") {
                params.fname_out = argv[++i];
            }
            else {
                fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
                whisper_print_usage(argc, argv, params);
                return false;
            }
        }
            catch (const std::exception &e) {
                std::cerr << "Error parsing argument: " << e.what() << std::endl;
                // очистка не нужна - вектор сам управляет памятью
                whisper_print_usage(argc, argv, params);
                return false;
            }
    }
    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  --interrupt-check-ms N     [%-7d] how often to check mic during generation (ms)\n", params.interrupt_check_ms);
    fprintf(stderr, "  --interrupt-threshold-ms N [%-7d] how much speech to interrupt generation (ms)\n", params.interrupt_threshold_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n",           params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice avg activity detection threshold\n",    params.vad_thold);
	fprintf(stderr, "  -vths N,  --vad-start-thold N [%-7.6f] vad min level to stop tts, 0: off, 0.000270: default\n",params.vad_start_thold);
    fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7.2f] vad min silence after speech, ms\n",       	 params.vad_last_ms);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up       [%-7s] speed up audio by x2 (not working)\n",          params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  --debug                    [%-7s] print debug info\n",                            params.debug ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n",                       params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  --verbose                  [%-7s] print speed\n",                                 params.verbose ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
	fprintf(stderr, "  -fa,      --flash-attn     [%-7s] flash attention\n",                             params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n",                       params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n",               params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n",         params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n",                            params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n",                             params.speak.c_str());
	fprintf(stderr, "  -sf FILE, --speak-file     [%-7s] file to pass to TTS\n",                         params.speak_file.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --instruct-preset TEXT     [%-7s] instruct preset to use without .json \n",     	 "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "   --ctx_size N              [%-7d] Size of the prompt context\n",                  params.ctx_size);
    fprintf(stderr, "  -b N,     --batch-size N   [%-7d] Size of input batch size\n",                    params.batch_size);
    fprintf(stderr, "  -n N,     --n_predict N    [%-7d] Max number of tokens to predict\n",             params.n_predict);
    fprintf(stderr, "  --temp N                   [%-7.2f] Temperature \n",                              params.temp);
    fprintf(stderr, "  --top_k N                  [%-7d] top_k \n",                                    params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p \n",                                    params.top_p);
    fprintf(stderr, "  --min_p N                  [%-7.2f] min_p \n",                                    params.min_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repeat_penalty \n",                           params.repeat_penalty);
    fprintf(stderr, "  --repeat_last_n N          [%-7d] repeat_last_n \n",                              params.repeat_last_n);
    fprintf(stderr, "  --n_keep N                 [%-7d] keep first n_tokens after context_shift \n",    params.n_keep);
    fprintf(stderr, "  --main-gpu N               [%-7d] main GPU id, starting from 0 \n",               params.main_gpu);
    fprintf(stderr, "  --split-mode NAME          [%-7s] GPU split mode: 'none' or 'layer'\n",           params.split_mode.c_str());
    fprintf(stderr, "  --tensor-split NAME        [    ] Tensor split, list of floats: 0.5,0.5\n"),
    fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n",                     params.xtts_voice.c_str());
    fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts/silero server URL, with trailing slash\n", params.xtts_url.c_str());
    fprintf(stderr, "  --xtts-control-path FNAME  [%-7s] not used anymore\n",                            params.xtts_control_path.c_str());
	fprintf(stderr, "  --xtts-intro               [%-7s] xtts instant short random intro like Hmmm.\n",  params.xtts_intro ? "true" : "false");
    fprintf(stderr, "  --sleep-before-xtts        [%-7d] sleep llama inference before xtts, ms.\n",      params.sleep_before_xtts);
    fprintf(stderr, "  --google-url TEXT          [%-7s] langchain google-serper server URL, with /\n",  params.google_url.c_str());
    fprintf(stderr, "  --allow-newline            [%-7s] allow new line in llama output\n",              params.allow_newline ? "true" : "false");
    fprintf(stderr, "  --multi-chars              [%-7s] xtts will use same wav name as in llama output\n", params.multi_chars ? "true" : "false");
    fprintf(stderr, "  --push-to-talk             [%-7s] hold Alt to speak\n",							 params.push_to_talk ? "true" : "false");
    fprintf(stderr, "  --seqrep                   [%-7s] sequence repetition penalty, search last 20 in 300\n",params.seqrep ? "true" : "false");
    fprintf(stderr, "  --split-after N            [%-7d] split after first n tokens for tts\n",          params.split_after);
    fprintf(stderr, "  --min-tokens N             [%-7d] min new tokens to output\n",                    params.min_tokens);
	fprintf(stderr, "  --stop-words TEXT          [%-7s] llama stop w: separated by ; \n",               params.stop_words.c_str());
    fprintf(stderr, "\n");
}

// ### ГЛОБАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ###

// Возвращает текущее время в секундах с точностью до миллисекунд
float get_current_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0f;
}

// -----------------------------------------------------------------------------
// @brief Потокобезопасное добавление задачи в вектор потоков
// @param threads_vec Вектор потоков
// @param task Функция для выполнения в потоке
// -----------------------------------------------------------------------------
static void safe_thread_emplace(std::vector<std::thread>& threads_vec,
                                 std::function<void()> task)
{
    // НЕ создаём новые потоки, если программа завершается
    if (g_shutting_down.load()) {
        return;
    }

    std::scoped_lock lock(g_threads_mutex);
    try {
        threads_vec.emplace_back(std::move(task));
    } catch (const std::exception& e) {
        std::cerr << "Ошибка создания потока: " << e.what() << std::endl;
    }
}

// -----------------------------------------------------------------------------
// @brief Функция транскрибации аудио с использованием Whisper
// @param ctx         Контекст Whisper
// @param params      Параметры транскрибации
// @param pcmf32      Аудиоданные в формате float32
// @param prompt_text Текст промпта
// @param prob        Средняя вероятность транскрипции (выходной параметр)
// @param t_ms        Время выполнения в миллисекундах (выходной параметр)
// @return Распознанный текст
// -----------------------------------------------------------------------------
static std::string transcribe(
    whisper_context* ctx,               // Контекст Whisper
    const whisper_params& params,       // Параметры транскрибации
    const std::vector<float>& pcmf32,   // Аудиоданные в формате float32
    const std::string& prompt_text,     // Текст промпта
    float& prob,                        // Средняя вероятность транскрипции
    int64_t& t_ms) {                    // Время выполнения в миллисекундах

    // Инициализация выходных параметров
    prob = 0.0f;
    t_ms = 0;

    // Проверка входных параметров
    if (!ctx) {
        std::cerr << "Ошибка: Контекст Whisper не инициализирован" << std::endl;
        return "";
    }

    if (pcmf32.empty()) {
        std::cerr << "Ошибка: Входные аудиоданные пусты" << std::endl;
        return "";
    }

    // Начало замера времени
    const auto t_start = std::chrono::high_resolution_clock::now();

    // Настройка параметров Whisper
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // ============================================================
    // ПЕРЕДАЧА ПРОМПТА В WHISPER (фикс галлюцинаций)
    // ============================================================
    // Промпт помогает модели понимать контекст: кто говорит, на каком языке,
    // какие фразы игнорировать. Без него Whisper "фантазирует" и повторяется.
    // ============================================================

    // Вектор для хранения токенов промпта.
    // ВАЖНО: должен жить всё время выполнения whisper_full(),
    // поэтому объявлен до присвоения wparams.prompt_tokens.
    std::vector<whisper_token> prompt_tokens_vec;

    if (!prompt_text.empty()) {
        // Токенизируем промпт
        prompt_tokens_vec.resize(prompt_text.size() + 1);
        int n_tokens = whisper_tokenize(ctx, prompt_text.c_str(),
                                         prompt_tokens_vec.data(),
                                         prompt_tokens_vec.size());
        if (n_tokens > 0) {
            prompt_tokens_vec.resize(n_tokens);
            wparams.prompt_tokens = prompt_tokens_vec.data();
            wparams.prompt_n_tokens = prompt_tokens_vec.size();
        } else {
            // Токенизация не удалась — отключаем промпт
            wparams.prompt_tokens = nullptr;
            wparams.prompt_n_tokens = 0;
        }
    } else {
        // Промпт пустой — отключаем
        wparams.prompt_tokens = nullptr;
        wparams.prompt_n_tokens = 0;
    }

    // Базовые параметры вывода
    wparams.print_progress = false;
    wparams.print_special = params.print_special;
    wparams.print_realtime = false;

    // ВАЖНО: no_timestamps управляет ВЫЧИСЛЕНИЕМ меток, а не только выводом
    // Вычисление меток резко повышает галлюцинации на тишине и шуме
    wparams.print_timestamps = !params.no_timestamps;
    wparams.no_timestamps    = params.no_timestamps;
    wparams.translate        = params.translate;

    // === ОПТИМАЛЬНЫЕ НАСТРОЙКИ ДЛЯ РАСПОЗНАВАНИЯ РЕЧИ ===
    // ИСПРАВЛЕНО: контекст ВКЛЮЧЕН для предотвращения повторов
    wparams.no_context       = false;  // ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: модель помнит что сказала
    wparams.single_segment   = false;  // ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: быстрее и стабильнее
    wparams.token_timestamps = false;  // Временные метки не нужны, экономим ресурсы

    // Подавление мусора и хезитаций
    wparams.suppress_blank   = true;   // Подавление "э-э-э", "ммм"
    wparams.suppress_nst     = true;   // Подавление мусорных токенов

    // Детерминированный режим с минимальной случайностью
    wparams.temperature      = 0.0f;   // Нулевая температура для стабильного результата
    wparams.temperature_inc  = 0.0f;   // Без повышения температуры
    wparams.length_penalty   = 0.0f;   // Без штрафа за длину
    wparams.entropy_thold    = 2.4f;   // ← ДОБАВЛЕНО: если модель неуверена → молчит
    wparams.max_len          = 0;      // ← ДОБАВЛЕНО: авто-ограничение длины

    // Настройка максимального количества токенов
    {
        int model_text_ctx = static_cast<int>(whisper_n_text_ctx(ctx));
        // Минимум 64 токена для русского, максимум — лимит модели
        int mt = (params.max_tokens > 0) ? params.max_tokens : 64;

        if (mt > model_text_ctx) {
            std::cerr << "Предупреждение: max_tokens (" << mt
                      << ") превышает лимит модели (" << model_text_ctx
                      << "), применяется лимит модели" << std::endl;
            mt = model_text_ctx;
        }
        wparams.max_tokens = mt;
    }

    // Настройка аудиоконтекста
    wparams.audio_ctx = params.audio_ctx;
    int model_audio_ctx = static_cast<int>(whisper_n_audio_ctx(ctx));

    if (wparams.audio_ctx > model_audio_ctx) {
        std::cerr << "Предупреждение: audio_ctx (" << wparams.audio_ctx
                  << ") превышает лимит модели (" << model_audio_ctx
                  << "), применяется лимит модели" << std::endl;
        wparams.audio_ctx = model_audio_ctx;
    }

    // Язык и потоки
    wparams.language  = params.language.empty() ? nullptr : params.language.c_str();
    wparams.n_threads = params.n_threads;

    // Выполнение транскрипции
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Ошибка: Не удалось выполнить транскрипцию аудио" << std::endl;
        // Расчёт времени выполнения даже при ошибке
        const auto t_end = std::chrono::high_resolution_clock::now();
        t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        return "";
    }

    // Если распознавание началось, значит юзер говорит — прерываем старый TTS
    g_is_interrupted.store(true);

    // Сбор результатов транскрипции
    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text != nullptr) {
            result += text;
        }

        // Расчёт вероятности для сегмента
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            // Защита от некорректных индексов токенов
            if (i >= 0 && i < n_segments && j >= 0 && j < n_tokens) {
                const auto token = whisper_full_get_token_data(ctx, i, j);
                prob += token.p;
                ++prob_n;
            }
        }
    }

    // Расчёт средней вероятности
    if (prob_n > 0) {
        prob /= static_cast<float>(prob_n);
    } else {
        prob = 0.0f;
        std::cerr << "Предупреждение: Нет токенов для вычисления вероятности" << std::endl;
    }

    // Замер времени выполнения
    const auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = t_end - t_start;

    // Защита от отрицательного времени (редкий случай проблем с системными часами)
    if (duration.count() < 0) {
        std::cerr << "Предупреждение: Обнаружено отрицательное время выполнения" << std::endl;
        t_ms = 0;
    } else {
        t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    return result;
}

// -----------------------------------------------------------------------------
// @brief Разбивает строку на слова
// @param txt Входной текст
// @return Вектор слов
// -----------------------------------------------------------------------------
static std::vector<std::string> get_words(const std::string& txt)
{
    std::vector<std::string> words;
    std::istringstream iss(txt);
    std::string word;

    while (iss >> word) {
        words.emplace_back(std::move(word));
    }

    return words;
}

// Функция для получения временной директории (с fallback)
std::string getTempDir() {
    // Пытаемся получить временную директорию через std::filesystem
    try {
        auto temp_path = std::filesystem::temp_directory_path();
        if (!temp_path.empty()) {
            return temp_path.string();
        }
    } catch (const std::exception &e) {
        std::cerr << "[getTempDir] filesystem exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[getTempDir] Unknown filesystem exception" << std::endl;
    }

#ifdef _WIN32
    // Fallback: WinAPI
    TCHAR path_buf[MAX_PATH] = {0};
    DWORD ret_val = GetTempPath(MAX_PATH, path_buf);

    if (ret_val == 0 || ret_val > MAX_PATH) {
        std::cerr << "[getTempDir] GetTempPath failed, error: " << GetLastError() << std::endl;
        return "";
    }

    // Проверяем, что буфер не пуст
    if (path_buf[0] == 0) {
        std::cerr << "[getTempDir] GetTempPath returned empty path" << std::endl;
        return "";
    }

    #if defined(UNICODE) || defined(_UNICODE)
        try {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            std::string result = converter.to_bytes(path_buf);
            // Удаляем завершающий слеш, если есть
            if (!result.empty() && (result.back() == '\\' || result.back() == '/')) {
                result.pop_back();
            }
            return result;
        } catch (const std::exception &e) {
            std::cerr << "[getTempDir] UTF-8 conversion failed: " << e.what() << std::endl;
            return "";
        } catch (...) {
            std::cerr << "[getTempDir] Unknown error during UTF-8 conversion" << std::endl;
            return "";
        }
    #else
        std::string result(path_buf);
        if (!result.empty() && (result.back() == '\\' || result.back() == '/')) {
            result.pop_back();
        }
        return result;
    #endif

#else
    // POSIX fallback
    const char* tmpdir = std::getenv("TMPDIR");
    if (tmpdir && tmpdir[0] != '\0') {
        return std::string(tmpdir);
    }
    return "/tmp";
#endif
}

// Записывает в файл значение 0 или 1, чтобы разрешить или запретить воспроизведение XTTS
// @path: ссылка на строку, куда будет записан полный путь к файлу (возвращается для отладки)
// @xtts_play_allowed: 0 — воспроизведение запрещено, 1 — разрешено
void allow_xtts_file(std::string& path, int xtts_play_allowed) {
    // Получаем путь к временной директории с использованием std::filesystem
    std::string temp_path = getTempDir();
    if (temp_path.empty()) {
        std::cerr << "ERROR: allow_xtts_file: Could not get temporary directory." << std::endl;
        return;
    }
    // Формируем путь к файлу корректно — через std::filesystem, чтобы избежать проблем со слешами
    // Это гарантирует работу как на Windows, так и на Linux/macOS
    #if __cplusplus >= 201703L
    std::filesystem::path p(temp_path);
    path = (p / "xtts_play_allowed.txt").string();
    #else
    // Резервный вариант для старых компиляторов
    if (!temp_path.empty() && temp_path.back() != '/' && temp_path.back() != '\\') {
        temp_path += '/';
    }
    path = temp_path + "xtts_play_allowed.txt";
    #endif

    const std::string fileName{path};
    // Открываем файл для чтения
    std::ifstream readStream(fileName);
    std::string singleLine;
    bool fileExists = readStream.is_open();

    if (!fileExists) {
        // Файл не существует — пробуем создать его
        std::ofstream writeStream(fileName);
        if (!writeStream.is_open()) {
            std::cerr << "ERROR: allow_xtts_file: Failed to create file: " << fileName << std::endl;
            return;
        }
        writeStream << xtts_play_allowed;
        writeStream.flush();
    } else {
        // Файл существует — читаем
        std::getline(readStream, singleLine);
        readStream.close();

        // Преобразуем строку в число
        int stored_value = 0;
        try {
            stored_value = std::stoi(singleLine);
        } catch (...) {
            stored_value = -1; // Некорректное значение — перезаписывать
        }

        // Если значение отличается — обновляем файл
        if (stored_value != xtts_play_allowed) {
            std::ofstream writeStream(fileName);
            if (!writeStream.is_open()) {
                std::cerr << "ERROR: allow_xtts_file: Failed to write to file: " << fileName << std::endl;
                return;
            }
            writeStream << xtts_play_allowed;
            writeStream.flush();
        }
    }
}

// Убирает пробельные символы из начала строки
inline void ltrim(std::string &s) {
    if (s.empty()) return;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
               && ch != '\f' && ch != '\v' && ch != 0xA0;
    }));
}

// Убирает пробельные символы из конца строки
inline void rtrim(std::string &s) {
    if (s.empty()) return;
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
               && ch != '\f' && ch != '\v' && ch != 0xA0;
    }).base(), s.end());
}

// Убирает пробельные символы с обеих сторон строки
inline void trim(std::string &s) {
    if (s.empty()) return;
    rtrim(s);
    ltrim(s);
}

// Проверяет, является ли символ пунктуационным знаком
bool IsPunctuationMark(char c) {
    switch (static_cast<unsigned char>(c)) {
        case ',':
            [[fallthrough]];
        case '.':
            [[fallthrough]];
        case '?':
            return true;
		case '!':
            return true;
        default:
            return false;
    }
}

// Убирает все пунктуационные знаки из строки
std::string StripPunctuationMarks(const std::string& text) {
    std::string cleanText;
    for (const auto& c : text) {
        if (!IsPunctuationMark(c)) {
           cleanText += c;
        }
    }
    return cleanText;
}

// Переводит все символы строки в нижний регистр
std::string LowerCase(const std::string& text) {
    std::string lowerCasedText;
    for (const auto& c : text) {
        lowerCasedText += std::tolower(c, std::locale());  // с учётом локали
    }
    return lowerCasedText;
}

// get part of the string that is after the @command
std::string ParseCommandAndGetKeyword(std::string textHeardTrimmed, const std::string& command="google") {
     textHeardTrimmed = StripPunctuationMarks(textHeardTrimmed);
    // Создаем копию входной строки для дальнейшей обработки
    std::string sanitizedInput = textHeardTrimmed;
    // Переменные для поиска позиции команды и флага начала команды
    std::size_t pos = 0;
    bool startsWithPrefix = false;
    // Множество слов и фраз, которые нужно удалить из входной строки для очистки команды
    static const std::unordered_set<std::string> please_needles = {
        "can you hear me", "Can you hear me", "Are you here", "are you here",
        "Do you hear me", "do you hear me", "Пожалуйста", "пожалуйста",
        "Позови", "позови", "ты тут", "Ты тут", "ты здесь", "Ты здесь",
        "ты меня слышишь", "Ты меня слышишь", "ты слышишь меня", "Ты слышишь меня",
        "Hey", "hey", "please", "Please", "can you", "Can you", "let's", "Let's",
        "What do you think", "Что ты думаешь", "что ты думаешь",
        "Что ты об этом думаешь", "что ты об этом думаешь"
    };

    // Переменная для хранения результата
	std::string result_param = "";

    // Удаляем ненужные слова из входной строки
    for (const auto& prefix : please_needles) {
		sanitizedInput = ::replace(sanitizedInput, prefix, "");
	}

    // Удаляем лишние пробелы в начале и конце строки
	trim(sanitizedInput);

// безопасный поиск позиции аргумента команды
    // Если команда - "google", ищем соответствующие префиксы
    if (command == "google") {
        static const std::unordered_set<std::string> prefixNeedles = {
           "Погугли", "погугли", "гугли", "гугл", "угли", "углe", "По гугле", "По угли"
        };

        // Ищем начальные команды в строке — без выхода за границы
        for (const auto& prefix : prefixNeedles) {
            if (sanitizedInput.size() >= prefix.size() &&
                sanitizedInput.compare(0, prefix.length(), prefix) == 0) {
                // установим базовую позицию сразу за префиксом
                size_t base = prefix.length();
                // продвигаемся через любые пробелы или двоеточие, чтобы найти начало ключевого слова
                // Защита от выхода за границы: проверяем base перед доступом к символу
                while (base < sanitizedInput.size()) {
                    unsigned char ch = static_cast<unsigned char>(sanitizedInput[base]);
                    if (std::isspace(ch) || ch == ':') {
                        ++base;
                    } else {
                        break;
                    }
                }
                pos = base;
                startsWithPrefix = true;
                break;
            }
        }
    }

    // Если команда не начинается с префикса — ищем расположение самого ключевого слова команды
    if (!startsWithPrefix) {
        size_t found = sanitizedInput.find(command);
        if (found != std::string::npos) {
            size_t base = found + command.size();
            // пропускаем разделители
            while (base < sanitizedInput.size() && (std::isspace((unsigned char)sanitizedInput[base]) || sanitizedInput[base] == ':' ))
                ++base;
            pos = base;
        } else {
            // резервный поиск с учётом написания с большой буквы (Call)
            size_t foundCall = sanitizedInput.find("Call");
            if (foundCall != std::string::npos) {
                size_t base = foundCall + 4;
                while (base < sanitizedInput.size() && (std::isspace((unsigned char)sanitizedInput[base]) || sanitizedInput[base] == ':' ))
                    ++base;
                pos = base;
            } else {
                pos = 0; // команда не найдена — вернём базовую 0 (означает "всё после начала")
            }
        }
    }


// Если команда - "call"
if (command == "call")
{
    // НАЧАЛО: Универсальная нормализация имён
    trim(sanitizedInput);
    // Специфичные замены UTF-8 с полной безопасностью
    if (sanitizedInput.size() >= 2) {
        bool utf8_rule_applied = false;
        const size_t len = sanitizedInput.size();

        // Функция для безопасной проверки и замены
        auto safeReplace = [&](size_t pos, const std::string& from, const std::string& to) -> bool {
            if (pos + from.length() <= len) {
                if (sanitizedInput.compare(pos, from.length(), from) == 0) {
                    sanitizedInput.replace(pos, from.length(), to);
                    return true;
                }
            }
            return false;
        };

        // Васю -> Вася
        utf8_rule_applied = safeReplace(len - 2, "\xD1\x83", "\xD0\xB0") || utf8_rule_applied;

        // Петю -> Петя
        utf8_rule_applied = safeReplace(len - 2, "\xD1\x8E", "\xD0\x8F") || utf8_rule_applied;

        if (utf8_rule_applied) {
            trim(sanitizedInput);
        }
    }

    // Общие замены через regex
    if (sanitizedInput.size() >= 2) {
        static const std::regex re_male_genitive_ogo_ego(R"((.+)([оe]го)$)", std::regex_constants::icase); // Ивана́ его -> Иван
        static const std::regex re_male_u(R"((.+)у$)", std::regex_constants::icase);        // Ивану -> Иван
        static const std::regex re_male_a(R"((.+)а$)", std::regex_constants::icase);        // Ивана -> Иван
        static const std::regex re_male_om(R"((.+)ом$)", std::regex_constants::icase);      // Иваном -> Иван
        static const std::regex re_male_em(R"((.+)ем$)", std::regex_constants::icase);      // Андреем -> Андрей
        static const std::regex re_male_yu(R"((.+)ю$)", std::regex_constants::icase);       // Сергею -> Сергей
        static const std::regex re_male_yem(R"((.+)еем$)", std::regex_constants::icase);    // Дмитрием -> Дмитрий

        static const std::regex re_female_e(R"((.+)е$)", std::regex_constants::icase);      // Маше -> Маша
        static const std::regex re_female_oj(R"((.+)ой$)", std::regex_constants::icase);    // Ольгой -> Ольга
        static const std::regex re_female_y(R"((.+)ы$)", std::regex_constants::icase);      // Эммы -> Эмма
        static const std::regex re_female_i(R"((.+)и$)", std::regex_constants::icase);      // Маши -> Маша
        static const std::regex re_female_ej(R"((.+)ей$)", std::regex_constants::icase);    // Наташей -> Наташа
        static const std::regex re_female_yu(R"((.+)ю$)", std::regex_constants::icase);     // Алёну -> Алёна

        // Новое: имена на -ь (Любовь)
        static const std::regex re_female_instr_lyubov(R"((.+)ью$)", std::regex_constants::icase);  // Любовью -> Любовь
        static const std::regex re_female_dat_lyubov(R"((.+)и$)", std::regex_constants::icase);     // Любови -> Любовь

        sanitizedInput = std::regex_replace(sanitizedInput, re_male_genitive_ogo_ego, "$1");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_om, "$1");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_em, "$1й");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_yem, "$1й");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_yu, "$1й");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_u, "$1");
        sanitizedInput = std::regex_replace(sanitizedInput, re_male_a, "$1");

        sanitizedInput = std::regex_replace(sanitizedInput, re_female_oj, "$1а");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_e, "$1а");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_y, "$1а");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_i, "$1а");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_ej, "$1а");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_yu, "$1а");

        sanitizedInput = std::regex_replace(sanitizedInput, re_female_instr_lyubov, "$1ь");
        sanitizedInput = std::regex_replace(sanitizedInput, re_female_dat_lyubov, "$1ь");
    }

    trim(sanitizedInput);
    textHeardTrimmed = sanitizedInput;
}

result_param = textHeardTrimmed.substr(pos);
return result_param;
}

// Callback-функция для записи данных, полученных через CURL, в строку / Используется, например, для сохранения ответа от HTTP-запроса
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    // Проверка атомарного флага прерывания
    if (g_is_interrupted.load()) {
        // Возвращаем число байт, отличное от реального (size*nmemb),
        // это штатный способ сказать cURL: "Остановись с ошибкой WRITE_ERROR"
        return 0;
    }
    size_t realsize = size * nmemb;
    ((std::string*)userp)->append((char*)contents, realsize);
    return realsize;
}

// Удаляет все ведущие справа символы, совпадающие с targetCharacter
std::string RemoveTrailingCharacters(const std::string &inputString, const char targetCharacter) {
    // Ищем первую позицию с конца, где символ не равен targetCharacter
    auto lastNonTargetPosition = std::find_if(inputString.rbegin(), inputString.rend(), [targetCharacter](auto ch) {
        return ch != targetCharacter;
    });
    // Возвращаем строку до найденной позиции
    return std::string(inputString.begin(), lastNonTargetPosition.base());
}

// Удаляет ведущие справа символы (только ASCII), совпадающие с любым из targetCharacters
// ВНИМАНИЕ: Эта функция УПРОЩЕНА и работает ТОЛЬКО с ASCII-символами (запятая, точка, скобки и т.д.)
//           Она НЕ обрезает русские буквы и корректно обрабатывает UTF-8 многобайтовые символы.
std::string RemoveTrailingCharactersUtf8(const std::string& inputString, const std::string& targetCharacters) {
    // Проверка на пустую строку
    if (inputString.empty()) {
        return inputString;
    }

    // Начинаем с конца строки
    size_t pos = inputString.length();

    // Идём с конца, пропуская целевые символы
    while (pos > 0) {
        // Определяем начало последнего UTF-8 символа
        // В UTF-8: байты 0x80-0xBF являются продолжением символа
        size_t char_start = pos - 1;
        while (char_start > 0 && (static_cast<unsigned char>(inputString[char_start]) & 0xC0) == 0x80) {
            char_start--;
        }

        // Извлекаем последний символ (может быть 1-4 байта)
        std::string last_char = inputString.substr(char_start, pos - char_start);

        // Проверяем, нужно ли удалить этот символ
        // Удаляем ТОЛЬКО если это ASCII-символ (1 байт) из targetCharacters
        bool should_remove = false;
        if (last_char.size() == 1) {  // Только ASCII-символы (1 байт)
            char c = last_char[0];
            for (char target : targetCharacters) {
                if (c == target) {
                    should_remove = true;
                    break;
                }
            }
        }

        // Если символ не подлежит удалению — выходим из цикла
        if (!should_remove) {
            break;
        }

        // Удаляем символ (сдвигаем позицию)
        pos = char_start;
    }

    // Возвращаем обрезанную строку
    return inputString.substr(0, pos);
}

// Кодирует строку в формат URL-кодирования (например, пробелы становятся %20)
std::string UrlEncode(const std::string& str) {
    // Инициализируем CURL для кодирования
    CURL* curl = curl_easy_init();
    if (curl) {
        // Кодируем строку
        char* encodedUrl = curl_easy_escape(curl, str.c_str(), static_cast<int>(str.length()));
        std::string escapedUrl;
        if (encodedUrl) {
            escapedUrl.assign(encodedUrl);
            curl_free(encodedUrl);
        }
        curl_easy_cleanup(curl);
        return escapedUrl;
        }
    // Если CURL не инициализировался — возвращаем пустую строку
    return {};
}

//  Отправляет JSON-данные на сервер по указанному URL.
std::string send_curl_json(const std::string &url, const std::map<std::string, std::string>& params) {
    CURL *curl = curl_easy_init();
    std::string readBuffer;
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

    // RAII-обёртка для автоматического освобождения curl
    struct CurlDeleter {
        void operator()(CURL* c) const { if (c) curl_easy_cleanup(c); }
    };
    std::unique_ptr<CURL, CurlDeleter> curl_guard(curl);

    // RAII-обёртка для автоматического освобождения заголовков
    struct SlistDeleter {
        void operator()(curl_slist* s) const { if (s) curl_slist_free_all(s); }
    };
    std::unique_ptr<curl_slist, SlistDeleter> headers_guard(nullptr);

// Локальная лямбда для экранирования специальных символов в JSON
auto escape_json = [](const std::string& s) -> std::string {
    std::string result;
    result.reserve(s.size() * 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                result += static_cast<char>(c);
        }
    }
    return result;
};
	// Настройка запроса
    try {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

        std::ostringstream oss;
        bool firstParam = true;
        oss << "{";
        for (const auto& param : params) {
            if (!firstParam) oss << ",";

            // Экранируем ключ и значение, проверяем на пустоту
            std::string escaped_key = escape_json(param.first);
            std::string escaped_value = escape_json(param.second);

            // Проверяем, что экранирование прошло успешно (не пустое)
            if (!escaped_key.empty()) {
                oss << "\"" << escaped_key << "\":\"" << escaped_value << "\"";
            } else {
                // Если ключ пустой после экранирования, пропускаем параметр
                fprintf(stderr, "Warning: skipping empty JSON key\n");
                continue;
            }

            firstParam = false;
        }
        oss << "}";
        std::string jsonData = oss.str();

        // Проверка, что JSON не пустой
        if (jsonData.size() <= 2) {
            fprintf(stderr, "Warning: generated empty JSON, using fallback\n");
            jsonData = "{}";
        }

        fprintf(stdout, "send_curl_json: %s\n", jsonData.c_str());

		// Устанавливаем заголовки
        curl_slist *headers = curl_slist_append(nullptr, "Content-Type: application/json");
        headers_guard.reset(headers);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		// Устанавливаем тело запроса
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

		// Устанавливаем callback для получения данных
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Выполняем запрос без лишнего шума в консоли
        CURLcode res = curl_easy_perform(curl);

        // Выводим только реальные ошибки, игнорируем прерывание пользователем
        /*
        if (res != CURLE_OK && !(res == CURLE_WRITE_ERROR && g_is_interrupted.load())) {
            extern whisper_params params;
            if (params.verbose) {
                fprintf(stderr, " [TTS Error: %s]", curl_easy_strerror(res));
            }
        }
        */

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return "";
    }

    // RAII автоматически освободит curl и headers при выходе из функции

    return readBuffer;
}


//Выполняет GET-запрос по указанному URL.
std::string send_curl(std::string url)
{
  CURL *curl;
  CURLcode res;
  std::string readBuffer;

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
  }

  return readBuffer;
}

// Возвращает количество UTF-8 символов в строке.
int utf8_length(const std::string& str)
{
    if (str.empty()) return 0;

    size_t i = 0;
    int chars = 0;
    const size_t ix = str.size();

    while (i < ix) {
        unsigned char c = static_cast<unsigned char>(str[i]);

        if (c <= 0x7F) {
            // ASCII (1 байт)
            ++i;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-байтовая последовательность
            if (i + 1 >= ix) return chars; // Обрезанный UTF-8, возвращаем подсчитанное
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-байтовая последовательность
            if (i + 2 >= ix) return chars; // Обрезанный UTF-8, возвращаем подсчитанное
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-байтовая последовательность
            if (i + 3 >= ix) return chars; // Обрезанный UTF-8, возвращаем подсчитанное
            i += 4;
        } else {
            // Некорректный байт, пропускаем его
            ++i;
        }

        ++chars;
    }

    return chars;
}


/**
 * Возвращает подстроку по индексам UTF-8 символов.
 */
std::string utf8_substr(const std::string& str, unsigned int start, unsigned int leng)
{
    // Пустая подстрока
    if (leng == 0) return "";

    const size_t ix = str.size();
    size_t i = 0;               // индекс в байтах
    unsigned int chars = 0;     // индекс в символах
    size_t min_byte_index = std::string::npos;
    size_t max_byte_index = std::string::npos;

    // Проходим по всем символам UTF-8
    while (i < ix) {
        // Запоминаем позицию начала нужного символа
        if (chars == start) min_byte_index = i;

        // Если дошли до конца запрошенной длины, запоминаем позицию и выходим
        if (chars == start + leng) {
            max_byte_index = i;
            break;
        }

        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t step = 1;

        // Определяем длину UTF-8 символа по первому байту
        if (c <= 0x7F) {
            step = 1;                                   // ASCII
        } else if ((c & 0xE0) == 0xC0) {
            step = 2;                                   // 2-байтовый символ
            if (i + 1 >= ix) return "";
        } else if ((c & 0xF0) == 0xE0) {
            step = 3;                                   // 3-байтовый символ
            if (i + 2 >= ix) return "";
        } else if ((c & 0xF8) == 0xF0) {
            step = 4;                                   // 4-байтовый символ
            if (i + 3 >= ix) return "";
        } else {
            return "";                                  // Некорректный UTF-8
        }

        i += step;
        ++chars;
    }

    // Если не нашли конец подстроки, берём до конца строки
    if (max_byte_index == std::string::npos) max_byte_index = ix;

    // Если не нашли начало или границы некорректны, возвращаем пустую строку
    if (min_byte_index == std::string::npos || max_byte_index > ix) return "";

    return str.substr(min_byte_index, max_byte_index - min_byte_index);
}

/**
 * Простейшая транслитерация английских букв в русские (en -> ru).
 * Правила:
 *  - Для однобайтовых ASCII символов (A-Z, a-z) выполняется замена по таблице.
 *  - Многобайтовые UTF-8 символы (русские, эмодзи и т.п.) копируются как есть (чтобы не порвать кодировку).
 *  - Если букве соответствует последовательность (например 'x' -> "кс"), возвращается несколько UTF-8 символов.
 */
std::string translit_en_ru(IN const std::string &str) {
    // Таблица соответствий ASCII -> UTF-8 (кириллица).
    static const std::unordered_map<char, std::string> tbl = {
        // нижний регистр
        {'a', u8"а"}, {'b', u8"б"}, {'c', u8"ц"}, {'d', u8"д"}, {'e', u8"е"},
        {'f', u8"ф"}, {'g', u8"г"}, {'h', u8"х"}, {'i', u8"и"}, {'j', u8"й"},
        {'k', u8"к"}, {'l', u8"л"}, {'m', u8"м"}, {'n', u8"н"}, {'o', u8"о"},
        {'p', u8"п"}, {'q', u8"к"}, {'r', u8"р"}, {'s', u8"с"}, {'t', u8"т"},
        {'u', u8"у"}, {'v', u8"в"}, {'w', u8"в"}, {'x', u8"кс"}, {'y', u8"й"},
        {'z', u8"з"},
        // верхний регистр
        {'A', u8"А"}, {'B', u8"Б"}, {'C', u8"Ц"}, {'D', u8"Д"}, {'E', u8"Е"},
        {'F', u8"Ф"}, {'G', u8"Г"}, {'H', u8"Х"}, {'I', u8"И"}, {'J', u8"Й"},
        {'K', u8"К"}, {'L', u8"Л"}, {'M', u8"М"}, {'N', u8"Н"}, {'O', u8"О"},
        {'P', u8"П"}, {'Q', u8"К"}, {'R', u8"Р"}, {'S', u8"С"}, {'T', u8"Т"},
        {'U', u8"У"}, {'V', u8"В"}, {'W', u8"В"}, {'X', u8"Кс"}, {'Y', u8"Й"},
        {'Z', u8"З"}
    };

    // Результат — резервируем место для эффективности (примерно в 2 раза больше байт, т.к. замены могут быть многобайтовыми).
    std::string out;
    out.reserve(str.size() * 2);

    // Проходим по входной строке байт за байтом.
    for (size_t i = 0; i < str.size();) {
        unsigned char c = static_cast<unsigned char>(str[i]);

        if (c < 0x80) {
            // ASCII: пытаемся сопоставить букву
            auto it = tbl.find(static_cast<char>(c));
            if (it != tbl.end()) {
                // Нашли замену (UTF-8 строка, может быть 1 или несколько символов)
                out += it->second;
            } else {
                // Не буква (цифра, пунктуация и т.д.) — копируем как есть
                out.push_back(static_cast<char>(c));
            }
            ++i; // продвигаемся на 1 байт
        } else {
            // Определяем длину по первому байту.
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;        // 110xxxxx
            else if ((c & 0xF0) == 0xE0) len = 3;   // 1110xxxx
            else if ((c & 0xF8) == 0xF0) len = 4;   // 11110xxx
            else {
                // Неверный стартовый байт UTF-8 — чтобы не зациклиться, копируем 1 байт и идём дальше.
                out.push_back(static_cast<char>(c));
                ++i;
                continue;
            }

            // Если последовательность обрезана (т.е. строка закончилась раньше), копируем оставшиеся байты и выходим.
            if (i + len <= str.size()) {
                out.append(str.data() + i, len);
                i += len;
            } else {
                // Кусок в конце — просто копируем остаток
                out.append(str.data() + i, str.size() - i);
                break;
            }
        }
    }
    return out;
}


/**
 * Находит имя в строке, которое следует после '\n' и перед ": "
 */
std::string find_name(const std::string& str) {
    if (str.size() < 4) return ""; // Минимальная длина для проверки

    // Ищем символ '\n'
    size_t pos = str.find('\n');
    if (pos == std::string::npos || pos + 1 >= str.size()) return "";

    // Ищем ": " после найденного '\n'
    size_t endPos = str.find(": ", pos + 1);
    if (endPos == std::string::npos || endPos <= pos + 1) return "";

    // Извлекаем подстроку между '\n' и ": "
    std::string substr = str.substr(pos + 1, endPos - (pos + 1));

    // Удаляем пробелы в начале и конце
    while (!substr.empty() && substr.front() == ' ') substr.erase(substr.begin());
    while (!substr.empty() && substr.back() == ' ') substr.pop_back();

    // Проверяем длину имени (2-70 символов)
    if (substr.length() < 2 || substr.length() > 70) return "";

    return substr;
}

/**
 * Преобразует вектор токенов LLaMA в строку.
 */
std::string emb_to_str(llama_context* ctx_llama, const std::vector<llama_token>& embd) {
    std::string ss;
    // Блокируем один раз на всю функцию, а не на каждый токен
    std::lock_guard<std::mutex> lock(g_llama_mutex);
    for (const auto& token : embd) {
        ss += llama_token_to_piece(ctx_llama, token);
    }
    return ss;
}

// ============================================================
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: преобразование URL в читаемый текст
// ============================================================
std::string url_to_speech(const std::string& url) {
    std::string result;

    // Удаляем протокол
    std::string clean_url = url;
    clean_url = replace(clean_url, "https://", "");
    clean_url = replace(clean_url, "http://", "");
    clean_url = replace(clean_url, "www.", "");

    // Удаляем параметры
    size_t qpos = clean_url.find('?');
    if (qpos != std::string::npos) clean_url = clean_url.substr(0, qpos);
    size_t hpos = clean_url.find('#');
    if (hpos != std::string::npos) clean_url = clean_url.substr(0, hpos);

    // Удаляем trailing slash
    if (!clean_url.empty() && clean_url.back() == '/') {
        clean_url.pop_back();
    }

    // Разбиваем на части
    std::vector<std::string> parts;
    std::string current;

    for (char c : clean_url) {
        if (c == '.' || c == '/' || c == '-' || c == '_') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
            if (c == '.') parts.push_back("dot");
            else if (c == '/') parts.push_back("slash");
            else if (c == '-') parts.push_back("dash");
            else if (c == '_') parts.push_back("underscore");
        } else {
            current += c;
        }
    }
    if (!current.empty()) parts.push_back(current);

    // Собираем результат
    for (size_t i = 0; i < parts.size(); ++i) {
        if (!result.empty()) result += " ";
        result += parts[i];
    }

    return result;
}

// Асинхронная функция для отправки текста в TTS (Text-to-Speech) сервис
// Все параметры передаются по значению для безопасности в многопоточном окружении.
// ВСЕ регулярные выражения компилируются ОДИН РАЗ при первом вызове функции.
// Использует оптимизированные regex и безопасную обработку UTF-8.
void send_tts_async(std::string text,
                    std::string speaker_wav = "Emma",
                    std::string language = "ru",
                    std::string tts_url = "http://localhost:8020/") {

// Быстрая защита: если пусто — сразу выходим
if (text.empty()) {
    return;
}

// ============================================================
// ЭТАП 0: ЗАЩИТА ЧИСЛОВЫХ ПАТТЕРНОВ (от последующих замен)
// ============================================================
// Сохраняем для num2words: время, даты, дроби, телефоны, проценты, валюты
std::vector<std::pair<std::string, std::string>> protected_patterns;

// Защита времени: 15:30, 15:30:45
try {
    static const std::regex re_time(R"(\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?\b)",
                                    std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_time);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "⏰TIME" + std::to_string(protected_patterns.size()) + "⏰";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (time protection): %s\n", e.what());
}

// Защита дат: 31.12.2025
try {
    static const std::regex re_date_dots(R"(\b(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\.(\d{4})\b)",
                                         std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_dots);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📅DATE" + std::to_string(protected_patterns.size()) + "📅";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (date dots protection): %s\n", e.what());
}

// Защита дат: 2025-12-31 (ISO формат)
try {
    static const std::regex re_date_iso(R"(\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b)",
                                        std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_iso);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📅DATE" + std::to_string(protected_patterns.size()) + "📅";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (date iso protection): %s\n", e.what());
}

// Защита дат: 12/31/2025 (американский формат)
try {
    static const std::regex re_date_slash(R"(\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b)",
                                          std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_slash);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📅DATE" + std::to_string(protected_patterns.size()) + "📅";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (date slash protection): %s\n", e.what());
}

// Защита десятичных дробей: 3.14, 0.5, 0,5 (русский формат)
try {
    static const std::regex re_decimal(R"(\b\d+[.,]\d+\b(?![\w-]))", std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_decimal);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "🔢DEC" + std::to_string(protected_patterns.size()) + "🔢";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (decimal protection): %s\n", e.what());
}

// Защита процентов: 50%, 12.5%, 12,5%
try {
    static const std::regex re_percent(R"(\b\d+(?:[.,]\d+)?\s*%)", std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_percent);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📊PCT" + std::to_string(protected_patterns.size()) + "📊";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (percent protection): %s\n", e.what());
}

// ============================================================
// ЗАЩИТА ВАЛЮТ (УЛУЧШЕННАЯ)
// ============================================================
try {
    // Улучшенный паттерн для валют:
    // 1. Сумма + символ: 100$, 50€, 1000₽, 99.99$
    // 2. Символ + сумма: $100, €50, £100, $99.99
    // 3. Сумма с разделителями тысяч: 1,000$, 1.000€
    // 4. Сумма с десятичной частью и разделителями: 1,000.50$
    static const std::regex re_currency(
        R"(\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*[$€£¥₽]|\b[$€£¥₽]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)",
        std::regex::ECMAScript);

    std::string processed;
    auto it_currency = std::sregex_iterator(text.begin(), text.end(), re_currency);
    auto end_currency = std::sregex_iterator();
    size_t last_pos = 0;

    for (auto i = it_currency; i != end_currency; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "💰CUR" + std::to_string(protected_patterns.size()) + "💰";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (currency protection): %s\n", e.what());
}

// ============================================================
// ЗАЩИТА ДРОБЕЙ (УЛУЧШЕННАЯ)
// ============================================================
try {
    // Улучшенный паттерн для дробей:
    // 1. Простые дроби: 1/2, 3/4, 5/8
    // 2. Смешанные дроби: 1 1/2, 2 3/4
    // 3. Дроби с пробелами: 1 / 2, 3 / 4
    static const std::regex re_fraction(
        R"(\b\d+\s*/\s*\d+\b|\b\d+\s+\d+\s*/\s*\d+\b)",
        std::regex::ECMAScript);

    std::string processed;
    auto it_fraction = std::sregex_iterator(text.begin(), text.end(), re_fraction);
    auto end_fraction = std::sregex_iterator();
    size_t last_pos = 0;

    for (auto i = it_fraction; i != end_fraction; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "🔢FRAC" + std::to_string(protected_patterns.size()) + "🔢";

        // Нормализация дроби: удаляем лишние пробелы
        std::string frac_value = match.str();
        frac_value = std::regex_replace(frac_value, std::regex(R"(\s+)"), " ");
        frac_value = std::regex_replace(frac_value, std::regex(R"(\s*/\s*)"), "/");

        protected_patterns.emplace_back(marker, frac_value);
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (fraction protection): %s\n", e.what());
}

// Защита телефонов: +7 (123) 456-78-90
try {
    static const std::regex re_phone(R"(\+?[\d\s\-\(\)]{7,})", std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_phone);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📞PHONE" + std::to_string(protected_patterns.size()) + "📞";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (phone protection): %s\n", e.what());
}

// Защита URL
try {
    static const std::regex re_url(R"(https?://[^\s]+)", std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_url);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "🌐URL" + std::to_string(protected_patterns.size()) + "🌐";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (URL protection): %s\n", e.what());
}

// Защита email
try {
    static const std::regex re_email(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                                     std::regex::ECMAScript);
    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_email);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "📧EMAIL" + std::to_string(protected_patterns.size()) + "📧";
        protected_patterns.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (email protection): %s\n", e.what());
}

// ============================================================
// ЭТАП 0.5: ЗАЩИТА ТОЧЕК, КОТОРЫЕ НЕ ЯВЛЯЮТСЯ КОНЦОМ ПРЕДЛОЖЕНИЯ
// ============================================================
std::vector<std::pair<std::string, std::string>> protected_dots;

try {
    // Сначала защищаем IP-адреса (если вдруг не попали в ЭТАП 0)
    static const std::regex re_ip(R"(\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b)",
                                   std::regex::ECMAScript);

    std::string processed;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_ip);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        processed += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "🔒IP" + std::to_string(protected_dots.size()) + "🔒";
        protected_dots.emplace_back(marker, match.str());
        processed += marker;
        last_pos = match.position() + match.length();
    }
    processed += text.substr(last_pos);
    text = processed;

    // Теперь алгоритмическая защита остальных точек
    struct DotInfo {
        size_t pos;
        std::string before;  // до 20 символов до точки
        std::string after;   // до 10 символов после точки
    };

    std::vector<DotInfo> dots;

    // Собираем все точки с контекстом
    for (size_t i = 0; i < text.length(); ++i) {
        if (text[i] == '.') {
            DotInfo info;
            info.pos = i;

            // До 20 символов ДО точки
            size_t start = (i > 20) ? i - 20 : 0;
            info.before = text.substr(start, i - start);

            // До 10 символов ПОСЛЕ точки
            info.after = text.substr(i + 1, std::min((size_t)10, text.length() - i - 1));

            dots.push_back(info);
        }
    }

    // Строим новую строку с защищёнными точками
    std::string new_text;
    last_pos = 0;

    for (const auto& dot : dots) {
        bool protect = false;

        // ========== КРИТЕРИИ ЗАЩИТЫ ТОЧКИ ==========

        // КРИТЕРИЙ 1: ЧАСТЬ ЧИСЛА (дата, версия, IP)
        if (!dot.before.empty() && !dot.after.empty()) {
            char prev_char = dot.before.back();
            char next_char = dot.after[0];

            if (isdigit(prev_char) && isdigit(next_char)) {
                protect = true;
            }
        }

        // КРИТЕРИЙ 2: ЧАСТЬ АББРЕВИАТУРЫ (т.д., т.е., P.S.)
        if (!protect && !dot.before.empty() && !dot.after.empty()) {
            char prev_char = dot.before.back();

            if (isalpha(prev_char)) {
                // Проверяем следующий непробельный символ
                size_t after_pos = 0;
                while (after_pos < dot.after.length() && isspace(dot.after[after_pos])) {
                    after_pos++;
                }

                if (after_pos < dot.after.length() && isalpha(dot.after[after_pos])) {
                    protect = true;
                }
            }
        }

        // КРИТЕРИЙ 3: ИНИЦИАЛЫ (А.С. Пушкин)
        if (!protect && !dot.before.empty()) {
            // Ищем вторую точку рядом
            size_t next_dot = text.find('.', dot.pos + 1);
            if (next_dot != std::string::npos && next_dot - dot.pos <= 4) {
                bool valid = true;
                for (size_t j = dot.pos + 1; j < next_dot; ++j) {
                    if (!isalpha(text[j]) && !isspace(text[j])) {
                        valid = false;
                        break;
                    }
                }
                if (valid) protect = true;
            }
        }

        // КРИТЕРИЙ 4: ПОСЛЕ ТОЧКИ НЕТ ЗАГЛАВНОЙ БУКВЫ (не конец предложения)
        if (!protect && !dot.after.empty()) {
            // Пропускаем пробелы и закрывающие скобки/кавычки
            size_t after_pos = 0;
            while (after_pos < dot.after.length() &&
                   (isspace(dot.after[after_pos]) ||
                    dot.after[after_pos] == ')' ||
                    dot.after[after_pos] == ']' ||
                    dot.after[after_pos] == '}' ||
                    dot.after[after_pos] == '"' ||
                    dot.after[after_pos] == '\'')) {
                after_pos++;
            }

            if (after_pos < dot.after.length()) {
                char next_char = dot.after[after_pos];
                // Если после точки строчная буква, цифра или знак препинания — это не конец
                if (islower(next_char) || isdigit(next_char) ||
                    next_char == '.' || next_char == ',' ||
                    next_char == ';' || next_char == ':' ||
                    next_char == '?' || next_char == '!') {
                    protect = true;
                }
            }
        }

        // КРИТЕРИЙ 5: ПОСЛЕ ТОЧКИ СРАЗУ ЗАПЯТАЯ, ТОЧКА С ЗАПЯТОЙ ИЛИ ДВОЕТОЧИЕ
        if (!protect && !dot.after.empty()) {
            char next_char = dot.after[0];
            if (next_char == ',' || next_char == ';' || next_char == ':' ||
                next_char == ')' || next_char == ']' || next_char == '}') {
                protect = true;
            }
        }

        // Добавляем текст до точки
        new_text += text.substr(last_pos, dot.pos - last_pos);

        if (protect) {
            // Защищаем точку маркером
            std::string marker = "🔵DOT" + std::to_string(protected_dots.size()) + "🔵";
            protected_dots.emplace_back(marker, ".");
            new_text += marker;
        } else {
            // Оставляем точку как есть (конец предложения)
            new_text += ".";
        }

        last_pos = dot.pos + 1;
    }

    // Добавляем остаток текста
    new_text += text.substr(last_pos);
    text = new_text;

} catch (const std::exception& e) {
    fprintf(stderr, "Error in dot protection: %s\n", e.what());
}

// ============================================================
// ЭТАП 1: БАЗОВАЯ ОЧИСТКА (убираем только явный мусор)
// ============================================================

// Унификация переводов строки - просто заменяем на пробел
try {
    static const std::regex re_newline(R"(\r\n|\r|\n)", std::regex::ECMAScript);
    text = std::regex_replace(text, re_newline, " ");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (newline): %s\n", e.what());
    text = replace(text, "\r\n", " ");
    text = replace(text, "\r", " ");
    text = replace(text, "\n", " ");
}
trim(text);
if (text.empty()) return;

// Удаление HTML-тегов (питону они не нужны)
try {
    static const std::regex re_html_tag(R"(<[^>]*>)", std::regex::ECMAScript);
    text = std::regex_replace(text, re_html_tag, " ");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (HTML): %s\n", e.what());
    text = replace(text, "<", " ");
    text = replace(text, ">", " ");
}

// =================================================================
// ДЕКОДИРОВАНИЕ HTML-СУЩНОСТЕЙ
// =================================================================
// Заменяем HTML-сущности на соответствующие символы
// Все замены безопасны: даже если подстрока не найдена, replace вернёт исходную строку
// =================================================================
text = replace(text, "&nbsp;", " ");
text = replace(text, "&amp;", "&");
text = replace(text, "&lt;", "<");
text = replace(text, "&gt;", ">");
text = replace(text, "&quot;", "\"");
text = replace(text, "&#39;", "'");
text = replace(text, "&apos;", "'");
text = replace(text, "&#34;", "\"");
text = replace(text, "&rsquo;", "'");
text = replace(text, "&lsquo;", "'");
text = replace(text, "&rdquo;", "\"");
text = replace(text, "&ldquo;", "\"");
text = replace(text, "&mdash;", "-");
text = replace(text, "&ndash;", "-");
text = replace(text, "&hellip;", "...");

// Защита от бесконечных замен (например, если результат замены снова содержит сущность)
// Ограничиваем количество итераций для безопасного декодирования
int max_iterations = 10;
for (int iter = 0; iter < max_iterations; iter++) {
    std::string prev_text = text;

    // Повторяем замены для вложенных сущностей (редкий случай)
    text = replace(text, "&amp;lt;", "<");
    text = replace(text, "&amp;gt;", ">");
    text = replace(text, "&amp;quot;", "\"");
    text = replace(text, "&amp;amp;", "&");

    // Если после итерации текст не изменился, выходим
    if (text == prev_text) break;
}

trim(text);
if (text.empty()) return;

// Обработка "умных" кавычек и тире в UTF-8 (приводим к ASCII)
text = replace(text, "\xE2\x80\x9C", "\"");
text = replace(text, "\xE2\x80\x9D", "\"");
text = replace(text, "\xE2\x80\x98", "'");
text = replace(text, "\xE2\x80\x99", "'");
text = replace(text, "\xE2\x80\x93", "-");
text = replace(text, "\xE2\x80\x94", "-");
text = replace(text, "\xC2\xA0", " ");
text = replace(text, "\xE2\x80\xA6", "...");

trim(text);
if (text.empty()) return;

// 4.1 УНИВЕРСАЛЬНАЯ НОРМАЛИЗАЦИЯ ЭМОЦИЙ И ВЫДЕЛЕНИЙ
// Обрабатывает: *смеется*, **смеется**, (смеется), [смеется]
// Результат: "смеется, " (запятая и пробел для паузы в TTS)
try {
    // 1. ДВОЙНЫЕ ЗВЁЗДОЧКИ: **смеется** -> смеется,
    {
        static const std::regex re_double_star(R"(\*\*([^*]+)\*\*)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_double_star, "$1,＃");
    }

    // 2. ОДИНАРНЫЕ ЗВЁЗДОЧКИ: *смеется* -> смеется,
    {
        static const std::regex re_star(R"(\*([^*]+)\*)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_star, "$1,＃");
    }

    // 3. КРУГЛЫЕ СКОБКИ: (смеется) -> смеется,
    {
        static const std::regex re_parens(R"(\(([^)]+)\))", std::regex::ECMAScript);
        text = std::regex_replace(text, re_parens, "$1,＃");
    }

    // 4. КВАДРАТНЫЕ СКОБКИ: [смеется] -> смеется,
    {
        static const std::regex re_brackets(R"(\[([^\]]+)\])", std::regex::ECMAScript);
        text = std::regex_replace(text, re_brackets, "$1,＃");
    }

    // 5. Удаляем оставшиеся служебные символы (скобки могли остаться)
    text = replace(text, "*", "");
    text = replace(text, "(", "");
    text = replace(text, ")", "");
    text = replace(text, "[", "");
    text = replace(text, "]", "");

    // 6. Чистка пунктуации (но сохраняем запятые!)

      // Убираем двойные запятые (один regex вместо while)
    static const std::regex re_triple_comma(",,,+", std::regex::ECMAScript);
    text = std::regex_replace(text, re_triple_comma, ",");

    static const std::regex re_comma_space_comma(", ,", std::regex::ECMAScript);
    text = std::regex_replace(text, re_comma_space_comma, ", ");

    // Восстанавливаем запятые из защищённых маркеров и нормализуем пунктуацию
    text = replace(text, ",＃", ",");

    // Убираем пробел между запятой и восклицательным/вопросительным знаком
    text = replace(text, ", !", "!");
    text = replace(text, ", ?", "?");
    text = replace(text, ", .", ".");

    // После этого добавляем пробел после запятой (если это не конец предложения)
    // \s = любой пробельный символ, чтобы не удваивать уже существующие пробелы
    std::regex re_comma_space(R"(,([^\s!?.,:;]))");
    text = std::regex_replace(text, re_comma_space, ", $1");

    // Страховка: схлопываем двойные пробелы, если где-то остались
    static const std::regex re_double_space("  +");
    text = std::regex_replace(text, re_double_space, " ");

    // Убираем запятые в начале строки (если эмоция первое слово)
    if (!text.empty() && text[0] == ',') {
        text.erase(0, 1);
        if (!text.empty() && text[0] == ' ') {
            text.erase(0, 1);
        }
    }

    // 7. Нормализация пробелов (схлопываем множественные)
    {
        static const std::regex re_spaces(R"(\s+)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_spaces, " ");
    }

    // 8. Финальная обрезка пробелов
    trim(text);

} catch (const std::regex_error& e) {
    // Fallback: минимальная очистка
    text = replace(text, "*", " ");
    text = replace(text, "(", " ");
    text = replace(text, ")", " ");
    text = replace(text, "[", " ");
    text = replace(text, "]", " ");
    text = replace(text, "  ", " ");
    trim(text);
}


// ============================================================
// ЭТАП 2: УДАЛЕНИЕ MARKDOWN (СОХРАНЯЕМ СОДЕРЖИМОЕ)
// ============================================================

try {
    // Код-блоки и инлайн-код — оставляем содержимое
    static const std::regex re_code_block(R"(```(.*?)```)", std::regex::ECMAScript);
    static const std::regex re_code_inline(R"(`([^`]*)`)", std::regex::ECMAScript);
    text = std::regex_replace(text, re_code_block, "$1");
    text = std::regex_replace(text, re_code_inline, "$1");

    // Подчёркивания и зачёркнутый — просто убираем маркеры (без пауз)
    // Звёздочки (** и *) уже обработаны в ЭТАПЕ нормализации эмоций (выше)
    static const std::regex re_bold2(R"(__([^_]+)__)", std::regex::ECMAScript);
    static const std::regex re_ital2(R"(_([^_]+)_)", std::regex::ECMAScript);
    static const std::regex re_del(R"(~~([^~]+)~~)", std::regex::ECMAScript);

    text = std::regex_replace(text, re_bold2, "$1");
    text = std::regex_replace(text, re_ital2, "$1");
    text = std::regex_replace(text, re_del, "$1");

    // ... дальше заголовки (без изменений) ...

    // Удаляем висячие маркеры (только подчёркивания и тильды, звёздочки уже обработаны)
    static const std::regex re_multi_unders(R"(_{2,})", std::regex::ECMAScript);
    static const std::regex re_multi_tildes(R"(~{2,})", std::regex::ECMAScript);

    text = std::regex_replace(text, re_multi_unders, " ");
    text = std::regex_replace(text, re_multi_tildes, " ");

} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (Markdown): %s\n", e.what());
    text = replace(text, "```", " ");
    text = replace(text, "`", " ");
    text = replace(text, "__", " ");
    text = replace(text, "~~", " ");
}
trim(text);
if (text.empty()) return;

// ============================================================
// ЭТАП 3: УДАЛЕНИЕ МАРКЕРОВ СПИСКОВ (ваша логика)
// ============================================================
try {
    static const std::regex re_list_markers(
        R"(^\s*(\d+[\.\)]|[A-Za-zА-Яа-яЁё][\.\)]|[\-\*\+\>\|#]+)\s*)",
        std::regex::ECMAScript
    );
    text = std::regex_replace(text, re_list_markers, "");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (list markers): %s\n", e.what());
    // Fallback
    if (text.size() > 2) {
        if (text[0] == '-' || text[0] == '*' || text[0] == '+' || text[0] == '#') {
            if (text[1] == ' ') text = text.substr(2);
        }
        else if (isdigit(text[0])) {
            size_t i = 1;
            while (i < text.size() && isdigit(text[i])) i++;
            if (i < text.size() && (text[i] == '.' || text[i] == ')')) {
                if (i + 1 < text.size() && text[i + 1] == ' ') {
                    text = text.substr(i + 2);
                } else {
                    text = text.substr(i + 1);
                }
            }
        }
    }
}
trim(text);
if (text.empty()) return;

try {
    // ============================================================
    // ШАГ 4.2.1: ЗАЩИТА АНГЛИЙСКИХ СОКРАЩЕНИЙ (don't, it's, we'll и т.д.)
    // ============================================================
    std::vector<std::pair<std::string, std::string>> saved_contractions;
    static const std::regex re_contractions("\\b\\w+'\\w+\\b", std::regex::ECMAScript);
    std::string protected_text;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_contractions);
    auto words_end = std::sregex_iterator();
    size_t last_pos = 0;

    for (auto i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        protected_text += text.substr(last_pos, match.position() - last_pos);
        std::string marker = "🔷CONTR" + std::to_string(saved_contractions.size()) + "🔷";
        saved_contractions.push_back({marker, match.str()});
        protected_text += marker;
        last_pos = match.position() + match.length();
    }
    protected_text += text.substr(last_pos);
    text = protected_text;

    // ============================================================
    // ШАГ 4.2.2: ОБРАБОТКА ВСЕХ ТИПОВ КАВЫЧЕК
    // ============================================================

    // Английские двойные кавычки: "text"
    static const std::regex re_quotes_double("\"([^\"]*)\"", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_double, "$1");

    // Английские одинарные кавычки: 'text'
    static const std::regex re_quotes_single("'([^']*)'", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_single, "$1");

    // Русские кавычки-ёлочки: «text» и »text«
    static const std::regex re_quotes_angle1("«([^»]*)»", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_angle1, "$1");

    static const std::regex re_quotes_angle2("»([^«]*)«", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_angle2, "$1");

    // Немецкие кавычки: „text“ и ‚text‘
    static const std::regex re_quotes_german_double("„([^“]*)“", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_german_double, "$1");

    static const std::regex re_quotes_german_single("‚([^‘]*)‘", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_german_single, "$1");

    // Французские/испанские кавычки: ‹text› и ›text‹
    static const std::regex re_quotes_french_double("‹([^›]*)›", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_french_double, "$1");

    static const std::regex re_quotes_french_single("›([^‹]*)‹", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_french_single, "$1");

    // Японские/китайские кавычки: 「text」 и 『text』
    static const std::regex re_quotes_jp_double("「([^」]*)」", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_jp_double, "$1");

    static const std::regex re_quotes_jp_single("『([^』]*)』", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_jp_single, "$1");

    // Польские кавычки: „text”
    static const std::regex re_quotes_polish("„([^”]*)”", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_polish, "$1");

    // Шведские/финские кавычки: ”text” и ’text’
    static const std::regex re_quotes_swedish_double("”([^”]*)”", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_swedish_double, "$1");

    static const std::regex re_quotes_swedish_single("’([^’]*)’", std::regex::ECMAScript);
    text = std::regex_replace(text, re_quotes_swedish_single, "$1");

    // ============================================================
    // ШАГ 4.2.3: УДАЛЕНИЕ ОСТАВШИХСЯ ОДИНОЧНЫХ КАВЫЧЕК (БЕЗОПАСНО)
    // ============================================================
    // Удаляем только символы, которые заведомо не нужны в тексте
    // НЕ удаляем двойные кавычки — они могут понадобиться для JSON
    // (хотя они уже должны быть удалены или экранированы)
    text = replace(text, "«", "");
    text = replace(text, "»", "");
    text = replace(text, "„", "");
    text = replace(text, "“", "");
    text = replace(text, "‚", "");
    text = replace(text, "‘", "");
    text = replace(text, "‹", "");
    text = replace(text, "›", "");
    text = replace(text, "「", "");
    text = replace(text, "」", "");
    text = replace(text, "『", "");
    text = replace(text, "』", "");
    text = replace(text, "”", "");
    text = replace(text, "’", "");

    // Двойные и одинарные кавычки НЕ удаляем глобально!
    // Они уже обработаны парными regex выше.
    // Одиночные апострофы (don't) защищены и не будут удалены.

    // ============================================================
    // ШАГ 4.2.4: ВОССТАНОВЛЕНИЕ АНГЛИЙСКИХ СОКРАЩЕНИЙ
    // ============================================================
    for (const auto& p : saved_contractions) {
        text = replace(text, p.first, p.second);
    }

    // Нормализация пробелов
    static const std::regex re_spaces("\\s+", std::regex::ECMAScript);
    text = std::regex_replace(text, re_spaces, " ");
    trim(text);

} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (quotes): %s\n", e.what());
    // Fallback: минимальная очистка
    text = replace(text, "«", "");
    text = replace(text, "»", "");
    text = replace(text, "„", "");
    text = replace(text, "“", "");
    text = replace(text, "‹", "");
    text = replace(text, "›", "");
    static const std::regex re_spaces_fallback("\\s+", std::regex::ECMAScript);
    text = std::regex_replace(text, re_spaces_fallback, " ");
    trim(text);
}

// ============================================================
// 4.3 ССЫЛКИ (интеллектуальное преобразование в читаемый текст)
// ============================================================
try {
    // Паттерн для Markdown ссылок: [текст](url)
    static const std::regex re_link_md(R"(\[([^\]]*)\]\(([^)\s]+)\))", std::regex::ECMAScript);

    // Паттерн для голых URL (без Markdown)
    static const std::regex re_bare_url(R"(https?://[^\s<>]+|www\.[^\s<>]+)", std::regex::ECMAScript);

    // ========== ШАГ 1: Обработка Markdown ссылок ==========
    std::string result1;
    auto it1 = std::sregex_iterator(text.begin(), text.end(), re_link_md);
    auto end1 = std::sregex_iterator();
    size_t last_pos = 0;

    for (auto i = it1; i != end1; ++i) {
        std::smatch match = *i;
        result1 += text.substr(last_pos, match.position() - last_pos);

        std::string link_text = match[1].str();
        std::string url = match[2].str();

        // Если текст ссылки осмысленный (>2 символов и не просто "ссылка")
        if (link_text.length() > 2 && link_text != "ссылка" && link_text != "link") {
            result1 += link_text + ", ";
        } else {
            result1 += url_to_speech(url) + ", ";
        }

        last_pos = match.position() + match.length();
    }
    result1 += text.substr(last_pos);
    text = result1;

    // ========== ШАГ 2: Обработка голых URL ==========
    std::string result2;
    auto it2 = std::sregex_iterator(text.begin(), text.end(), re_bare_url);
    auto end2 = std::sregex_iterator();
    last_pos = 0;

    for (auto i = it2; i != end2; ++i) {
        std::smatch match = *i;
        result2 += text.substr(last_pos, match.position() - last_pos);
        result2 += url_to_speech(match.str()) + ", ";
        last_pos = match.position() + match.length();
    }
    result2 += text.substr(last_pos);
    text = result2;

} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (links): %s\n", e.what());
    // Fallback: просто удаляем Markdown-синтаксис
    text = replace(text, "[", " ");
    text = replace(text, "]", " ");
    text = replace(text, "(", " ");
    text = replace(text, ")", " ");
}

// ============================================================
// 4.4 ИЗОБРАЖЕНИЯ (оставляем alt-текст с паузой)
// ============================================================
try {
    static const std::regex re_img_md(R"(!\[([^\]]*)\]\([^)]+\))", std::regex::ECMAScript);
    text = std::regex_replace(text, re_img_md, "$1, ");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (images): %s\n", e.what());
}

// ============================================================
// ЭТАП 5: УДАЛЕНИЕ ФИГУРНЫХ СКОБОК (технический мусор)
// ============================================================
try {
    static const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
    bool changed = true;
    int max_iterations = 100;  // ← ДОБАВЛЕНО: защита от бесконечного цикла
    int iteration = 0;
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        std::string t1 = std::regex_replace(text, re_curly, " ");
        if (t1 != text) {
            text.swap(t1);
            changed = true;
        }
    }
    if (iteration >= max_iterations) {
        fprintf(stderr, "Warning: Too many iterations while removing curly braces, stopping\n");
    }
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (curly): %s\n", e.what());
    text = replace(text, "{", " ");
    text = replace(text, "}", " ");
}
trim(text);
if (text.empty()) return;

// ============================================================
// ЭТАП 6: УДАЛЕНИЕ МУСОРНЫХ СИМВОЛОВ
// ============================================================
try {
    static const std::regex re_noise(R"([#\|\\])", std::regex::ECMAScript);
    text = std::regex_replace(text, re_noise, " ");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (noise): %s\n", e.what());
    text = replace(text, "#", " ");
    text = replace(text, "|", " ");
    text = replace(text, "\\", " ");
}
trim(text);
if (text.empty()) return;

// ============================================================
// ЭТАП 7: XTTS-СПЕЦИФИЧНЫЕ ЗАМЕНЫ (минимально)
// ============================================================

// Точка с запятой -> запятая (XTTS на ; заикается)
text = replace(text, ";", ",");

// Одиночные двойные кавычки — удаляем (XTTS их не понимает)
// ВАЖНО: апострофы ' не трогаем — они защищены в Этапе 4.2.1 (don't, it's)
text = replace(text, "\"", "");

// Убираем пробелы перед ! ? .
try {
    static const std::regex re_space_before_excl(R"(\s+(!))", std::regex::ECMAScript);
    static const std::regex re_space_before_ques(R"(\s+(\?))", std::regex::ECMAScript);
    static const std::regex re_space_before_dot(R"(\s+(\.))", std::regex::ECMAScript);
    text = std::regex_replace(text, re_space_before_excl, "$1");
    text = std::regex_replace(text, re_space_before_ques, "$1");
    text = std::regex_replace(text, re_space_before_dot, "$1");
} catch (const std::regex_error& e) {
    text = replace(text, " !", "!");
    text = replace(text, " ?", "?");
    text = replace(text, " .", ".");
}

// ============================================================
// ЭТАП 7.1: НОРМАЛИЗАЦИЯ МНОГОТОЧИЙ -> ТОЧКА
// ============================================================
try {
    // ШАГ 0: Unicode-многоточие → обычная точка
    text = replace(text, "\xE2\x80\xA6", ".");  // "…" (U+2026) → "."

    // ШАГ 1: Точки с пробелами между ними → убираем пробелы между точками
    // ". . ." → "...", ". ." → "..", " . . . " → "..."
    static const std::regex re_spaced_dots(R"(\.\s+\.)");  // точка-пробел(ы)-точка
    int dots_max_iterations = 10;  // ← защита от бесконечного цикла
    while (dots_max_iterations-- > 0 && std::regex_search(text, re_spaced_dots)) {
        text = std::regex_replace(text, re_spaced_dots, "..");
    }
    if (dots_max_iterations <= 0) {
        fprintf(stderr, "Warning: Too many iterations while normalizing spaced dots, stopping\n");
    }

    // ШАГ 2: Убираем пробелы вокруг групп точек
    // " ... " → "...", "hello ... world" → "hello...world"
    static const std::regex re_spaces_around_dots(R"(\s*(\.{2,})\s*)");
    text = std::regex_replace(text, re_spaces_around_dots, "$1");

    // ШАГ 3: Любые 2+ точек подряд → одна точка
    static const std::regex re_any_dots(R"(\.{2,})");
    text = std::regex_replace(text, re_any_dots, ".");

    // ШАГ 4: Убираем пробел перед точкой (если остался)
    static const std::regex re_space_before_dot(R"(\s+\.)");
    text = std::regex_replace(text, re_space_before_dot, ".");

    // ШАГ 5: Страховка — дублирующиеся точки
    static const std::regex re_double_dots(R"(\.{2,})");
    text = std::regex_replace(text, re_double_dots, ".");

} catch (const std::regex_error& e) {
    // Fallback
    text = replace(text, "\xE2\x80\xA6", ".");
    text = replace(text, ". . .", ".");
    text = replace(text, ". .", ".");
    text = replace(text, "...", ".");
    text = replace(text, "…", ".");
    text = replace(text, "....", ".");
    text = replace(text, " ...", ".");
    text = replace(text, "..", ".");
}

// ============================================================
// ЭТАП 8: МИНИМАЛЬНАЯ НОРМАЛИЗАЦИЯ ПУНКТУАЦИИ
// ============================================================
try {
    // Схлопываем только явные повторы
    static const std::regex re_bangs(R"(!{2,})", std::regex::ECMAScript);
    static const std::regex re_qmarks(R"(\?{2,})", std::regex::ECMAScript);

    text = std::regex_replace(text, re_bangs, "!");
    text = std::regex_replace(text, re_qmarks, "?");

    // Чистим артефакты от наших замен
    text = replace(text, ". ,", ". ");
    text = replace(text, "! ,", "! ");
    text = replace(text, "? ,", "? ");

    // Убираем двойные запятые
    while (text.find(", ,") != std::string::npos) {
        text = replace(text, ", ,", ", ");
    }

} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (punctuation): %s\n", e.what());
}
trim(text);
if (text.empty()) return;

// ============================================================
// ЭТАП 9: НОРМАЛИЗАЦИЯ ПРОБЕЛОВ (финальная)
// ============================================================
try {
    static const std::regex re_spaces(R"(\s+)", std::regex::ECMAScript);
    text = std::regex_replace(text, re_spaces, " ");
} catch (const std::regex_error& e) {
    fprintf(stderr, "Regex error (spaces): %s\n", e.what());
    std::string temp; bool last_was_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_was_space) { temp += ' '; last_was_space = true; }
        } else { temp += c; last_was_space = false; }
    }
    text = temp;
}
trim(text);
if (text.empty()) return;

// ============================================================
// ЭТАП 10: ВОССТАНОВЛЕНИЕ ЗАЩИЩЁННЫХ ПАТТЕРНОВ
// ============================================================
for (const auto& p : protected_patterns) {
    text = replace(text, p.first, p.second);
}

// Восстанавливаем защищённые точки
for (const auto& p : protected_dots) {
    text = replace(text, p.first, p.second);
}

// ============================================================
// ЭТАП 11: ОБРАБОТКА СПИКЕРА
// ============================================================

// Удаляем префикс вида "Эмма: "
std::string prefix = speaker_wav + ":";
if (text.size() >= prefix.size() && text.find(prefix) == 0) {
    size_t pos = prefix.size();
    // Пропускаем пробел после двоеточия, если есть
    if (pos < text.length() && text[pos] == ' ') {
        pos++;
    }
    // Проверяем, что pos не выходит за границы
    if (pos <= text.length()) {
        text = text.substr(pos);
        trim(text);
    }
}

// Финальная нормализация имени спикера (минимальная)
speaker_wav = replace(speaker_wav, ":", "");
speaker_wav = replace(speaker_wav, "\\", "");
speaker_wav = replace(speaker_wav, "\r", "");
speaker_wav = replace(speaker_wav, "\"", "");
speaker_wav = replace(speaker_wav, "/", "_");
speaker_wav = replace(speaker_wav, "<", "_");
speaker_wav = replace(speaker_wav, ">", "_");
speaker_wav = replace(speaker_wav, "|", "_");
speaker_wav = replace(speaker_wav, "?", "_");
speaker_wav = replace(speaker_wav, "*", "_");
trim(speaker_wav);
if (speaker_wav.size() < 2) speaker_wav = "default";

// Финальная зачистка: убираем все переводы строк и невидимые символы
text = replace(text, "\r\n", " ");
text = replace(text, "\r", " ");
text = replace(text, "\n", " ");
trim(text);
if (text.empty()) return;

    // Подготовка JSON
    auto escape_json = [](const std::string& s) -> std::string {
        std::string result; result.reserve(s.size());
        for (unsigned char c : s) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\b': result += "\\b";  break;
                case '\f': result += "\\f";  break;
                case '\n': result += "\\n";  break;
                case '\r': result += "\\r";  break;
                case '\t': result += "\\t";  break;
                default:
                    if (c >= 32 && c != 127) result += static_cast<char>(c);
                    else {
                        char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int)c);
                        result += buf;
                    }
            }
        }
        return result;
    };

    std::string data = "{\"text\":\"" + escape_json(text) + "\", "
                       "\"language\":\"" + escape_json(language) + "\", "
                       "\"speaker_wav\":\"" + escape_json(speaker_wav) + "\"}";

        // Формируем URL и делаем запрос через cURL
    std::string full_url = tts_url + "tts_to_audio/";                          // Собираем полный URL: http://localhost:8020/tts_to_audio/
    CURL* http_handle = curl_easy_init();                                      // Создаём новый cURL handle для HTTP-запроса
    if (http_handle) {                                                         // Проверяем, что handle создан успешно
        struct curl_slist* headers = nullptr;                                  // Список HTTP-заголовков (пока пуст)
        headers = curl_slist_append(headers, "Content-Type: application/json");// Добавляем заголовок: отправляем JSON

        // Настройка таймаутов для стабильности соединения
        curl_easy_setopt(http_handle, CURLOPT_TIMEOUT, 60L);                   // Макс. время на ВЕСЬ запрос: 60 сек (длинные фразы генерируются долго)
        curl_easy_setopt(http_handle, CURLOPT_CONNECTTIMEOUT, 2L);             // Макс. время на TCP-подключение: 2 сек (быстро узнаём что сервер упал)
        curl_easy_setopt(http_handle, CURLOPT_FAILONERROR, 1L);                // Считать HTTP-статусы 4xx/5xx ошибкой (не "успехом")

        curl_easy_setopt(http_handle, CURLOPT_HTTPHEADER, headers);            // Прикрепляем заголовки к запросу
        curl_easy_setopt(http_handle, CURLOPT_URL, full_url.c_str());          // Устанавливаем URL запроса
        curl_easy_setopt(http_handle, CURLOPT_POSTFIELDS, data.c_str());       // Тело запроса: JSON с текстом, языком, голосом
        curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);                    // Отключаем подробный лог cURL (0 = тихо)

        std::string responseData;                                              // Буфер для сохранения ответа сервера
        curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);       // Куда писать данные ответа (в нашу строку)
        curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);   // Функция обратного вызова для записи данных

        CURLcode res = curl_easy_perform(http_handle);                         // ВЫПОЛНЯЕМ ЗАПРОС (блокирующий вызов, ждёт ответа)

        // Ничего не выводим
        (void)res;  // чтобы компилятор не ругался

        /* if (res != CURLE_OK && !(res == CURLE_WRITE_ERROR && g_is_interrupted.load())) {
            static bool tts_error_printed = false;
            if (!tts_error_printed) {
                fprintf(stderr, " [TTS warning: %s]", curl_easy_strerror(res));
                tts_error_printed = true;
            }
         }*/

        curl_slist_free_all(headers);                                          // Освобождаем память, занятую списком заголовков
        curl_easy_cleanup(http_handle);                                        // Освобождаем cURL handle (закрываем соединение)
    } else {
        fprintf(stderr, "Failed to initialize cURL handle\n");                 // Не удалось создать handle (критическая ошибка)
    }
}

// Поток для чтения пользовательского ввода с клавиатуры
void input_thread_func() {
    std::string line;
    std::string buffer;
    bool found_another_line = true;
    while (keyboard_input_running) {
        do {
            // Читаем строку из консоли
            found_another_line = console::readline(line, false);
            buffer += line;
        } while (found_another_line);
        trim(buffer); // Убираем лишние пробелы у введённой строки
        if (!buffer.empty()) { // Проверка на пустую строку
            std::lock_guard<std::mutex> lock(input_mutex); // 🔥 ФИКС: защищаем запись
            input_queue.push(buffer);
            buffer = ""; // Очищаем буфер
        }
    }
}

// Только для Windows: проверяет, фокусировано ли окно консоли
bool IsConsoleWindowFocused() {
    HWND console_window = GetConsoleWindow();
    if (console_window == NULL) {
        return false;  // Окно консоли не найдено
    }
    HWND foreground_window = GetForegroundWindow();
    if (foreground_window == NULL) {
        return false;  // Нет активного окна
    }
    return (console_window == foreground_window);
}
	// Стоп: Ctrl+Space
	// Перегенерировать: Ctrl+Right
	// Удалить: Ctrl+Delete
	// Сбросить: Ctrl+R
	// Функция обработки горячих клавиш, изменяет глобальную переменную g_hotkey_pressed
	// Логика отслеживания нажатий Ctrl+..., Alt и т.д.

void keyboard_shortcut_func() {
	// Логика отслеживания нажатий Ctrl+..., Alt и т.д.
    // Подробности ниже...
    bool b_ctr_space_processed = false;
    bool b_ctr_right_processed = false;
    bool b_ctr_delete_processed = false;
    bool b_ctr_r_processed = false;
    bool b_ctr_space_prev = false;
    bool b_ctr_right_prev = false;
    bool b_ctr_delete_prev = false;
    bool b_ctr_r_prev = false;
    bool b_ctr_space = false;
    bool b_ctr_right = false;
    bool b_ctr_delete = false;
    bool b_ctr_r = false;
    bool b_alt = false;
    bool isFocused = false;

    { // Инициализация глобальной переменной
        std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
        g_hotkey_pressed = "";
    }

    while (g_shortcut_thread_running.load()) {
        isFocused = IsConsoleWindowFocused();
        if (isFocused) {
            b_ctr_space = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SPACE) & 0x8000);
            b_ctr_right = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_RIGHT) & 0x8000);
            b_ctr_delete = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_DELETE) & 0x8000);
            b_ctr_r = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState('R') & 0x8000);
            b_alt = GetAsyncKeyState(VK_MENU) & 0x8000;

            if (b_alt) { // Обработка Alt (Push-to-Talk)
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);

                    // Устанавливаем Alt только если нет других активных горячих клавиш
                    if (g_hotkey_pressed.empty() || g_hotkey_pressed == "Alt") {
                        g_hotkey_pressed = "Alt";
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // Обработка Ctrl+Space (Стоп)
            if (b_ctr_space && !b_ctr_space_prev) {
                if (!b_ctr_space_processed) {
                    fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);
					{
                        std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                        g_hotkey_pressed = "Ctrl+Space";
                    }
                    b_ctr_space_processed = true;
                }
            }
            else if (!b_ctr_space && b_ctr_space_prev && b_ctr_space_processed) {
                b_ctr_space_processed = false;
            }

            if (b_ctr_right && !b_ctr_right_prev) {
                if (!b_ctr_right_processed) {
					fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
					g_hotkey_pressed = "Ctrl+Right";
                    b_ctr_right_processed = true;
                }
            }
            else if (!b_ctr_right && b_ctr_right_prev && b_ctr_right_processed) {
                b_ctr_right_processed = false;
            }

            if (b_ctr_delete && !b_ctr_delete_prev) {
                if (!b_ctr_delete_processed) {
					fflush(stdout);
					printf("\b"); // remove printed symbols
					fflush(stdout);
                    	std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
					g_hotkey_pressed = "Ctrl+Delete";
                    b_ctr_delete_processed = true;
                }
            }
            else if (!b_ctr_delete && b_ctr_delete_prev && b_ctr_delete_processed) {
                b_ctr_delete_processed = false;
            }

            if (b_ctr_r && !b_ctr_r_prev) {
                if (!b_ctr_r_processed) {
					fflush(stdout);
					printf("\b\b"); // remove printed ^R
					fflush(stdout);
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
					g_hotkey_pressed = "Ctrl+R";
                    b_ctr_r_processed = true;
                }
            }
            else if (!b_ctr_r && b_ctr_r_prev && b_ctr_r_processed) {
                b_ctr_r_processed = false;
            }

            b_ctr_space_prev = b_ctr_space;
            b_ctr_right_prev = b_ctr_right;
            b_ctr_delete_prev = b_ctr_delete;
            b_ctr_r_prev = b_ctr_r;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // добавлен выход из функции при завершении цикла
    // поток завершается автоматически при выходе из функции
}


// Шаблоны промптов для Whisper (будут дополнены информацией о роде пользователя)
const std::string k_prompt_whisper_ru = R"({1}: разговор с голосовым ассистентом. Распознавай только речь {0}.)";
const std::string k_prompt_whisper    = R"({1}: conversation with voice assistant. Recognize only {0}'s speech.)";

const std::string k_prompt_llama = R"({1} — дружелюбный и умный помощник. {1} отвечает кратко, по делу, только текстом. Без скобок, звёздочек и других спецсимволов.
{0}{4} Привет, {1}!
{1}{4} Привет! Как дела?
{0}{4} Который час?
{1}{4} Сейчас {2}.
{0}{4} Какая сегодня дата?
{1}{4} {5}, {3}.
{0}{4})";

int run(int argc, char ** argv) {
whisper_params params;          // параметры Whisper
std::vector<std::thread> threads;
std::thread t;

int reply_part = 0;
bool last_output_has_username = false;

int input_tokens_count = 0;

// Инициализация указателей на сэмплеры (nullptr для безопасной очистки)
llama_sampler * smpl = nullptr;
llama_sampler * smpl_high_temp = nullptr;

	// HWND cur_window_handle больше не нужен — получаем хендл консоли внутри IsConsoleWindowFocused()

    if (whisper_params_parse(argc, argv, params) == false) {
            return 1;
        }

        // Проверяем, поддерживается ли указанный язык
        if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
            fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }

#ifdef _WIN32
    init_console_colors();
#endif

	allow_xtts_file(params.xtts_control_path, 1);  // разрешаем воспроизведение звука XTTS

    // Инициализация Whisper
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
    if (!ctx_wsp) {
        fprintf(stderr, "No whisper.cpp model specified. Please provide using -mw <modelfile>\n");
        return 1;
    }

    // Инициализация LLaMA модели
    llama_backend_init();

    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }

	lmparams.main_gpu = params.main_gpu;
	if (params.split_mode == "layer") lmparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
	else lmparams.split_mode = LLAMA_SPLIT_MODE_NONE;

    lmparams.tensor_split = params.tensor_split.empty() ? nullptr : params.tensor_split.data();

    struct llama_model * model_llama = llama_model_load_from_file(params.model_llama.c_str(), lmparams);
    if (!model_llama) {
        fprintf(stderr, "No llama.cpp model specified. Please provide using -ml <modelfile>\n");
        return 1;
    }
    // Очистка не требуется - вектор сам управляет памятью
    params.tensor_split.clear();

    const llama_vocab * vocab_llama = llama_model_get_vocab(model_llama);


	bool add_bos_token = llama_vocab_get_add_bos(vocab_llama);
	const int n_keep   = params.n_keep + add_bos_token;

    llama_context_params lcparams = llama_context_default_params();

    // настройте их по своему вкусу
    lcparams.n_ctx      = params.ctx_size; // 2048 по умолчанию
    fprintf(stdout, "n_ctx %d", lcparams.n_ctx); // Выводим фактический размер контекста

    lcparams.n_threads  = params.n_threads;
    //lcparams.n_batch    = params.batch_size; // Значение 512 было задано по умолчанию

    // === НОВЫЙ КОД (вместо lcparams.flash_attn = params.flash_attn) ===
    // В llama.cpp >= 1.80 поле flash_attn убрали — теперь используется flash_attn_type.
    // Чтобы сохранить поведение старого флага --flash-attn: если пользователь явно
    // запросил flash-attn, ставим AUTO (llama сама решит, можно ли включить).
    // Если флаг не указан — принудительно выключаем.
    lcparams.flash_attn_type = params.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_AUTO
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    struct llama_context * ctx_llama = llama_init_from_model(model_llama, lcparams);

    if (!ctx_llama) {
        fprintf(stderr, "error: failed to initialize llama context\n");
        return 1;
    }

    // ============================================================
    // СПЕЦИАЛЬНЫЕ ТОКЕНЫ ДЛЯ ОСТАНОВКИ ГЕНЕРАЦИИ
    // ============================================================
    // Заполняем независимо от наличия JSON-пресета.
    // Это обеспечивает чистую работу даже без --instruct-preset.
    // ============================================================
    llama_token special_token_ids[64] = {0};
    int special_token_count = 0;

    // 1. Токены из JSON-пресета (если есть)
    if (!params.instruct_preset_data["bot_message_suffix"].empty()) {
        std::vector<llama_token> tokens = ::llama_tokenize(ctx_llama, params.instruct_preset_data["bot_message_suffix"], false);
        if (!tokens.empty() && special_token_count < 64) {
            special_token_ids[special_token_count++] = tokens[0];
        }
    }
    if (!params.instruct_preset_data["stop_sequence"].empty()) {
        std::vector<llama_token> tokens = ::llama_tokenize(ctx_llama, params.instruct_preset_data["stop_sequence"], false);
        if (!tokens.empty() && special_token_count < 64) {
            bool already = false;
            for (int j = 0; j < special_token_count; j++) {
                if (special_token_ids[j] == tokens[0]) { already = true; break; }
            }
            if (!already) {
                special_token_ids[special_token_count++] = tokens[0];
            }
        }
    }

    // 2. Базовые EOT-маркеры (ChatML, Llama, Gemma, Mistral, Qwen)
    //    Эти паттерны добавляются ВСЕГДА, даже если пресет не указан.
    const char* special_patterns[] = {
        "<|eot_id|>",        // ChatML, Llama 3, Yandex
        "<|im_end|>",        // Qwen
        "</s>",              // Llama 2, Mistral
        "<end_of_turn|>",    // Gemma
        "<|endoftext|>",     // Некоторые модели
        "<|im_start|>",      // Начало сообщения (тоже стоп-сигнал)
        "<|end|>",           // Альтернативный маркер
    };

    for (const char* pattern : special_patterns) {
        std::vector<llama_token> tokens = ::llama_tokenize(ctx_llama, pattern, false);
        if (!tokens.empty()) {
            bool already = false;
            for (int j = 0; j < special_token_count; j++) {
                if (special_token_ids[j] == tokens[0]) { already = true; break; }
            }
            if (!already && special_token_count < 64) {
                special_token_ids[special_token_count++] = tokens[0];
            }
        }
    }

    // 3. Пользовательские стоп-слова из --stop-words (токенизированные)
    if (!params.stop_words.empty()) {
        std::vector<llama_token> tokens = ::llama_tokenize(ctx_llama, params.stop_words, false);
        for (llama_token t : tokens) {
            if (special_token_count >= 64) break;
            bool already = false;
            for (int j = 0; j < special_token_count; j++) {
                if (special_token_ids[j] == t) { already = true; break; }
            }
            if (!already) {
                special_token_ids[special_token_count++] = t;
            }
        }
    }

    // Отладочный вывод
    if (params.debug && special_token_count > 0) {
        printf("[DEBUG] Special token IDs to filter: ");
        for (int i = 0; i < special_token_count; i++) {
            printf("%d ", special_token_ids[i]);
        }
        printf("\n");
    }
    // ============================================================

    // распечатать некоторую информацию об обработке
    {
    fprintf(stderr, "\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // =================================================================
    // ИНИЦИАЛИЗАЦИЯ АУДИОБУФЕРА
    // =================================================================
    // Создаём асинхронный аудиобуфер длительностью 15 секунд
    // audio_async — это класс, который в реальном времени захватывает звук
    // с микрофона и сохраняет в кольцевом буфере
    // =================================================================
    audio_async audio(15 * 1000);

    // =================================================================
    // ПОДКЛЮЧЕНИЕ К АУДИОУСТРОЙСТВУ
    // =================================================================
    // Пытаемся инициализировать аудиоустройство с указанным ID захвата
    // Если capture_id == -1, используется устройство по умолчанию
    // WHISPER_SAMPLE_RATE = 16000 Гц (частота, ожидаемая моделью Whisper)
    // =================================================================
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: Ошибка инициализации аудиоустройства (ID: %d)\n",
                __func__, params.capture_id);
        fprintf(stderr, "Проверьте доступные аудиоустройства и правильность ID захвата\n");
        fprintf(stderr, "Для списка устройств запустите программу с параметром --list-devices\n");
        return 1;
    }

    // =================================================================
    // ЗАПУСК ЗАХВАТА АУДИО
    // =================================================================
    // resume() начинает заполнять буфер данными с микрофона
    // Без этого вызова audio.get() будет возвращать пустые данные
    // =================================================================
    audio.resume();

bool is_running  = true;
bool force_speak = false;
float prob0 = 0.0f;
const std::string chat_symb = ":";
std::vector<float> pcmf32_cur;
std::vector<float> pcmf32_prev;
std::vector<float> pcmf32_prompt;

    // Инициализируем промпт для Whisper — он должен знать род пользователя
    std::string prompt_whisper;

    // Определяем пол пользователя: женский если есть признаки, иначе мужской
    bool user_is_female = false;
    std::string user_lower = params.person;
    std::transform(user_lower.begin(), user_lower.end(), user_lower.begin(), ::tolower);

    if (user_lower.length() >= 1) {
        char last_char = user_lower.back();
        // Женские окончания: -а, -я, -ь
        if (last_char == 'а' || last_char == 'я' || last_char == 'ь') {
            user_is_female = true;
        }
        // Женское окончание: -ия
        else if (user_lower.length() >= 2 && user_lower.substr(user_lower.length() - 2) == "ия") {
            user_is_female = true;
        }
        // Исключения: мужские имена на -а/-я
        if (user_is_female) {
            static const std::unordered_set<std::string> male_exceptions = {
                "никита", "илья", "фома", "лука", "кузьма", "добрыня"
            };
            if (male_exceptions.find(user_lower) != male_exceptions.end()) {
                user_is_female = false;
            }
        }
    }

    // Формируем промпт для Whisper на основе констант
    if (params.language == "ru") {
        prompt_whisper = k_prompt_whisper_ru;
        if (user_is_female) {
            prompt_whisper += " {0} говорит о себе в женском роде: сделала, пошла, сказала.";
        } else {
            prompt_whisper += " {0} говорит о себе в мужском роде: сделал, пошёл, сказал.";
        }
        prompt_whisper += " {0} общается с голосовым ассистентом {1}. Речь: короткие фразы, вопросы, просьбы.";
    } else {
        prompt_whisper = k_prompt_whisper;
        if (user_is_female) {
            prompt_whisper += " {0} is female: she did, she went, she said.";
        } else {
            prompt_whisper += " {0} is male: he did, he went, he said.";
        }
        prompt_whisper += " {0} talks to voice assistant {1}. Speech: short phrases, questions, requests.";
    }
    prompt_whisper = ::replace(prompt_whisper, "{0}", params.person);
    prompt_whisper = ::replace(prompt_whisper, "{1}", params.bot_name);

    // Конструируем начальный промпт для LLaMA
    // 1. Берем базовый текст
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

// Режим инструкций
if (!params.instruct_preset.empty())
{
        try {
            std::string filename = "instruct_presets/" + params.instruct_preset + ".json";
        nlohmann::json jsonData;
            std::ifstream jsonFile(filename);

            if (jsonFile.is_open()) {
                jsonFile >> jsonData;
            jsonFile.close();
                params.instruct_preset_data = jsonData;
        } else { // не найден
            std::cout << "Warning: preset file '" << filename << "' does not exist. Turning off instruct mode" << std::endl;
                params.instruct_preset = "";
            }
    }
    catch (const std::exception &e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return 1;
    }
}
    else // не передан
        {
            params.instruct_preset = "";
        }
    //Нужен начальный пробел ' '
    prompt_llama.insert(0, 1, ' ');

    // Определяем пол бота для правильного склонения в ответах
    bool bot_is_female = false;
    std::string bot_lower = params.bot_name;
    std::transform(bot_lower.begin(), bot_lower.end(), bot_lower.begin(), ::tolower);

    if (bot_lower.length() >= 1) {
        char last_char = bot_lower.back();
        // Женские окончания: -а, -я, -ь
        if (last_char == 'а' || last_char == 'я' || last_char == 'ь') {
            bot_is_female = true;
        }
        // Женское окончание: -ия
        else if (bot_lower.length() >= 2 && bot_lower.substr(bot_lower.length() - 2) == "ия") {
            bot_is_female = true;
        }
        // Исключения: мужские имена на -а/-я
        if (bot_is_female) {
            static const std::unordered_set<std::string> male_exceptions = {
                "никита", "илья", "фома", "лука", "кузьма", "добрыня"
            };
            if (male_exceptions.find(bot_lower) != male_exceptions.end()) {
                bot_is_female = false;
            }
        }
    }

    // ============================================================
    // Разделение Raw/Instruct для начального промпта
    // Подсказка о роде вставляется ПЕРЕД примерами диалога,
    // а НЕ дописывается в конец (где она ломала структуру)
    // ============================================================
    if (params.instruct_preset.empty()) {
        // RAW-РЕЖИМ: заменяем плейсхолдеры {0} и {1} здесь
        prompt_llama = ::replace(prompt_llama, "{0}", params.person);
        prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

        // Грамматическая подсказка (без запретов) — вставляем ТОЛЬКО если
        // в промпте ещё нет явного указания рода.
        std::string gender_hint;

        // Проверяем, не содержит ли промпт уже гендерную инструкцию
        bool already_has_gender = false;
        {
            std::string lower_prompt = prompt_llama;
            std::transform(lower_prompt.begin(), lower_prompt.end(), lower_prompt.begin(), ::tolower);
            if (params.language == "ru") {
                if (lower_prompt.find("женский род") != std::string::npos ||
                    lower_prompt.find("мужской род") != std::string::npos) {
                    already_has_gender = true;
                }
            } else {
                if (lower_prompt.find("female gender") != std::string::npos ||
                    lower_prompt.find("male gender") != std::string::npos) {
                    already_has_gender = true;
                }
            }
        }

        if (!already_has_gender) {
            if (params.language == "ru") {
                if (bot_is_female) {
                    gender_hint = "\n[Ты женщина. Говори в женском роде: сделала, сказала, подумала, пошла.]\n";
                } else {
                    gender_hint = "\n[Ты мужчина. Говори в мужском роде: сделал, сказал, подумал, пошёл.]\n";
                }
            } else {
                if (bot_is_female) {
                    gender_hint = "\n[You are female. Use: she did, she said, she thought.]\n";
                } else {
                    gender_hint = "\n[You are male. Use: he did, he said, he thought.]\n";
                }
            }

            // Вставляем перед первым примером диалога
            size_t insert_pos = prompt_llama.find(params.person + chat_symb);
            if (insert_pos != std::string::npos && insert_pos > 0) {
                while (insert_pos > 0 && prompt_llama[insert_pos - 1] != '\n') {
                    insert_pos--;
                }
                prompt_llama.insert(insert_pos, gender_hint);
            }
        }
    }
    // INSTRUCT-РЕЖИМ: промпт из файла уже в ChatML-формате с именами,
    // грамматический род не добавляем — модель знает пол из контекста
    // ============================================================

    // ВЫНОСИМ ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ НАРУЖУ (будут видны в цикле генерации)
    std::string time_str, year_str, ymd;

    // Получаем текущее время и дату
    {
        time_t t = time(0);
        struct tm * now = localtime(&t);
        char buf[128];

        // {2} — время
        strftime(buf, sizeof(buf), "%H:%M", now);
        time_str = buf;

        // {3} — год
        strftime(buf, sizeof(buf), "%Y", now);
        year_str = buf;

        // {5} — дата по-русски
        strftime(buf, sizeof(buf), "%d %B %Y года", now);
        std::string ymd_str = buf;
        ymd_str = ::replace(ymd_str, "January",   "января");
        ymd_str = ::replace(ymd_str, "February",  "февраля");
        ymd_str = ::replace(ymd_str, "March",     "марта");
        ymd_str = ::replace(ymd_str, "April",     "апреля");
        ymd_str = ::replace(ymd_str, "May",       "мая");
        ymd_str = ::replace(ymd_str, "June",      "июня");
        ymd_str = ::replace(ymd_str, "July",      "июля");
        ymd_str = ::replace(ymd_str, "August",    "августа");
        ymd_str = ::replace(ymd_str, "September", "сентября");
        ymd_str = ::replace(ymd_str, "October",   "октября");
        ymd_str = ::replace(ymd_str, "November",  "ноября");
        ymd_str = ::replace(ymd_str, "December",  "декабря");
        ymd = ymd_str;
    }

    // ============================================================
    // ПАТЧ №1 (продолжение): замена {2}{3}{4}{5} здесь, где переменные уже доступны
    // ============================================================
    if (params.instruct_preset.empty()) {
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
        prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);
        prompt_llama = ::replace(prompt_llama, "{5}", ymd);
    }
    // В Instruct-режиме дата/время уже вписаны в промпт-файл статично
    // ============================================================


    // llama_batch batch = llama_batch_init(2048, 0, 1); // <-- ВСЕГДА ИНИЦИАЛИЗИРУЕМ С n_tokens=0!
    llama_batch batch = llama_batch_init(params.ctx_size, 0, 1);

    fprintf(stdout, "llama_n_ctx %d", llama_n_ctx(ctx_llama));

    // Инициализация сэмплера
	const float top_k          = params.top_k;
	const float top_p          = params.top_p;
	const float min_p          = params.min_p;
	float temp                 = params.temp;
	const float repeat_penalty = params.repeat_penalty;
    const int seed = 0;
    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);           // ← без llama_sampler*
    smpl_high_temp = llama_sampler_chain_init(sparams); // ← без llama_sampler*

    if (temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp (temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist (seed));

		llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_temp (2.00));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_dist (seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_greedy());
    }

    // Инициализация сессии генерации текста с использованием модели LLaMA
    // Получаем путь к файлу сессии из параметров
    std::string path_session = params.path_session;
    // Вектор токенов для хранения текущей сессии
    std::vector<llama_token> session_tokens;

// -------------------------------------------------------------------------
// УНИВЕРСАЛЬНАЯ ПОДДЕРЖКА INSTRUCT-PRESET (любой формат из JSON)
// -------------------------------------------------------------------------
if (!params.instruct_preset.empty()) {
    std::string wrapped_prompt;

    // 1. System prompt (если задан --prompt-file и в пресете есть префикс/суффикс)
    // ИСПРАВЛЕНО: используем prompt_llama вместо params.prompt
    if (!prompt_llama.empty() &&
        !params.instruct_preset_data["system_prompt_prefix"].empty()) {
        wrapped_prompt += params.instruct_preset_data["system_prompt_prefix"];
        wrapped_prompt += prompt_llama;
        wrapped_prompt += params.instruct_preset_data["system_prompt_suffix"];
    }

    // 2. User message — только если НЕТ кастомного промпта
    //    (проверяем params.prompt, он может быть пустым или нет)
    if (params.prompt.empty()) {
        if (!params.instruct_preset_data["user_message_prefix"].empty()) {
            wrapped_prompt += params.instruct_preset_data["user_message_prefix"];
        }
        // Добавляем пользовательскую часть ТОЛЬКО если system не был добавлен
        // Иначе контекст уже содержит всё необходимое
        if (prompt_llama.empty() || wrapped_prompt.find(prompt_llama) == std::string::npos) {
            wrapped_prompt += prompt_llama;
        }
        if (!params.instruct_preset_data["user_message_suffix"].empty()) {
            wrapped_prompt += params.instruct_preset_data["user_message_suffix"];
        }
    }

    // 3. Assistant prefix (модель продолжит отсюда)
    if (!params.instruct_preset_data["bot_message_prefix"].empty()) {
        wrapped_prompt += params.instruct_preset_data["bot_message_prefix"];
    }

    // Заменяем исходный промпт
    prompt_llama = wrapped_prompt;
}

    // Токенизируем входной промпт (prompt_llama) в последовательность токенов
    auto embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

    // Безопасное ограничение длины контекста модели ---
    // Цель: предотвратить переполнение контекста (ctx_size) и зацикливание на длинных диалогах
    // Если количество токенов приближается к лимиту контекста модели
    if ((int)embd_inp.size() > params.ctx_size - 512) {
        // Количество токенов, которые нужно сохранить в начале (system prompt, ChatML заголовки)
        int keep = std::min(params.n_keep, (int)embd_inp.size());
        // Обрезаем середину контекста, сохраняя начало и хвост
        if ((int)embd_inp.size() > keep + 256) {
            embd_inp.erase(embd_inp.begin() + keep, embd_inp.end() - 256);
        }

        std::cerr << "[warn] Context trimmed: " << embd_inp.size()
                << " tokens (ctx limit " << params.ctx_size << ")\n";
    }

    // --- Контроль повторов (repeat_last_n) ---
    // Для предотвращения зацикливания фраз сохраняем последние токены
    static std::vector<llama_token> recent_tokens;

    if (params.repeat_last_n > 0) {
        if ((int)embd_inp.size() > params.repeat_last_n) {
            recent_tokens.assign(embd_inp.end() - params.repeat_last_n, embd_inp.end());
        } else {
            recent_tokens = embd_inp;
        }
    }
    // Примечание: recent_tokens можно использовать далее при генерации
    // чтобы вручную penalize часто повторяющиеся токены,
    // но сам по себе этот блок уже предотвращает потерю контекста.
    // Если путь к файлу сессии указан
    if (!path_session.empty()) {
    // Сообщаем о попытке загрузить сохранённую сессию
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, path_session.c_str());
    // Пытаемся открыть файл сессии для проверки его наличия
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
        std::fclose(fp); // Закрываем файл, так как он существует
        // Подготавливаем вектор для хранения токенов из сохранённой сессии
        session_tokens.resize(llama_n_ctx(ctx_llama)); // Устанавливаем размер, равный максимальному контексту модели
            size_t n_token_count_out = 0;
        // Загружаем состояние модели из файла сессии
        // Используем .size(), так как мы только что установили его через resize()
            if (!llama_state_load_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size(), &n_token_count_out)) {
            // Если загрузка не удалась — выводим сообщение об ошибке
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
        // Корректируем размер вектора под реальное количество загруженных токенов
            session_tokens.resize(n_token_count_out);
        // Копируем токены из сессии в входной буфер
        // assign автоматически выделяет нужный объем памяти и копирует данные
        embd_inp.assign(session_tokens.begin(), session_tokens.end());
        // Сообщаем о успешной загрузке сессии и выводим количество токенов
            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
        // Если файл сессии не найден — сообщаем, что будет создан новый
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

/// Оценка начального промпта
printf("\n");
printf("%s : initializing - please wait ...\n", __func__);
float llama_start_time = get_current_time_ms();
int n_past = 0;
// Инициализируем batch для начальной оценки промпта
batch = llama_batch_init(2048, 0, 1); // ←  Инициализируем с запасом (2048 — это n_ctx, макс. размер)

// Подготовка батча для декодирования промпта
{
    // ===== ПРОВЕРКА РАЗМЕРА =====
    if (embd_inp.size() > 2048) {
        fprintf(stderr, "FATAL: Initial prompt size (%zu tokens) exceeds batch limit (2048)\n",
                embd_inp.size());
        fprintf(stderr, "Please reduce prompt size or increase batch limit in code.\n");
        return 1;
    }
    // ===========================
    batch.n_tokens = embd_inp.size();

    for (int i = 0; i < batch.n_tokens; i++) {
        batch.token[i]     = embd_inp[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == batch.n_tokens - 1;
    }
}

if (llama_decode(ctx_llama, batch)) {
    fprintf(stderr, "%s : failed to decode\n", __func__);
    return 1;
}

	float llama_end_time = get_current_time_ms();
	float llama_time_total = 0;
	float llama_time_input = 0;
	float llama_time_output = 0;

	llama_time_total = llama_end_time - llama_start_time;

    printf(" \nLlama start prompt: %zu/%d tokens in %.3f s at %.0f t/s\n",

        embd_inp.size(),
       (int)params.ctx_size, // Предполагаем, что params.ctx_size это int32_t или совместимый тип
       (double)llama_time_total, // Явное приведение float к double для устранения предупреждения
       (double)(embd_inp.size() / llama_time_total)); // Результат деления size_t/float -> double

    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s", prompt_llama.c_str());
        fflush(stdout);
    }

     // Сообщение о совпадении сессии, если применимо
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

	// HACK - так как сохранение сессии занимает время, мы не будем его повторно сохранять,
    // если уже загружена сессия с 75% совпадением
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);
    printf("%s : done! start speaking in the microphone\n", __func__);

    // показывать команду пробуждения, если она включена
    const std::string wake_cmd = params.wake_cmd;
    if (!wake_cmd.empty()) {
        printf("%s : the wake-up command is: '%s%s%s'\n", __func__, "\033[1m", wake_cmd.c_str(), "\033[0m");
    }

    // ---------------------------------------------
    // ===== НАЧАЛО ДИАЛОГА: приглашение выводится ПОСЛЕ этого блока =====
    printf("\n"); // ← просто пустая строка перед началом диалога, поэтому здесь не выводим лишний "Друг:"
    fflush(stdout);
    // ---------------------------------------------

    // Очистка аудио-буфера
    audio.clear();

    // Переменные для текстового вывода
    const int voice_id = 2;

    // ПЕРЕД генерацией сбрасываем ОБА флага прерывания
    g_is_interrupted.store(false);   // Чтобы cURL снова начал качать аудио
    llama_interrupted.store(0);      // Чтобы Llama снова начала генерировать токены
    const int n_ctx = llama_n_ctx(ctx_llama);

    n_past = embd_inp.size();
    int n_prev = 64; // TODO arg
    std::vector<int> past_prev_arr{};
    int n_past_prev = 0; // количество токенов, которое было перед последним ответом
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;
    std::vector<llama_token> embd;
	std::string text_heard_prev;
	std::string text_heard_trimmed;
	int new_command_allowed = 1;
	std::string google_resp;
	std::vector<std::string> tts_intros;
	std::string rand_intro_text = "";
	std::string last_output_buffer = "";
	std::string last_output_needle = "";
	if (params.language == "ru")
	{
		tts_intros = {"Хм", "Ну", "Нуу", "О", "А", "А?", "Угу", "Ох", "Ха", "Ах", "Блин", "Короче", "В общем", "Ой", "Слышь", "Ну вообще-то", "Ну а вообще", "Кароче", "Вот", "Знаешь", "Как бы", "Прикинь", "Послушай", "Типа", "Это", "Так вот", "Погоди", params.person};
	}
	else
	{
		tts_intros = {"Hm", "Hmm", "Well", "Well well", "Huh", "Ugh", "Uh", "Um", "Mmm", "Oh", "Ooh", "Haha", "Ha ha", "Ahh", "Whoa", "Really", "I mean", "By the way", "Anyway", "So", "Actually", "Uh-huh", "Seriously", "Whatever", "Ahh", "Like", "But", "You know", "Wait", "Ahem", "Damn", params.person};
	}

    // Современный генератор случайных чисел (потокобезопасный через thread_local)
    // Объявляем здесь, чтобы использовать во всём основном цикле
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist;

	int last_command_time = 0;
	std::string current_voice = params.xtts_voice;

    // === АНТИПРОМПТЫ (только для смены говорящего) ===
    // Останавливаем генерацию, когда модель сгенерировала начало реплики пользователя
    // или перевод строки (новый абзац). Спецтокены типа <|eot_id|> обрабатываются
    // отдельно через special_token_ids (остановка по ID).
    std::vector<std::string> antiprompts;
    if (!params.allow_newline) {
        antiprompts.push_back("\n");               // стоп-слово только если новые строки запрещены
    }
    antiprompts.push_back("\n" + params.person + chat_symb);       // "\nДруг:"
    antiprompts.push_back("\n" + params.person + " " + chat_symb); // "\nДруг :"

    // Пользовательские стоп-слова из --stop-words (если нужны)
    if (!params.stop_words.empty()) {
        size_t start = 0, end = params.stop_words.find(';');
        auto add_word = [&](std::string w) {
            if (w.length() >= 2) {
                w = ::replace(w, "\\r", "\r");
                w = ::replace(w, "\\n", "\n");
                if (std::find(antiprompts.begin(), antiprompts.end(), w) == antiprompts.end()) {
                    antiprompts.push_back(w);
                }
            }
        };
        if (end == std::string::npos) {
            add_word(params.stop_words);
        } else {
            while (start < params.stop_words.size()) {
                std::string word = params.stop_words.substr(start, end - start);
                add_word(word);
                start = end + 1;
                end = params.stop_words.find(';', start);
                if (end == std::string::npos) end = params.stop_words.size();
            }
        }
    }

    // === ФУНКЦИЯ ДЛЯ ОБНОВЛЕНИЯ АНТИПРОМПТОВ ПРИ СМЕНЕ БОТА ===
    // Вызывается при команде "call" в режиме --multi-chars.
    // Обновляет имя пользователя И добавляет стоп-слова для старого/нового бота.
    auto update_antiprompts = [&](const std::string& new_person, const std::string& new_bot_name) {
        // 1. Обновляем антипромпты для имени пользователя
        //    Индексы зависят от того, есть ли начальный "\n" в antiprompts.
        size_t user_offset = (!antiprompts.empty() && antiprompts[0] == "\n") ? 1 : 0;

        if (antiprompts.size() >= user_offset + 2) {
            antiprompts[user_offset]     = "\n" + new_person + chat_symb;       // "\nНовыйДруг:"
            antiprompts[user_offset + 1] = "\n" + new_person + " " + chat_symb; // "\nНовыйДруг :"
        }

        // 2. Добавляем антипромпты для старого имени бота (чтобы модель не продолжала его речь)
        //    Это предотвращает ситуацию, когда модель говорит и от старого, и от нового бота.
        std::string old_bot_pattern1 = "\n" + params.bot_name + chat_symb;
        std::string old_bot_pattern2 = "\n" + params.bot_name + " " + chat_symb;

        bool found1 = false, found2 = false;
        for (const auto& ap : antiprompts) {
            if (ap == old_bot_pattern1) found1 = true;
            if (ap == old_bot_pattern2) found2 = true;
        }

        if (!found1 && !old_bot_pattern1.empty() && old_bot_pattern1 != "\n" + new_person + chat_symb) {
            antiprompts.push_back(old_bot_pattern1);
        }
        if (!found2 && !old_bot_pattern2.empty() && old_bot_pattern2 != "\n" + new_person + " " + chat_symb) {
            antiprompts.push_back(old_bot_pattern2);
        }

        // 3. Опционально: добавляем стоп-последовательность из JSON-пресета
        if (!params.instruct_preset_data["stop_sequence"].empty()) {
            std::string stop_seq = params.instruct_preset_data["stop_sequence"];
            bool stop_seq_found = false;
            for (const auto& ap : antiprompts) {
                if (ap == stop_seq) { stop_seq_found = true; break; }
            }
            if (!stop_seq_found && !stop_seq.empty()) {
                antiprompts.push_back(stop_seq);
            }
        }

        if (params.verbose) {
            printf("\n[DEBUG] Antiprompts updated. New bot: '%s'. Total antiprompts: %zu\n",
                   new_bot_name.c_str(), antiprompts.size());
        }
    };

    // Отладка: выводим ВСЕ стоп-слова для полного контроля
    printf("Llama stop words (%zu total): ", antiprompts.size());
    for (size_t i = 0; i < antiprompts.size(); i++) {
        // Экранируем ВСЕ спецсимволы для читаемости
        std::string display = antiprompts[i];
        // Заменяем \n, \r, \t на читаемые аналоги
        display = ::replace(display, "\r", "\\r");
        display = ::replace(display, "\n", "\\n");
        display = ::replace(display, "\t", "\\t");
        printf("%s'%s'", i > 0 ? ", " : "", display.c_str());
    }

    // Дополнительно показываем пользовательские стоп-слова отдельно
    if (!params.stop_words.empty()) {
        printf(" [+ from --stop-words: %s]", params.stop_words.c_str());
    }

    printf("\n");

	std::thread input_thread(input_thread_func);
	std::thread shortcut_thread([]() {
        keyboard_shortcut_func();
    });

    // ===== ИНФОРМАЦИЯ О ГОРЯЧИХ КЛАВИШАХ =====
    printf("\nVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right), Delete(Ctrl+Delete), Reset(Ctrl+R)\n");

    if (params.push_to_talk)
        printf("Type anything or hold 'Alt' to speak:\n");
    else
        printf("Start speaking or typing:\n");

    // ===== НАЧАЛЬНОЕ ПРИГЛАШЕНИЕ =====
    // Формат старта должен совпадать с форматом последующих ходов
    printf("\n");
    #ifdef _WIN32
        set_console_color(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    #else
        printf("\033[32m");
    #endif
        printf("%s%s ", params.person.c_str(), chat_symb.c_str());    // ← "Друг: "
        reset_console_color();
        fflush(stdout);
    // -------------------------------------------------------

	int vad_result_prev = 2; // ended
	float speech_start_ms = 0;
	float speech_end_ms = 0;
	float speech_len = 0;
	int len_in_samples = 0;
	int64_t speech_start_sample = 0; // сохраняем номер сэмпла, когда началась речь
	float llama_interrupted_time = 0.0;
	llama_start_time = 0.0;
	float llama_start_generation_time = 0.0; // после оперативной обработки
	llama_end_time = 0.0;
	llama_time_total = 0.0;
    std::string user_typed = "";
    bool user_typed_this = false;


// ### ОСНОВНОЙ ЦИКЛ РАБОТЫ ПРИЛОЖЕНИЯ ###
    while (is_running) {
        // ===== ПРОВЕРКА СОБЫТИЙ В НАЧАЛЕ КАЖДОЙ ИТЕРАЦИИ =====
        // Проверяем SDL события (закрытие окна, Ctrl+C и т.д.)
        // Это единственное место, где вызывается sdl_poll_events()
        is_running = sdl_poll_events();
        if (!is_running) {
            printf("\n[Shutdown requested, cleaning up...]\n");
            break;
        }
        // ===== КОНЕЦ ПРОВЕРКИ СОБЫТИЙ =====

        // СБРОС СОСТОЯНИЯ ПРЕРЫВАНИЯ
        g_is_interrupted.store(false);
        llama_interrupted.store(0);

        // задержка. попробуйте опустить?
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int64_t t_ms = 0;
        // === FIX: Объявляем text_heard раньше, чтобы он был виден в keyboard input ===
        static std::string text_heard = "";  // static — сохраняет значение между итерациями

// ===  Буфер накопления ввода пользователя ===
// keyboard input
user_typed_this = false;
console::set_display(console::reset);

        // === ФИКС: Накопление ввода вместо мгновенной отправки ===
        user_typed_this = false;  // ← СБРОС ФЛАГА В НАЧАЛЕ КАЖДОЙ ИТЕРАЦИИ
{
    std::lock_guard<std::mutex> lock(input_mutex);
    if (!input_queue.empty())
    {
        std::string buffer;
        while (!input_queue.empty())
        {
            buffer += input_queue.front() + " ";
            input_queue.pop();
        }
                user_typed = buffer;
                trim(user_typed);
                user_typed_this = true;
            }
        }

        // hotkeys - атомарное чтение и сброс
        std::string hk_copy;
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            hk_copy = g_hotkey_pressed;
            g_hotkey_pressed = "";   // Сбрасываем ВСЕГДА после копирования, чтобы избежать залипания
        }

        if (!hk_copy.empty())
        {
            if (hk_copy == "Ctrl+Space") {
                user_typed = "Stop";
            } else if (hk_copy == "Ctrl+Right") {
                user_typed = "Regenerate";
            } else if (hk_copy == "Ctrl+Delete") {
                user_typed = "Delete";
            } else if (hk_copy == "Ctrl+R") {
                user_typed = "Reset";
            }

            if (hk_copy != "Alt")
            {
                user_typed_this = true;
            }
        }
        {
            // Получаем аудио из буфера длительностью step_ms (2000 мс), async — асинхронно
            audio.get(2000, pcmf32_cur);

            // Защита от пустого аудиобуфера
            if (pcmf32_cur.empty()) {
                // Нет данных для анализа, пропускаем VAD
                continue;
            }

            // WHISPER_SAMPLE_RATE — частота дискретизации аудио для Whisper (16 кГц)
            // vad_last_ms — минимальная длина речевого сегмента для VAD (по умолчанию 1250 мс)
            // Вызываем VAD (Voice Activity Detection) для определения наличия речи в аудиосигнале
            // vad_simple_int возвращает:
            // 0 — тишина, 1 — начало речи, 2 — конец речи
            int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms,
                                            params.vad_thold, params.freq_thold, params.print_energy,
                                            params.vad_start_thold);

            // =================================================================
            // ОБНАРУЖЕНИЕ НАЧАЛА РЕЧИ (VOICE ACTIVITY DETECTION)
            // =================================================================
            // vad_result == 1 означает, что VAD обнаружил начало речевого сигнала
            // params.vad_start_thold > 0.0f проверяет, что порог начала речи включён
            // (если порог == 0, функция VAD начала речи отключена)
            // =================================================================
            if (vad_result == 1 && params.vad_start_thold > 0.0f) // speech started
            {
                if (vad_result_prev != 1) // реальное начало речи
                {
                    // Запоминаем время начала речи
                    speech_start_ms = get_current_time_ms(); // float

                    // Обновляем статус VAD
                    vad_result_prev = 1;

                    // НИКАКОЙ ТРАНСКРИПЦИИ ЗДЕСЬ НЕ НУЖНО — только запоминаем время начала.
                    // Раньше здесь был вызов transcribe() для "прогревки", но он:
                    // 1. Создавал лишнюю задержку
                    // 2. Мог сбивать внутреннее состояние Whisper
                    // 3. Результат нигде не использовался (all_heard_pre не читается)
                }

                // Пользователь начал говорить — запрещаем воспроизведение через XTTS
                std::string current_hotkey;
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    current_hotkey = g_hotkey_pressed;
                }
                if (!params.push_to_talk || (params.push_to_talk && current_hotkey == "Alt"))
                {
                    allow_xtts_file(params.xtts_control_path, 0);

                    // Устанавливаем флаги прерывания
                    llama_interrupted.store(1);
                    g_is_interrupted.store(true);
                }
            }

            // Если VAD обнаружил конец речи (vad_result >= 2) и предыдущее состояние было началом речи, или была нажата горячая клавиша, или пользователь ввёл текст вручную
            if (vad_result >= 2 && vad_result_prev == 1 || force_speak || user_typed.size())  // speech ended or user typed
            {
                // Запоминаем время окончания речи
                speech_end_ms = get_current_time_ms(); // float в секундах.мс
                // Вычисляем длительность речи
                speech_len = speech_end_ms - speech_start_ms;
                // Фильтруем слишком короткие или слишком длинные речевые сегменты
                if (speech_len < 0.10) speech_len = 0;
                else if (speech_len > 10.0) speech_len = 0;

                vad_result_prev = 2;

                // Пропускаем обработку, если длина речи нулевая и нет введённого пользователем текста
                if (!speech_len && !user_typed.size()) {
                    speech_start_ms = 0;
                    audio.clear();  // ← ДОБАВИТЬ
                    continue;
                }

                // ============================================================
                // ПРОСТОЕ И НАДЁЖНОЕ ИЗВЛЕЧЕНИЕ АУДИО
                // ============================================================
                // Вместо сложной и некорректной обрезки по времени (которая не работает
                // из-за отсутствия абсолютных меток в audio_async), просто берём
                // последние params.voice_ms миллисекунд (по умолчанию 10000 мс = 10 сек).
                // Whisper сам отфильтрует тишину в начале и конце.
                // ============================================================
                audio.get(params.voice_ms, pcmf32_cur);

                // Сбрасываем время начала речи для следующего раза
                speech_start_ms = 0;


                std::string all_heard;
                // Если пользователь ввёл текст вручную — используем его
                if (user_typed.size())
                    {
                        all_heard = user_typed;
                        user_typed = "";
                    }
                else if (!force_speak)
                    {
                        // Если нет принудительного распознавания — транскрибируем аудио
                        if (!params.push_to_talk || (params.push_to_talk && hk_copy == "Alt"))
                        {
                            // === ФИЛЬТР ГАЛЛЮЦИНАЦИЙ ПО УВЕРЕННОСТИ ===
                            all_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms));

                            // === УСИЛЕННЫЙ ФИЛЬТР ГАЛЛЮЦИНАЦИЙ ===
                            bool discard = false;

                            // 1. Низкая уверенность
                            if (prob0 < 0.45f) {
                                discard = true;
                            }

                            // 2. Явные галлюцинации (повторы)
                            if (!discard && !all_heard.empty()) {
                                std::string lower = all_heard;
                                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

                                // Паттерны галлюцинаций
                                if (lower.find("субтитр") != std::string::npos ||
                                    lower.find("thanks for watching") != std::string::npos ||
                                    lower.find("пожалуйста подпишитесь") != std::string::npos ||
                                    lower == "you" || lower == "you." ||
                                    lower == "i" || lower == "i.") {
                                    discard = true;
                                }

                                // Повтор одного слова > 3 раз
                                std::istringstream iss(lower);
                                std::vector<std::string> words;
                                std::string w;
                                while (iss >> w) words.push_back(w);

                                if (words.size() >= 4) {
                                    for (size_t i = 0; i < words.size() - 3; i++) {
                                        if (words[i] == words[i+1] &&
                                            words[i] == words[i+2] &&
                                            words[i] == words[i+3]) {
                                            discard = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            if (discard) {
                                if (params.verbose) {
                                    fprintf(stdout, "[Whisper] ОТБРОШЕНО (prob=%.3f): '%s'\n", prob0, all_heard.c_str());
                                }
                                all_heard = "";
                                audio.clear();
                                continue;
                            }
                            { // Сброс под защитой мьютекса
                                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                                g_hotkey_pressed = "";
                            }
                        }
                    }

                    // Проверка wake-command (если включено)
                    if (!params.wake_cmd.empty()) {
                        // Проверяем, начинается ли распознанный текст с команды пробуждения
                        // Например: "Эмма, привет" -> начинается с "Эмма,"
                        if (all_heard.find(params.wake_cmd) != 0) {
                            // Фраза не начинается с имени — игнорируем
                            if (params.verbose) {
                                fprintf(stdout, "[wake] ignored: \"%s\"\n", all_heard.c_str());
                            }
                            audio.clear();
                            continue;
                        }

                        // Убираем имя из текста, оставляем только суть запроса
                        // "Эмма, привет" -> "привет"
                        text_heard = all_heard.substr(params.wake_cmd.length());
                        trim(text_heard);

                        if (params.verbose) {
                            fprintf(stdout, "[wake] accepted: \"%s\"\n", text_heard.c_str());
                        }
                    } else {
                        // Режим без wake-word — используем весь распознанный текст
                        text_heard = all_heard;
                    }

                // Выводим уровень энергии, если включён (для отладки)
                if (params.print_energy) fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());

                // ============================================================
                // УДАЛЕНО: Двойная проверка wake_cmd.
                // Первая проверка уже выполнена выше (строка ~5850),
                // а text_heard уже содержит очищенную от имени строку.
                // Повторная проверка на all_heard ломает логику для обычных фраз.
                // ============================================================

                if (!params.heard_ok.empty()) {
                    std::string voice_copy = current_voice;
                    std::string lang_copy = params.language;
                    std::string url_copy = params.xtts_url;
                    std::string heard_ok_copy = params.heard_ok;
                    safe_thread_emplace(threads, [heard_ok_copy, voice_copy, lang_copy, url_copy]() {
                        send_tts_async(heard_ok_copy, voice_copy, lang_copy, url_copy);
                    });
                }

                // Удалить текст в квадратных скобках: [всё внутри], но не жадно
                // Используем [^\\]]* вместо .*?, потому что std::regex нестабильно работает с ленивыми квантификаторами
                // Пример: "Привет [звук] и [музыка]" → "Привет  и "
                try {
                    std::regex re(R"(\[[^\[\]]*\])");  // Надёжная замена для "\\[.*?\\]"
                    text_heard = std::regex_replace(text_heard, re, "");
                } catch (const std::regex_error& e) {
                    // Логируем, но не падаем — если регулярка сломалась
                    fprintf(stderr, "Regex error while removing [brackets]: %s\n", e.what());
                    // Оставляем text_heard как есть
                }

                // удалить все символы, за исключением букв, цифр, знаков препинания и ':', '\'', '-', ' '
                if (params.language == "en" && !user_typed_this) text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), ""); // breaks non latin text, e.g. Russian
                // take first line
                text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

                // Удаляем пробелы в начале и в конце строки
                text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

                // Удаляем нежелательные знаки в конце строки
                text_heard = RemoveTrailingCharactersUtf8(text_heard, ",");
                text_heard = RemoveTrailingCharactersUtf8(text_heard, ".");
                text_heard = RemoveTrailingCharactersUtf8(text_heard, "»");
                text_heard = RemoveTrailingCharactersUtf8(text_heard, "[");
                text_heard = RemoveTrailingCharactersUtf8(text_heard, "]");
                text_heard = RemoveTrailingCharactersUtf8(text_heard, "\"");

                // Удаляем нежелательные символы в начале строки
                if (!text_heard.empty() && text_heard[0] == '.') text_heard.erase(0, 1);
                if (!text_heard.empty() && text_heard[0] == '[') text_heard.erase(0, 1);
                trim(text_heard);

                // === ДОБАВЛЕНО: Очистка буфера ПОСЛЕ успешного распознавания ===
                audio.clear();

                // ============================================================
                // ФИЛЬТРАЦИЯ ГАЛЛЮЦИНАЦИЙ WHISPER
                // ============================================================
                // Удаляем только явный мусор, который Whisper иногда генерирует:
                // - служебные фразы из видео/аудио
                // - одни знаки препинания без слов
                //
                // ВАЖНО: НЕ удаляем вежливые фразы ("Спасибо", "Пока"),
                // короткие ответы ("Да", "Нет"), обращения к боту.
                // ============================================================
                bool is_garbage = false;

                // 1. Текст состоит только из знаков препинания или пустой
                if (text_heard.empty() ||
                    text_heard == "!" || text_heard == "." || text_heard == "?" ||
                    text_heard == "..." || text_heard == "!!" || text_heard == "??") {
                    is_garbage = true;
                }

                // 2. Служебные фразы из видео/аудио (Whisper часто их галлюцинирует)
                if (!is_garbage && (
                    text_heard.find("Редактор субтитров") != std::string::npos ||
                    text_heard.find("Субтитры") != std::string::npos ||
                    text_heard.find("Спасибо за внимание") != std::string::npos ||
                    text_heard.find("Продолжение следует") != std::string::npos ||
                    text_heard.find("End of") != std::string::npos ||
                    text_heard.find("The End") != std::string::npos ||
                    text_heard.find("Translated by") != std::string::npos ||
                    text_heard.find("Thanks for watching") != std::string::npos ||
                    text_heard.find("Thank you for watching") != std::string::npos ||
                    text_heard.find("*click*") != std::string::npos ||
                    text_heard.find("Silence") != std::string::npos ||
                    text_heard.find("ПЕСНЯ") != std::string::npos
                )) {
                    is_garbage = true;
                }

                // 3. Короткие служебные слова (только если это ВСЯ фраза)
                //    "Sil", "Bye", "Okay." — часто галлюцинации
                if (!is_garbage && (
                    text_heard == "Sil" || text_heard == "Bye" ||
                    text_heard == "Okay." || text_heard == "Thanks." || text_heard == "Bye."
                )) {
                    is_garbage = true;
                }

                // 4. Если текст — это только имя бота (без вопроса/обращения)
                if (!is_garbage && text_heard == params.bot_name) {
                    is_garbage = true;
                }

                // 5. Если это явный звуковой эффект из транскрипции
                if (!is_garbage && text_heard == "*Звук!*") {
                    is_garbage = true;
                }

                // Очищаем, если это мусор
                if (is_garbage) {
                    text_heard = "";
                    if (params.verbose) {
                        fprintf(stdout, "\n[Фильтр: удалён мусор]\n");
                    }
                }

                // Иначе — оставляем текст как есть, даже если он короткий
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), ""); // trailing whitespace
                text_heard_trimmed = text_heard; // no periods or spaces
                trim(text_heard_trimmed);

                // Безопасное удаление начальных знаков препинания
                if (!text_heard_trimmed.empty()) {
                    if (text_heard_trimmed[0] == '.') text_heard_trimmed.erase(0, 1);
                    if (!text_heard_trimmed.empty() && text_heard_trimmed[0] == '!') text_heard_trimmed.erase(0, 1);
                }

                // Безопасное удаление конечных знаков препинания
                if (!text_heard_trimmed.empty()) {
                    size_t last_pos = text_heard_trimmed.length() - 1;
                    if (text_heard_trimmed[last_pos] == '.' || text_heard_trimmed[last_pos] == '!') {
                        text_heard_trimmed.erase(last_pos, 1);
                    }
                }

                // ============================================================
                // ФИНАЛЬНАЯ ОЧИСТКА text_heard (один проход)
                // ============================================================
                // Удаляем пробелы в начале и конце
                trim(text_heard);

                // Удаляем начальные и конечные знаки препинания
                if (!text_heard.empty()) {
                    // Начальные
                    if (text_heard[0] == '.' || text_heard[0] == '!') {
                        text_heard.erase(0, 1);
                        trim(text_heard);
                    }
                    // Конечные
                    if (!text_heard.empty()) {
                        size_t last_pos = text_heard.length() - 1;
                        if (text_heard[last_pos] == '.' || text_heard[last_pos] == '!') {
                            text_heard.erase(last_pos, 1);
                            trim(text_heard);
                        }
                    }
                }

                text_heard_trimmed = LowerCase(text_heard);
                trim(text_heard_trimmed);

                fflush(stdout);

                std::string user_command; // здесь будет храниться распознанная команда пользователя

                // Если VAD начала речи включён (порог > 0), разрешаем воспроизведение XTTS
                if (params.vad_start_thold > 0.0f)
                {
                    allow_xtts_file(params.xtts_control_path, 1);
                }

                // ВВОДНОЕ предложение TTS rand для мгновенного ответа
                if (params.xtts_intro && text_heard_trimmed.size())
                {
                    dist = std::uniform_int_distribution<size_t>(0, tts_intros.size() - 1);
                    rand_intro_text = tts_intros[dist(gen)];

                    if (!rand_intro_text.empty()) {
                        for (auto it = threads.begin(); it != threads.end(); ) {
                            if (it->joinable()) {
                                it->detach();
                                it = threads.erase(it);
                            } else {
                                ++it;
                            }
                        }

                        std::string voice_copy = current_voice;
                        std::string lang_copy = params.language;
                        std::string url_copy = params.xtts_url;
                        safe_thread_emplace(threads, [rand_intro_text, voice_copy, lang_copy, url_copy]() {
                            send_tts_async(rand_intro_text, voice_copy, lang_copy, url_copy);
                        });
                    }
                }

                // Определяем, какая команда была произнесена пользователем
                if (text_heard_trimmed.find("regenerate") != std::string::npos ||
                    text_heard_trimmed.find("Переделай") != std::string::npos  ||
                    text_heard_trimmed.find("Переделаем") != std::string::npos ||
                    text_heard_trimmed.find("егенерируй") != std::string::npos ||
                    text_heard_trimmed.find("егенерировать") != std::string::npos)
                {
                    user_command = "regenerate";
                }
                else if (text_heard_trimmed.find("google") != std::string::npos ||
                        text_heard_trimmed.find("Погугли") != std::string::npos ||
                        text_heard_trimmed.find("По гугл") != std::string::npos)
                {
                    user_command = "google";
                }
                else if (text_heard_trimmed.find("reset") != std::string::npos ||
                        text_heard_trimmed.find("delete everything") != std::string::npos ||
                        text_heard_trimmed.find("Сброс") != std::string::npos ||
                        text_heard_trimmed.find("Сбросить") != std::string::npos ||
                        text_heard_trimmed.find("Удали все") != std::string::npos ||
                        text_heard_trimmed.find("Удалить все") != std::string::npos)
                {
                    user_command = "reset";
                }
                else if (text_heard_trimmed.find("delete") != std::string::npos ||
                        text_heard_trimmed.find("please do it") != std::string::npos ||
                        text_heard_trimmed.find("Удалить сообщение") != std::string::npos ||
                        text_heard_trimmed.find("Удали сообщение") != std::string::npos ||
                        text_heard_trimmed.find("Удали два сообщения") != std::string::npos ||
                        text_heard_trimmed.find("Удали три сообщения") != std::string::npos)
                {
                    user_command = "delete";
                }
                else if (text_heard_trimmed == "step" ||
                        text_heard_trimmed.find("stop") != std::string::npos ||
                        text_heard_trimmed.find("Стоп") != std::string::npos ||
                        text_heard_trimmed.find("тановись") != std::string::npos ||
                        text_heard_trimmed.find("Хватит") != std::string::npos
                        )
                {
                    user_command = "stop";
                }
                else if (text_heard_trimmed.find("call") == 0 ||
                        text_heard_trimmed.find("can you call") != std::string::npos ||
                        text_heard_trimmed.find("let's call") != std::string::npos ||
                        text_heard_trimmed.find("please call") != std::string::npos ||
                        text_heard_trimmed.find("can you hear me") != std::string::npos ||
                        text_heard_trimmed.find("do you hear me") != std::string::npos ||
                        text_heard_trimmed.find("are you here") != std::string::npos ||
                        (text_heard_trimmed.find("what do you think") != std::string::npos &&
                        text_heard_trimmed.find("what do you think of") == std::string::npos) ||
                        text_heard_trimmed.find("позови") != std::string::npos ||
                        text_heard_trimmed.find("ты тут") != std::string::npos ||
                        text_heard_trimmed.find("Ты тут") != std::string::npos ||
                        text_heard_trimmed.find("ты меня слышишь") != std::string::npos ||
                        text_heard_trimmed.find("Ты меня слышишь") != std::string::npos ||
                        text_heard_trimmed.find("ты слышишь меня") != std::string::npos ||
                        text_heard_trimmed.find("Ты слышишь меня") != std::string::npos ||
                        text_heard_trimmed.find("Ты здесь") != std::string::npos ||
                        text_heard_trimmed.find("ты здесь") != std::string::npos ||
                        (text_heard_trimmed.find("то ты думаешь") != std::string::npos &&
                        text_heard != "Что ты думаешь?" &&
                        text_heard_trimmed.find("то ты об этом думаешь") == std::string::npos) ||
                        (text_heard_trimmed.find("то ты об этом думаешь") != std::string::npos &&
                        text_heard != "Что ты об этом думаешь?"))

{
    user_command = "call";
}

// Проверяем, можно ли выполнять новую команду (с задержкой, чтобы избежать дублирования)
	if (user_command.size() && !new_command_allowed && std::time(0)-last_command_time >= 2)
	{
    new_command_allowed = 1; // даём разрешение на выполнение новой команды
}

// Если команда — "regenerate" — перегенерируем последний ответ модели
if (user_command == "regenerate" ||
    text_heard_trimmed == "Please regenerate" ||
    text_heard_trimmed == "Regenerate please" ||
    text_heard_trimmed == "Regenerate, please" ||
    text_heard_trimmed == "Try again please" ||
    text_heard_trimmed == "Try again, please" ||
    text_heard_trimmed == "Please try again" ||
    text_heard_trimmed == "Try again")
				{
					if (new_command_allowed)
                        {
                            new_command_allowed = 0;
                            last_command_time = std::time(0);

                            if (!past_prev_arr.empty())
                            {
                                // Возвращаем контекст к предыдущему состоянию
                                n_past_prev = past_prev_arr.back();
                                past_prev_arr.pop_back();

                                int rollback_num = (int)(embd_inp.size() - n_past_prev);

                                if (rollback_num > 0 && rollback_num <= (int)embd_inp.size())
                                {
                                    // Удаляем последние токены из контекста
                                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                                    printf(" [regenerating %d tokens. Context: %zu]\n", rollback_num, embd_inp.size());

                                    n_past = embd_inp.size();
                                    n_session_consumed = n_past;

                                    // Удаляем последовательность из KV-кэша
                                    llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);

                                    // Восстанавливаем предыдущий запрос
                                    text_heard = text_heard_prev;
                                    text_heard_trimmed = "";

                                    // НОВЫЙ КОД: берём последний текст из g_last_tts_text (без массива)
                                    std::string text_to_respeak_safe;
                                    {
                                        std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                                        text_to_respeak_safe = g_last_tts_text;
                                    }

                                    // Отправляем в TTS, если есть что озвучивать (безопасно, с мьютексом)
                                    if (!text_to_respeak_safe.empty()) {
                                        std::string voice_copy = current_voice;  // <-- КОПИЯ
                                        safe_thread_emplace(threads, [text_to_respeak_safe, voice_copy, params]() {
                                            send_tts_async(text_to_respeak_safe, voice_copy, params.language, params.xtts_url);
                                        });
                                    }


                                }
                            }
                        }
				}

            // УДАЛЕНИЕ СООБЩЕНИЙ
            else if (user_command == "delete" ||
            text_heard_trimmed == "Please delete" ||
            text_heard_trimmed == "Please delete the last message" ||
            text_heard_trimmed == "Delete please" ||
            text_heard_trimmed == "Delete, please")
                    {

            // Проверяем, можно ли выполнять команду (с учётом таймаута)
            if (new_command_allowed)
					{

if (!past_prev_arr.empty())
{
            // Удаление двух сообщений
            if (text_heard_trimmed == "delete two messages" ||
                text_heard_trimmed == "Удали 2 сообщения" ||
                text_heard_trimmed == "Удали два сообщения" ||
                text_heard_trimmed == "Please donate to the messages")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}
            // Удаление трёх сообщений
            else if (text_heard_trimmed == "delete three messages" ||
                     text_heard_trimmed == "Удали 3 сообщения" ||
                     text_heard_trimmed == "Удали три сообщения")
							{
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
								n_past_prev = past_prev_arr.back();
								past_prev_arr.pop_back();
							}

                            // Удаление одного сообщения
							n_past_prev = past_prev_arr.back();
							past_prev_arr.pop_back();

							int rollback_num = embd_inp.size()-n_past_prev;

							if (rollback_num)
							{
                            // Удаляем токены из контекста
								embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
								printf(" deleting %I32d tokens. Tokens in ctx: %zu\n", rollback_num, embd_inp.size());
								n_past = embd_inp.size();
								n_session_consumed = n_past;
                                // Удаляем последовательность 0 из KV-кэша (новый API)
                                // Диапазон [embd_inp.size(), end)
                                llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);

                            // Сбрасываем переменные
								text_heard = "";
								text_heard_trimmed = "";
								last_command_time = std::time(0);
								new_command_allowed = 0;

                            // Асинхронное воспроизведение "Deleted" через TTS (безопасно, с мьютексом)
                            std::string text_for_deleted_tts = "Deleted";

                            if (!text_for_deleted_tts.empty()) {
                                std::string voice_copy = current_voice;  // <-- КОПИЯ
                                safe_thread_emplace(threads, [text_for_deleted_tts, voice_copy, params]() {
                                    send_tts_async(text_for_deleted_tts, voice_copy, params.language, params.xtts_url);
                                });
                            }

                            // При удалении не нужно переозвучивать предыдущий текст,
                            // поэтому убираем всю логику с text_to_respeak_safe и дублирующий вызов
							}
						}
						else
                            {
                            // Если удалять нечего — сообщаем об этом
                                printf("Nothing to delete more\n");
                                send_tts_async("Nothing to delete more", "ux", params.language);
                            }
					}
    audio.clear(); // Очищаем аудио-буфер
	}

// СБРОС КОНТЕКСТА
else if (user_command == "reset")
{
    if (new_command_allowed)
    {
        if (!past_prev_arr.empty())
        {
            n_past_prev = past_prev_arr.front();
            past_prev_arr.clear();
            int rollback_num = embd_inp.size() - n_past_prev;
            if (rollback_num)
            {
                printf(" [Resetting context of %zd tokens.]\n", embd_inp.size());

                {
                    std::lock_guard<std::mutex> lock(g_llama_mutex);

                    llama_batch_free(batch);

                    if (ctx_llama) {
                        llama_free(ctx_llama);
                        ctx_llama = nullptr;
                    }

                    ctx_llama = llama_init_from_model(model_llama, lcparams);

                    if (!ctx_llama) {
                        fprintf(stderr, "%s : ERROR: Failed to reinitialize llama context on reset\n", __func__);
                        return 1;
                    }

                    batch = llama_batch_init(2048, 0, 1);
                    embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

                    if (embd_inp.empty()) {
                        fprintf(stderr, "%s : ERROR: Failed to tokenize prompt after reset\n", __func__);
                        return 1;
                    }

                    // ===== ИСПРАВЛЕНИЕ: ПРОВЕРКА РАЗМЕРА =====
                    if (embd_inp.size() > 2048) {
                        fprintf(stderr, "%s : FATAL ERROR: Prompt size (%zu tokens) exceeds batch limit (2048)\n",
                                __func__, embd_inp.size());
                        fprintf(stderr, "Please reduce prompt size or increase batch limit in code.\n");
                        return 1;
                    }
                    // =======================================

                    batch.n_tokens = embd_inp.size();

                    for (int i = 0; i < batch.n_tokens; i++) {
                        batch.token[i] = embd_inp[i];
                        batch.pos[i] = i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = i == batch.n_tokens - 1;
                    }

                    if (llama_decode(ctx_llama, batch)) {
                        fprintf(stderr, "%s : failed to decode after reset\n", __func__);
                        return 1;
                    }
                }

                n_past = embd_inp.size();
                n_session_consumed = embd_inp.size();
                printf(" [Context is now %zu/%d tokens. n_past: %d]\n", embd_inp.size(), params.ctx_size, n_past);

                text_heard = "";
                text_heard_trimmed = "";

                send_tts_async("Reset whole context", params.xtts_voice, params.language, params.xtts_url);

                new_command_allowed = 0;
                last_command_time = std::time(0);
            }
        }
        else
        {
            printf(" [Nothing to reset more]\n");
            send_tts_async("Nothing to reset more", params.xtts_voice, params.language, params.xtts_url);
        }
    }
    audio.clear();
    continue;
}

// ОСТАНОВКА stop
if (user_command == "stop")
    {
        std::string lower_text = LowerCase(text_heard_trimmed);
        // Расширенный список стоп-команд с учетом разных вариантов
        static const std::vector<std::string> stop_commands = {
            "стоп", "stop", "остановись", "останови", "хватит", "прекрати",
            "стоп пожалуйста", "stop please", "хватит пожалуйста", "прекрати пожалуйста"
        };
        // Фразы с именем бота (например, "Эмма, стоп")
        static const std::vector<std::string> stop_with_bot = {
            params.bot_name + " стоп",
            params.bot_name + " stop",
            params.bot_name + " остановись",
            params.bot_name + " хватит"
        };

        bool is_stop_command = false;
        // 1. Проверяем точное совпадение с базовыми командами
        for (const auto& cmd : stop_commands) {
            if (lower_text == cmd) {
                is_stop_command = true;
                break;
            }
        }
        // 2. Проверяем команды с именем бота
        if (!is_stop_command) {
            for (const auto& cmd : stop_with_bot) {
                if (lower_text.find(cmd) != std::string::npos) {
                    is_stop_command = true;
                    break;
                }
            }
        }
        // 3. Проверяем короткие фразы, которые начинаются со стоп-слова
        if (!is_stop_command && lower_text.length() < 20) {
            for (const auto& cmd : stop_commands) {
                if (cmd.length() < lower_text.length() &&
                    lower_text.find(cmd) == 0) {
                    // Начинается со стоп-слова и короткая
                    is_stop_command = true;
                    break;
                }
            }
        }
        if (!is_stop_command) {
            // Не стоп-команда — продолжаем обычную обработку
            user_command.clear();
            continue;
        }
        // Реальный STOP-запрос
        fprintf(stdout, "[user] requested STOP: \"%s\"\n", text_heard_trimmed.c_str());
        // 1) Безопасно очищаем буферы и ввод
        text_heard.clear();
        text_heard_trimmed.clear();
        audio.clear();
        user_typed.clear();
        user_typed_this = false;
        // 2) Прерываем озвучку XTTS
        allow_xtts_file(params.xtts_control_path, 0);
        // 3) Устанавливаем флаг остановки генерации
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            g_hotkey_pressed = "Ctrl+Space";
        }
        // Продолжаем цикл — модель корректно завершит текущую генерацию
        continue;
    }
    // Скажи сколько время
    else if (text_heard_trimmed.find("время") != std::string::npos ||
            text_heard_trimmed.find("который час") != std::string::npos ||
            text_heard_trimmed.find("what time") != std::string::npos ||
            text_heard_trimmed.find("сколько времени") != std::string::npos ||
            text_heard_trimmed.find("сколько время") != std::string::npos)
    {
        user_command = "time";
    }

        // Обработчик команды "google"
    else if (user_command == "google")
    {
        // Простая лямбда для отправки текста в TTS (без массивов и мьютексов)
        auto speak_direct = [&](const std::string& msg) {
            if (msg.empty()) return;
            std::string msg_copy = msg;
            std::string voice_copy = current_voice;  // <-- ДОБАВИТЬ КОПИЮ
            try {
                safe_thread_emplace(threads, [msg_copy, voice_copy, params]() {
                    send_tts_async(msg_copy, voice_copy, params.language, params.xtts_url);
                });
            } catch (const std::exception& e) {
                fprintf(stderr, "[google] TTS thread spawn failed: %s\n", e.what());
            }
        };

        // Достаём ключевые слова
        std::string q = ParseCommandAndGetKeyword(text_heard_trimmed, user_command);

        if (q.empty()) {
            fprintf(stdout, "[google] can't get keyword from: %s\n", text_heard_trimmed.c_str());
            speak_direct("Извините, не удалось понять, что именно вы хотите найти.");
            user_typed.clear();
            user_typed_this = false;
        } else {
            // Аудио-квитанция — отправляем в TTS безопасно, с мьютексом
            std::string google_search_msg = "Ищу информацию по запросу: " + q;
            std::string voice_copy = current_voice;  // <-- КОПИЯ
            std::string lang_copy  = params.language;   // <-- КОПИЯ ДЛЯ ПОТОКОБЕЗОПАСНОСТИ
            std::string url_copy   = params.xtts_url;   // <-- КОПИЯ ДЛЯ ПОТОКОБЕЗОПАСНОСТИ
            safe_thread_emplace(threads, [google_search_msg, voice_copy, lang_copy, url_copy]() {
                send_tts_async(google_search_msg, voice_copy, lang_copy, url_copy);
            });

            // (запрос к серверу, проверка resp.empty(), формирование промпта для LLaMA)

// Запрос к поисковому серверу
const std::string url = params.google_url + "google?q=" + UrlEncode(q);
std::string resp = send_curl(url);
    if (resp.empty()) {
        fprintf(stdout, "[google] empty response for (%s) — check backend\n", q.c_str());

        // ГОЛОСОВОЕ СООБЩЕНИЕ ОБ ОШИБКЕ (безопасно, с мьютексом)
        std::string error_msg = "Извините, не удалось найти информацию по запросу: " + q;
        std::string voice_copy = current_voice;  // <-- КОПИЯ
        safe_thread_emplace(threads, [error_msg, voice_copy, params]() {
            send_tts_async(error_msg, voice_copy, params.language, params.xtts_url);
        });

        // Не прерываем цикл — просто не будем формировать спец-промпт
    } else {
    fprintf(stdout, "[google] resp (%s): %s\n", q.c_str(), resp.c_str());
    // Подрезаем ответ «по границе предложения»
    auto truncate_smart = [](std::string s, size_t hard = 600, size_t prefer = 420) {
        if (s.size() <= hard) return s;
        size_t cut = s.find_last_of(".!?");
        if (cut != std::string::npos && cut >= std::min(prefer, hard)) {
            s.erase(cut + 1);
        } else {
            s.erase(std::min(hard, s.size()));
            s += "...";
        }
        return s;
    };
    resp = truncate_smart(resp);

    // Формируем реплику пользователя для LLaMA
    std::string llm_prompt =
        params.person + ": " + params.bot_name +
        ", пожалуйста, кратко изложи основное из текста, найденного по запросу \"" + q + "\": " + resp;
    text_heard = llm_prompt;
    user_typed_this = true;
    }
}
    // Чистим одноразовые буферы, НО НЕ ДЕЛАЕМ continue;
    audio.clear();
    user_typed.clear();
    // user_typed_this оставляем как есть: true — если мы подменили text_heard промптом; false — если нет
    // Конец обработчика "google" ПОГУГЛИ
}
    // Скажи время
    else if (user_command == "time") {
        // Цель: ответить только текущее время в цифровом формате, без даты
        // 1. Получаем текущее системное время
        std::time_t t_now = std::time(nullptr);
        std::tm tm_local_now {};
        #ifdef _WIN32
            localtime_s(&tm_local_now, &t_now); // Windows версия
        #else
            localtime_r(&t_now, &tm_local_now); // POSIX версия
        #endif
        int hour = tm_local_now.tm_hour;  // 0-23
        int minute = tm_local_now.tm_min; // 0-59
        // Форматируем время как HH:MM
        char time_buffer[64];
        std::snprintf(time_buffer, sizeof(time_buffer), "Сейчас %02d:%02d", hour, minute);

        // Формируем промпт для LLM
        std::string llm_prompt = params.person + ": Который час?\n" + params.bot_name + ": " + std::string(time_buffer);
        // Подменяем вход
        text_heard = llm_prompt;
        user_typed_this = true;
        // Очищаем аудио и состояние
        audio.clear();
        user_typed.clear();
        text_heard_trimmed = "";
        // reply_part не сбрасываем
        // Продолжаем основной цикл обработки
    }

    // CALL
    // В функции обработки команды "call" обновляем логику выбора бота
    else if (user_command == "call") {
        // Проверяем, включена ли функция множества ботов
        if (params.multi_chars) {
            std::string q = ParseCommandAndGetKeyword(text_heard, user_command);
            if (!q.empty()) {
                fprintf(stdout, "Переключаюсь на бота: %s", q.c_str());
                std::string old_bot_name = params.bot_name;  // сохраняем для отладки (опционально)
                params.bot_name = q;                         // меняем имя бота

                // ОБНОВЛЯЕМ АНТИПРОМПТЫ С НОВЫМ ИМЕНЕМ ПОЛЬЗОВАТЕЛЯ
                update_antiprompts(params.person, params.bot_name);

                if (params.verbose) {
                    fprintf(stdout, " [antiprompts updated for bot: %s]\n", params.bot_name.c_str());
                }
            } else {
                fprintf(stdout, "Error: can't find bot name in text_heard_trimmed: %s", text_heard_trimmed.c_str());
            }
        } else {
            // Если multi_chars отключен, игнорируем команду call
            // fprintf(stdout, "Команда 'call' игнорируется: режим multi_chars отключен.");
        }
    }

        int translation_is_going = 0;
        int n_embd_inp_before_trans = 0;
        int tokens_in_reply = 0;
        std::string current_voice_tmp = "";
        reply_part = 0;

// ### ЦИКЛ ГЕНЕРАЦИИ ТЕКСТА (LLaMA) ###
    // Объяви это ПЕРЕД циклом while(true) генерации LLaMA
    float speech_vad_start_ms = 0.0f;
    float llama_start_generation_time = 0.0f;  // <-- ДОБАВИТЬ ИНИЦИАЛИЗАЦИЮ
    llama_start_time = get_current_time_ms();

    // ОПТИМИЗАЦИЯ: Убрана первичная токенизация. Проверяем только пустоту строки.
    if (text_heard.empty() || force_speak) {
            audio.clear();
            {   // Сброс под защитой мьютекса
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                g_hotkey_pressed = "";
            }
            force_speak = false; // Сбрасываем флаг ПЕРЕД continue, чтобы он не залип
            continue;
        }
    trim(text_heard);
    text_heard_prev = text_heard;
    n_past_prev = embd_inp.size();
    past_prev_arr.push_back(embd_inp.size());
    std::string translation_full = "";
    std::string bot_name_current = params.bot_name;
    std::string bot_name_current_ru = params.bot_name;
    std::string text_heard_with_instruct = text_heard;

    if (params.translate) bot_name_current_ru = translit_en_ru(params.bot_name);
    int n_comas = 0;

    // ===== ФОРМАТИРОВАНИЕ РЕПЛИКИ ПОЛЬЗОВАТЕЛЯ ДЛЯ LLAMA =====
    // Алгоритмический смысл: модели YandexGPT/Llama-3 требуют явных разделителей ролей в контексте.
    // Мы формируем их ТОЛЬКО для embd_inp. В stdout эта разметка попадать не должна.
        if (last_output_has_username && !user_typed_this) {
        // Предыдущий ответ модели уже содержал имя пользователя — добавляем только пробел для склейки
            text_heard.insert(0, 1, ' ');
        text_heard_with_instruct.insert(0, 1, ' ');
        } else {
        // Стандартное начало реплики пользователя в контексте модели
            text_heard.insert(0, params.person + chat_symb + " ");
        text_heard_with_instruct.insert(0, params.instruct_preset_data["user_message_prefix"] + "\n" + params.person + chat_symb + " ");
        }

    // ===== ФОРМАТИРОВАНИЕ ОТВЕТА БОТА =====
        text_heard += "\n\n" + params.bot_name + chat_symb;
    text_heard_with_instruct += params.instruct_preset_data["user_message_suffix"] + "\n" + params.instruct_preset_data["bot_message_prefix"] + "\n" + params.bot_name + chat_symb;

    // ===== ВЫВОД В КОНСОЛЬ =====
    // Перезаписываем строку ожидания и выводим реплику пользователя
    printf("\r");
#ifdef _WIN32
    set_console_color(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
#else
    printf("\033[32m");
#endif
    printf("%s%s ", params.person.c_str(), chat_symb.c_str());
    reset_console_color();
    printf("%s\n", user_typed_this ? user_typed.c_str() : text_heard_trimmed.c_str());

    // Выводим имя бота (цветной)
#ifdef _WIN32
    set_console_color(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
#else
    printf("\033[32m");
#endif
    printf("%s%s", params.bot_name.c_str(), chat_symb.c_str());
    reset_console_color();
    fflush(stdout);

    // НЕ ТОКЕНИЗИРУЕМ здесь — токенизация будет ниже, после split_after

    int split_after = params.split_after;

    // ЕДИНСТВЕННАЯ ТОКЕНИЗАЦИЯ: сразу в embd
    embd = ::llama_tokenize(ctx_llama, text_heard_with_instruct, false);
    input_tokens_count = embd.size();

    // Append the new input tokens to the session_tokens vector
    if (!path_session.empty()) {
        // Используем embd (актуальные токены), а не удаленный вектор tokens
        session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
    }

    float temp_next = params.temp;
    int n_discard = 0;
    int n_left = 0;

    // =================================================================
    // ЦИКЛ ГЕНЕРАЦИИ ТЕКСТА (LLaMA)
    // =================================================================
    // В этом цикле модель генерирует токен за токеном до тех пор,
    // пока не встретит стоп-последовательность или не достигнет лимита
    // =================================================================
    bool done = false;
    std::string text_to_speak;
    std::string full_response_text;   // полный текст ответа для Regenerate
    int new_tokens = 0;
    bool first_token_after_bot = true;  // ← СБРАСЫВАЕМ ФЛАГ ПЕРЕД КАЖДОЙ НОВОЙ ГЕНЕРАЦИЕЙ

    // СБРОС СТАТИЧЕСКИХ ПЕРЕМЕННЫХ перед новой генерацией
    {
        static std::string pending_fragment;
        static int pending_count;
        static bool first_token_after_bot_static;
        pending_fragment = "";
        pending_count = 0;
        first_token_after_bot_static = true;
    }

    while (true) {
        // =============================================================
        // ПРОВЕРКА ПРЕРЫВАНИЯ ПО ГОРЯЧЕЙ КЛАВИШЕ
        // =============================================================
        // Каждую итерацию цикла проверяем, не нажал ли пользователь
        // Ctrl+Space или Alt для прерывания генерации.
        // Это позволяет мгновенно остановить генерацию, не дожидаясь
        // окончания текущего токена или следующей итерации главного цикла.
        // =============================================================
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            if (!g_hotkey_pressed.empty()) {
                // Нажата горячая клавиша — прерываем генерацию
                llama_interrupted.store(1);
                g_is_interrupted.store(true);
                done = true;

                // Очищаем текст для озвучки, чтобы не отправлять в TTS
                text_to_speak = "";

                // Сбрасываем горячую клавишу после обработки
                g_hotkey_pressed = "";

                printf(" [Hotkey interrupt: generation stopped]\n");
                break;
            }
        }


// ============================================================
// БЛОК УПРАВЛЕНИЯ ГЕНЕРАЦИЕЙ ТОКЕНОВ
// ============================================================

// predict
if (new_tokens > params.n_predict) break;      // превышен лимит токенов ответа
new_tokens++;                                   // увеличиваем счётчик сгенерированных токенов
if (embd.size() > 0) {                          // есть токены на декодирование
    if (n_past + (int) embd.size() > n_ctx) {   // контекст переполнен → нужен сдвиг

// ============================================================================
// РОТАЦИЯ КОНТЕКСТА (CONTEXT SHIFT) — ИСПРАВЛЕННАЯ ВЕРСИЯ
// ============================================================================
if (n_past + (int)embd.size() > n_ctx) {
    // ------------------------------------------------------------------------
    // ШАГ 1: Получаем вокабуляр для работы с BOS-токеном
    // ------------------------------------------------------------------------
    const llama_vocab * vocab_llama = llama_model_get_vocab(model_llama);

    // ------------------------------------------------------------------------
    // ШАГ 2: Вычисляем, сколько токенов можно удалить
    // ------------------------------------------------------------------------
    const int n_left = std::max(0, n_past - n_keep);

    // n_discard = сколько токенов удаляем (1/4 от n_left, но минимум 1)
    int n_discard = 0;
    if (n_left > 0) {
        n_discard = std::max(1, n_left / 4);
        n_discard = std::min(n_discard, n_left);
    }

    // ------------------------------------------------------------------------
    // ШАГ 3: Флаг успешного сдвига
    // ------------------------------------------------------------------------
    bool context_updated = false;

    // ------------------------------------------------------------------------
    // ШАГ 4: Основная логика сдвига (если есть что удалять)
    // ------------------------------------------------------------------------
    if (n_discard > 0 && n_keep + n_discard <= n_past) {
        if (n_keep >= 0 && n_keep + n_discard <= (int)embd_inp.size()) {

            // 4.1 Удаляем диапазон токенов из KV-кэша модели
            llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, n_keep, n_keep + n_discard);

            // 4.2 Сдвигаем оставшиеся токены влево
            if (n_keep + n_discard < n_past) {
                llama_memory_seq_add(llama_get_memory(ctx_llama), 0, n_keep + n_discard, n_past, -n_discard);
            }

            // 4.3 Удаляем те же токены из локального буфера embd_inp
            embd_inp.erase(embd_inp.begin() + n_keep, embd_inp.begin() + n_keep + n_discard);

            // 4.4 Если используется сессия, синхронизируем её
            if (!path_session.empty() && !session_tokens.empty()) {
                size_t start = std::min((size_t)n_keep, session_tokens.size());
                size_t end = std::min((size_t)(n_keep + n_discard), session_tokens.size());
                if (start < end) {
                    session_tokens.erase(session_tokens.begin() + start, session_tokens.begin() + end);
                }

                // Ограничение размера сессии (защита от утечки памяти)
                if (session_tokens.size() > (size_t)(n_ctx * 2)) {
                    fprintf(stderr, "[warn] Session tokens overflow (%zu), trimming\n", session_tokens.size());
                    session_tokens.resize(n_ctx);
                }
            }

            context_updated = true;

            // Короткое сообщение в консоль
            printf("\n[Сдвиг: удал. %d ток, ост: %zu]\n", n_discard, embd_inp.size());
        }
    }

    // ------------------------------------------------------------------------
    // ШАГ 5: Fallback
    // ------------------------------------------------------------------------
    if (!context_updated) {
        size_t new_size = std::min((size_t)std::max(0, n_keep), embd_inp.size());
        if (new_size < embd_inp.size()) {
            embd_inp.resize(new_size);

            if (!path_session.empty() && !session_tokens.empty()) {
                session_tokens.resize(std::min((size_t)std::max(0, n_keep), session_tokens.size()));
            }

            printf("\n[Сдвиг: сброс до %zu ток.]\n", embd_inp.size());
        }
    }

    // ------------------------------------------------------------------------
    // ШАГ 6: Обновляем счётчики
    // ------------------------------------------------------------------------
    n_past = (int)embd_inp.size();
    n_session_consumed = n_past;

    // ------------------------------------------------------------------------
    // ШАГ 7: Проверяем и восстанавливаем BOS-токен
    // ------------------------------------------------------------------------
    if (vocab_llama) {
        const llama_token bos_token = llama_token_bos(vocab_llama);
        if (!embd_inp.empty() && embd_inp[0] != bos_token) {
            embd_inp.insert(embd_inp.begin(), bos_token);

            if (!session_tokens.empty()) {
                session_tokens.insert(session_tokens.begin(), bos_token);
            }

            n_past = (int)embd_inp.size();
            n_session_consumed = n_past;
            printf("[BOS]");
        }
    } else {
        fprintf(stderr, "WARNING: vocab_llama is null, cannot check BOS token\n");
    }

    // ------------------------------------------------------------------------
    // ШАГ 8: Отключаем сессию после сдвига
    // ------------------------------------------------------------------------
    path_session = "";

    // ========== 🔧 ==========
    // После сдвига контекста ОЧИЩАЕМ embd, потому что токены в нём имеют старые pos
    // Это безопасно: при следующей итерации цикла embd пуст, модель запросит новый токен
    // ========================

    embd.clear();
    text_to_speak = "";  // Также очищаем буфер озвучки, чтобы не отправить мусор
    past_prev_arr.clear();
    continue;  // Переходим к следующей итерации, минуя декодирование старых токенов
}

}
    // Попытка повторного использования совпадающего префикса из загруженной сессии
    if (n_session_consumed < (int) session_tokens.size()) {
        size_t i = 0;
        // Добавляем безопасные границы
        int max_check = std::min((int)embd.size(), (int)session_tokens.size() - n_session_consumed);

        for ( ; i < (size_t)max_check; i++) {
            // Дополнительная проверка индекса
            if (n_session_consumed >= (int)session_tokens.size()) {
                break;
            }

            if (embd[i] != session_tokens[n_session_consumed]) {
                session_tokens.resize(n_session_consumed);
                break;
            }

            //Вместо n_past++ — добавляем токен в embd_inp, чтобы сохранить инвариант
            embd_inp.push_back(embd[i]);
            n_session_consumed++;

            if (n_session_consumed >= (int) session_tokens.size()) {
                i++;
                break;
            }
        }
        if (i > 0) {
            embd.erase(embd.begin(), embd.begin() + i);
        }
        //Обновляем n_past один раз — после всех изменений embd_inp
        n_past = embd_inp.size();
    }

    // Если есть новые токены и используется сессия, добавляем их в сессию
    if (embd.size() > 0 && !path_session.empty()) {
        session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
        n_session_consumed = session_tokens.size();  // Обновляем счётчик потреблённых токенов
    }
        // безопасная подготовка batch с обнулением logits
    {
        if (embd.empty()) {
            embd.clear();
            continue;
        }
        if (embd.size() > 2048) {
            fprintf(stderr, "ERROR: Input sequence too long (%zu tokens). Max batch size is 2048.\n", embd.size());
            embd.clear();
            continue;
        }

        batch.n_tokens = static_cast<int>(embd.size());

        // Обнуляем logits только для реально используемых токенов
        // memset безопасен, так как работает с байтами и не выходит за границы
        memset(batch.logits, 0, sizeof(batch.logits[0]) * batch.n_tokens);

        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = embd[i];
            batch.pos[i] = n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == batch.n_tokens - 1);
        }
    }

// Выполняем декодирование (потокобезопасно с защитой от reset)
{
    std::lock_guard<std::mutex> lock(g_llama_mutex);

    // Проверяем, что контекст не был сброшен во время генерации
    if (!ctx_llama) {
        fprintf(stderr, "\n[Context was reset during generation - aborting]\n");
        done = true;
        break;
    }

    if (llama_decode(ctx_llama, batch)) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        fprintf(stderr, "\n LLaMA decoding failed. Press ENTER to continue...\n");
        fflush(stderr);
        std::string dummy;
        std::getline(std::cin, dummy);
        embd.clear();
        n_past = embd_inp.size();
        n_session_consumed = n_past;
        continue;
    }
}
}  // Закрываем блок декодирования

// Добавляем обработанные токены в общий контекст
embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
n_past = embd_inp.size();  // Обновляем позицию в контексте
embd.clear();  // Очищаем временный буфер
if (done) break;  // Если завершено, выходим из цикла
// Инициализируем переменные для обработки следующего токена
std::string out_token_str = "";
char out_token_symbol;
// Запоминаем время начала генерации (если ещё не установлено)
    if (llama_start_generation_time == 0.0f) llama_start_generation_time = get_current_time_ms();
    {
        // Обработка вне пользовательского ввода, сэмплирование следующего токена
        // Если используется сессия и нужно сохранить её
        if (!path_session.empty() && need_to_save_session) {
            need_to_save_session = false;
            // Сохраняем состояние модели в файл сессии
            llama_state_save_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
        }
        llama_token id = 0;  // ID сгенерированного токена
        int person_name_is_found = 0; // Флаг обнаружения имени пользователя
        int bot_name_is_found = 0;    // Флаг обнаружения имени бота
        // Сэмплирование токена с возможным изменением температуры
        if (temp != temp_next) // Повышенная температура только для 1 токена
            {
                id = llama_sampler_sample(smpl_high_temp, ctx_llama, -1);  // Сэмплируем с высокой температурой
                temp = temp_next = params.temp; // Возвращаем нормальную температуру
            }
        else // Нормальная температура
            {
                // std::lock_guard<std::mutex> lock(g_llama_mutex);
                id = llama_sampler_sample(smpl, ctx_llama, -1);  // Сэмплируем с нормальной температурой
            }

        // ============================================================
        // ПРОВЕРКА: является ли токен специальным стоп-токеном
        // (включая EOS, bot_message_suffix и другие из JSON-пресета)
        // ============================================================
        bool is_stop_token = false;
        for (int i = 0; i < special_token_count; i++) {
            if (id == special_token_ids[i]) {
                is_stop_token = true;
                break;
            }
        }
        // Дополнительно проверяем стандартный EOS (на всякий случай)
        if (id == llama_vocab_eos(vocab_llama)) {
            is_stop_token = true;
        }

        if (is_stop_token) {
            // Немедленно останавливаем генерацию, не добавляя токен в контекст
            done = true;
            break;
        }
        // ============================================================

        // Если токен не является токеном окончания (EOS) – эта проверка теперь избыточна,
        // но оставляем для совместимости (EOS уже отловлен выше).
        if (id != llama_vocab_eos(vocab_llama)) {

            // Добавляем токен в контекст для следующей итерации
            embd.push_back(id);

        out_token_str = llama_token_to_piece(ctx_llama, id);

        // ============================================================
        // ФИЗИЧЕСКИЙ СМЫСЛ: Замена плейсхолдеров {0}/{1} и фильтрация мусора
        // Модели могут генерировать плейсхолдеры из весов ИЛИ разбивать их на субтокены.
        // Обрабатываем оба случая: полную строку и частичные совпадения.
        // ============================================================

        // 1. Полная замена для обычных токенов
        size_t pos0 = out_token_str.find("{0}");
        if (pos0 != std::string::npos) {
            out_token_str.replace(pos0, 3, params.person);
        }
        size_t pos1 = out_token_str.find("{1}");
        if (pos1 != std::string::npos) {
            out_token_str.replace(pos1, 3, params.bot_name);
        }
        size_t pos2 = out_token_str.find("{2}");
        if (pos2 != std::string::npos) {
            out_token_str.replace(pos2, 3, time_str);
        }
        size_t pos3 = out_token_str.find("{3}");
        if (pos3 != std::string::npos) {
            out_token_str.replace(pos3, 3, year_str);
        }
        size_t pos5 = out_token_str.find("{5}");
        if (pos5 != std::string::npos) {
            out_token_str.replace(pos5, 3, ymd);
        }

        // 2. Обработка разбитых токенов (субтокены типа "{0", "}", "{1")
        if (out_token_str == "{0" || out_token_str == "{0}") {
            out_token_str = params.person;
        } else if (out_token_str == "{1" || out_token_str == "{1}") {
            out_token_str = params.bot_name;
        } else if (out_token_str == "{2" || out_token_str == "{2}") {
            out_token_str = time_str;
        } else if (out_token_str == "{3" || out_token_str == "{3}") {
            out_token_str = year_str;
        } else if (out_token_str == "{5" || out_token_str == "{5}") {
            out_token_str = ymd;
        } else if (out_token_str == "}" && !text_to_speak.empty()) {
            if (text_to_speak.size() >= 2) {
                std::string last2 = text_to_speak.substr(text_to_speak.size() - 2);
                if (last2 == "{0" || last2 == "{1" || last2 == "{2" || last2 == "{3" || last2 == "{5") {
                    text_to_speak.pop_back();
                    out_token_str = "";
                }
            }
        }

        // ============================================================
        // 3. УМНАЯ ФИЛЬТРАЦИЯ СПЕЦТОКЕНОВ (без хардкода строк)
        // ============================================================
        // Проблема: старый код хардкодил "<|", "|>", "<|eot" и пропускал
        // разбитые BPE-субтокены спецтокенов.
        //
        // Решение:
        // a) Проверяем ID токена по списку special_token_ids (собран из
        //    JSON-пресета, --stop-words и базовых EOT-маркеров).
        // b) Если токен совпал — подавляем его и сбрасываем накопительный буфер.
        // c) Если токен НЕ совпал, но содержит '<' или '|' — накапливаем
        //    в буфере (ждём полный спецтокен).
        // d) Обычный токен сначала "выталкивает" буфер, потом выводится сам.
        // e) Если накопилось > MAX_PENDING подозрительных токенов — выводим
        //    принудительно (это не спецтокен, а обычный текст).
        // ============================================================

        // Шаг 3a: Проверяем ID токена по заранее собранному списку
        bool is_special_id = false;
        for (int si = 0; si < special_token_count; si++) {
            if (id == special_token_ids[si]) {
                is_special_id = true;
                break;
            }
        }

        // Шаг 3b: Дополнительно проверяем EOS через токен конца словаря
        if (!is_special_id) {
            llama_token eos_token = llama_vocab_eos(vocab_llama);
            if (id == eos_token) {
                is_special_id = true;
            }
        }

        // Шаг 3c: Буфер для накопления подозрительных фрагментов
        static std::string pending_fragment = "";
        static int pending_count = 0;
        static const int MAX_PENDING = 8;
        // (переменные статические, но сбрасываются перед каждым новым диалогом)

        // Шаг 3d: Определяем, похож ли токен на фрагмент спецтокена
        bool looks_like_special = false;
        if (!out_token_str.empty()) {
            for (size_t ci = 0; ci < out_token_str.size(); ci++) {
                if (out_token_str[ci] == '<' || out_token_str[ci] == '|') {
                    looks_like_special = true;
                    break;
                }
            }
        }

        bool should_print = true;

        if (is_special_id) {
            // Токен гарантированно специальный — подавляем и очищаем буфер
            should_print = false;
            pending_fragment = "";
            pending_count = 0;
        } else if (looks_like_special && !out_token_str.empty()) {
            // Токен похож на фрагмент спецтокена — накапливаем
            pending_fragment += out_token_str;
            pending_count++;

            if (pending_count >= MAX_PENDING) {
                // Слишком много фрагментов — это обычный текст
                printf("%s", pending_fragment.c_str());
                fflush(stdout);
                text_to_speak += pending_fragment;
                tokens_in_reply += utf8_length(pending_fragment);
                pending_fragment = "";
                pending_count = 0;
            } else {
                should_print = false;
            }
        } else {
            // Обычный токен — выводим накопленный буфер, если есть
            if (!pending_fragment.empty()) {
                printf("%s", pending_fragment.c_str());
                fflush(stdout);
                text_to_speak += pending_fragment;
                tokens_in_reply += utf8_length(pending_fragment);
                pending_fragment = "";
                pending_count = 0;
            }
        }
        // Убираем лишний пробел в начале первого токена после имени бота
        {
            static bool first_token_after_bot_static = true;
            if (first_token_after_bot_static && !out_token_str.empty() && out_token_str[0] == ' ') {
                out_token_str.erase(0, 1);  // убираем первый пробел
            }
            first_token_after_bot_static = false;
        }

        // Выводим только очищенный и безопасный токен
        if (should_print && !out_token_str.empty()) {
            // Очистка спецтокенов для вывода на экран
            std::string display_str = out_token_str;
            display_str = ::replace(display_str, "<|start_header_id|>", "");
            display_str = ::replace(display_str, "<|end_header_id|>", "");
            display_str = ::replace(display_str, "<|eot_id|>", "");
            display_str = ::replace(display_str, "<|im_start|>", "");
            display_str = ::replace(display_str, "<|im_end|>", "");
            display_str = ::replace(display_str, "assistant", "");
            display_str = ::replace(display_str, "system", "");
            display_str = ::replace(display_str, "user", "");

            if (!display_str.empty()) {
                printf("%s", display_str.c_str());
                fflush(stdout);
            }
            text_to_speak += out_token_str;  // В TTS идёт исходная строка (очистится позже)
        }

        if (should_print) {
            tokens_in_reply += utf8_length(out_token_str);
        }
        // ============================================================

            // Проверка на зацикливание последовательности
            if (params.seqrep)  // Если включена проверка на повторения
            {
                // Обновляем "игольчатый" буфер (для поиска повторений)
                if (utf8_length(last_output_needle) > 25)
                    last_output_needle = utf8_substr(last_output_needle, 5, utf8_length(last_output_needle)-5);
                last_output_needle += out_token_str;  // Добавляем текущий токен

                out_token_symbol = out_token_str[out_token_str.size()-1];  // Последний символ токена

                // Если символ является знаком препинания (конец слова/предложения)
                if (out_token_symbol == ' ' || out_token_symbol == '.' || out_token_symbol == ',' ||
                    out_token_symbol == '!' || out_token_symbol == '?')
                {
                    // Проверяем, есть ли эта последовательность в буфере
                    if (utf8_length(last_output_buffer) > 300 && utf8_length(last_output_needle) >= 20 &&
                        last_output_buffer.find(last_output_needle) != std::string::npos)
                    {
                        // Обнаружено зацикливание - выводим сообщение
                        printf(" [LOOP: %s] (length: %d)\n", last_output_needle.c_str(), utf8_length(last_output_needle));
                        // Рассчитываем количество символов и токенов для удаления
                        int symbols_to_delete = static_cast<int>(utf8_length(last_output_needle) * 1); // Удаляем всю последовательность
                        const std::vector<llama_token> tokens_to_del = llama_tokenize(ctx_llama, last_output_needle.c_str(), false);
                        int rollback_num = tokens_to_del.size();  // Количество токенов для отката
                        if (rollback_num) // Если есть токены для удаления
                        {
                            // Удаляем токены из контекста
                            embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                            n_past = embd_inp.size();  // Обновляем позицию
                            n_session_consumed = n_past;  // Обновляем сессию

                            // Очищаем KV-кэш модели (через новый memory API)
                            llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);

                            // Удаляем текст из буферов
                            text_to_speak = utf8_substr(text_to_speak, 0, utf8_length(text_to_speak)-symbols_to_delete);
                            last_output_needle = utf8_substr(last_output_needle, 0, utf8_length(last_output_needle)-symbols_to_delete);
                            last_output_buffer = utf8_substr(last_output_buffer, 0, utf8_length(last_output_buffer)-symbols_to_delete);

                            temp_next = 1.8; // Повышаем температуру для следующего токена (чтобы выйти из цикла)
                        }
                    }
    }
            // Обновляем буфер для проверки повторений
            if (utf8_length(last_output_buffer) > 1000)
                last_output_buffer = utf8_substr(last_output_buffer, 100, last_output_buffer.size()-100);
            last_output_buffer += out_token_str;  // Добавляем текущий токен в буфер
        }
        // Проверка на появление имён персонажей
        // Если обнаружено имя пользователя
        if (text_to_speak == '\n'+params.person+':')
        {
            person_name_is_found = 1; // Установить флаг
            translation_is_going = 0; // Остановить перевод

        }
        // Если обнаружено имя бота (формат: \nИмя:)
        else if (text_to_speak[0] == '\n' && text_to_speak[text_to_speak.size()-1] == ':' && text_to_speak.size() < 10)
        {
            bot_name_is_found = 1;         // Установить флаг
            bot_name_current = text_to_speak.substr(1, text_to_speak.size()-2);  // Извлечь имя
            if (params.translate)
            bot_name_current_ru = translit_en_ru(bot_name_current);  // Транслитерировать на русский
            translation_full = "";         // Очистить буфер перевода
            text_to_speak = "";            // Очистить текст для озвучки
        }
        // Обнуляем текст для TTS, если было обнаружено имя бота — в любом виде
        if (bot_name_is_found) {
            text_to_speak = "";
        }
        // Обработка знаков препинания и разбиение текста
        int text_len = text_to_speak.size();

        if (text_len > 0 && text_to_speak[text_len-1] == ',') n_comas++;
        if (text_len > 0 && new_tokens == split_after && params.split_after && text_to_speak[text_len-1] == '\'')

        split_after++;
        // Не разбиваем по Mr.
        if (text_to_speak.size() >= 3 && text_to_speak.substr(text_to_speak.size()-3, 3) == "Mr.")
            text_to_speak[text_len-1] = ' ';


            // Проверяем каждые 2 токена, НО не прерываем первые 50 токенов
        if (new_tokens % 2 == 0 && new_tokens > 50)  // ← ФИКС 1: даём Эмме начать фразу
        {
            audio.get(2000, pcmf32_cur);
            // Проверяем уровень энергии (VAD - Voice Activity Detection)
            int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms,
                                            params.vad_thold, params.freq_thold, params.print_energy,
                                            params.vad_start_thold);

            /// Если обнаружена речь или нажата горячая клавиша
            if ((!params.push_to_talk && vad_result == 1) ||
                hk_copy == "Ctrl+Space" || hk_copy == "Alt")
            {
                // 1. Взводим флаги для остановки сетевых потоков и генерации
                llama_interrupted.store(1);
                g_is_interrupted.store(true);
                // 2. ОЧИЩАЕМ буфер текста, чтобы следующая итерация цикла не отправила "ошметки" фразы в TTS
                text_to_speak = "";
                // 3. ОСТАНОВКА ЗВУКА (SDL2)
                // Очищаем очередь аудио, чтобы недоигранные фрагменты исчезли мгновенно
                // В talk-llama вывод обычно идет через устройство с ID 2 или глобальный микшер
                SDL_PauseAudio(1);         // Ставим на паузу всё аудио
                SDL_ClearQueuedAudio(2);   // Очищаем очередь вывода (ID 2 — стандарт для вывода в этом коде)
                SDL_PauseAudio(0);         // Снимаем паузу (устройство готово к новым данным, но очередь пуста)
                llama_interrupted_time = get_current_time_ms();
                printf(" [Speech/Stop!]\n");

                // === НЕ очищаем аудиобуфер при VAD-прерывании ===
                // Аудио должно быть доступно для последующего распознавания в главном цикле.
                // Очистка произойдёт после успешного распознавания фразы.
                // audio.clear();  // ← ЗАКОММЕНТИРОВАНО

                // Сигнализируем внешнему сервису
                allow_xtts_file(params.xtts_control_path, 0);
                done = true;
                { // Сброс под защитой мьютекса
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    g_hotkey_pressed = "";
                }
                // 5. Выход из цикла генерации
                break;
            }
        }
// Очистка микрофона после генерации 20 токенов
// Это помогает избежать накопления шума в буфере микрофона
    if (new_tokens == 20 && !llama_interrupted)
    {
        audio.clear();  // Очищаем буфер аудио
    }


// Разбиение текста для TTS
// Условия для разбиения: текст достаточно длинный и не найдено имя персонажа
if (text_len >= 2 && new_tokens >= 5 && !person_name_is_found &&
    (
        // 1. ЕСТЕСТВЕННЫЕ КОНЦЫ ПРЕДЛОЖЕНИЙ - отправляем сразу
        text_to_speak[text_len-1] == '.' ||    // Конец предложения
        text_to_speak[text_len-1] == '?' ||    // Вопрос
        text_to_speak[text_len-1] == '!' ||    // Восклицание
        text_to_speak[text_len-1] == ':' ||    // ← двоеточие перед перечислением
        text_to_speak[text_len-1] == '\n' ||   // Новая строка (разделитель)

        // РЕЖИМ ПРИНУДИТЕЛЬНОГО РАЗБИЕНИЯ
        // Срабатывает каждые N токенов, если параметр --split-after N указан в командной строке
        // new_tokens % params.split_after == 0  - каждый N-ный токен (100, 200, 300...)
        // new_tokens > 50 - защита от слишком раннего срабатывания
        //
        // ВНИМАНИЕ: Этот режим режет текст независимо от знаков препинания!
        // Может разрывать слова и предложения. Используйте осторожно.
        (params.split_after > 0 && new_tokens % params.split_after == 0 && new_tokens > 50)
    )
)
{
    // Если идёт процесс перевода, добавляем текст в буфер перевода
    if (translation_is_going == 1)
    {
        translation_full += text_to_speak;  // Накапливаем текст для перевода
        //fprintf(stdout, " translation_full: (%s)\n", translation_full.c_str());  // Отладочный вывод
    }

    // =================================================================
    // ПОДГОТОВКА ТЕКСТА ДЛЯ TTS: УДАЛЕНИЕ ИМЕНИ БОТА
    // =================================================================
    // Удаляем все вхождения "Эмма:" и "Эмма :" из текста
    std::string bot_prefix_1 = params.bot_name + ":";
    std::string bot_prefix_2 = params.bot_name + " :";
    text_to_speak = ::replace(text_to_speak, bot_prefix_1, "");
    text_to_speak = ::replace(text_to_speak, bot_prefix_2, "");
    trim(text_to_speak);

    // Удаляем имя пользователя только если оно в конце текста
    // Дополнительная проверка: antiprompts не пуст и содержит корректный маркер
    if (!antiprompts.empty() && !antiprompts[0].empty()) {
        std::string user_name_marker = antiprompts[0];
        // Проверяем, что текст не короче маркера и заканчивается на него
        if (text_to_speak.size() >= user_name_marker.size()) {
            std::string end_of_text = text_to_speak.substr(text_to_speak.size() - user_name_marker.size());
            if (end_of_text == user_name_marker) {
                text_to_speak = text_to_speak.substr(0, text_to_speak.size() - user_name_marker.size());
                trim(text_to_speak);
            }
        }
    }

    // Если есть текст для озвучки (первая или средняя часть предложения)
    // Очистка от спецтокенов перед TTS
    text_to_speak = ::replace(text_to_speak, "<|eot_id|>", "");
    text_to_speak = ::replace(text_to_speak, "<|start_header_id|>", "");
    text_to_speak = ::replace(text_to_speak, "<|end_header_id|>", "");
    text_to_speak = ::replace(text_to_speak, "<|im_end|>", "");
    text_to_speak = ::replace(text_to_speak, "<|im_start|>", "");
    text_to_speak = ::replace(text_to_speak, "</s>", "");
    text_to_speak = ::replace(text_to_speak, "<|endoftext|>", "");
    text_to_speak = ::replace(text_to_speak, "<|", "");
    text_to_speak = ::replace(text_to_speak, "|>", "");
    trim(text_to_speak);

    if (text_to_speak.size())
    {
        // Перевод текста (если включён)
        // Каждое сгенерированное предложение переводится той же моделью LLaMA в том же контексте
        if (params.translate)
        {
            // Если перевод ещё не начат
            if (translation_is_going == 0)
            {
                std::string text_to_speak_translated = "";
                // Запоминаем размер контекста до перевода для последующего отката
                n_embd_inp_before_trans = embd_inp.size();
                fprintf(stdout, "\n	Перевод: %d", n_embd_inp_before_trans);  // Выводим позицию начала перевода
                // Формируем промпт для перевода
                std::string trans_prompt = "\nПеревод последнего предложения на русский.\n"+bot_name_current_ru+":"+translation_full;
                // Преобразуем промпт перевода в токены
                std::vector<llama_token> trans_prompt_emb = ::llama_tokenize(ctx_llama, trans_prompt, false);
                // Вставляем промпт перевода в начало следующего батча (инъекция промпта)
                embd.insert(embd.end(), trans_prompt_emb.begin(), trans_prompt_emb.end());
                translation_is_going = 1;  // Перевод начат
                text_to_speak = "";        // Очищаем текст для озвучки
                continue;  // Переходим к следующей итерации для обработки промпта перевода
            }
        }

try
    {
        // Накапливаем полный ответ перед очисткой
        if (!text_to_speak.empty()) {
            full_response_text += text_to_speak;
        }

        // Захватываем ВСЁ по значению — безопасно с мьютексом
        std::string voice_copy = current_voice;  // <-- КОПИЯ
        safe_thread_emplace(threads, [text_to_speak, voice_copy, params]() {
            send_tts_async(text_to_speak, voice_copy, params.language, params.xtts_url);
        });
        // Очищаем локальную переменную
        text_to_speak = "";

        // Если задержка перед XTTS включена, делаем паузу
        // Это помогает ускорить инференс xtts
        if (params.sleep_before_xtts)
            std::this_thread::sleep_for(std::chrono::milliseconds(params.sleep_before_xtts));


    }
                // Обработка исключений при создании потока
                catch (const std::exception& ex) {
                    // Выводим сообщение об ошибке создания потока
                    std::cerr << "[Exception]: Failed to push_back mid thread: " << ex.what() << '\n';
                }

                // Проверяем уровень энергии, если пользователь говорит
                if (!params.push_to_talk || (params.push_to_talk && hk_copy == "Alt"))
                {
                    // Получаем аудио данные (неблокирующий вызов)
                    audio.get(params.interrupt_check_ms, pcmf32_cur);

                    int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms,
                    params.vad_thold, params.freq_thold, params.print_energy,
                    params.vad_start_thold);

                    if (vad_result == 1) {
                        if (speech_vad_start_ms == 0.0f) {
                            speech_vad_start_ms = get_current_time_ms() * 1000.0f;
                        }

                        if ((get_current_time_ms() * 1000) - speech_vad_start_ms > params.interrupt_threshold_ms) {
                            printf(" [Speech interruption confirmed!]\n");
                            llama_interrupted.store(1);
                            g_is_interrupted.store(true);
                            allow_xtts_file(params.xtts_control_path, 0);
                            done = true;
                            break;
                        }
                    } else {
                        speech_vad_start_ms = 0;
                    }
                }


                // Удаление перевода из контекста (откат после перевода)
                if (params.translate && translation_is_going == 1)
                {
                    translation_is_going = 0; // Перевод завершён
                    // Если есть сохранённая позиция до перевода и текущий контекст
                    if (n_embd_inp_before_trans && embd_inp.size())
                    {
                        // Вычисляем количество токенов, которые нужно удалить (перевод)
                        int rollback_num = embd_inp.size()-n_embd_inp_before_trans;
                        if (rollback_num)
                        {
                            // Удаляем токены перевода из контекста
                            embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                            n_past = embd_inp.size();  // Обновляем позицию в контексте
                            n_session_consumed = n_past;  // Обновляем сессию
                            // Удаляем последовательность из KV-кэша (новый API)
                            llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);
                            printf("\n"); // Выводим пустую строку для разделения перевода и оригинала
                        }
                        continue;  // Переходим к следующей итерации (продолжаем генерацию)
                    }
                }
            }
        }
    }
}

// Обработка последнего вывода и антипромптов
{
    // Возвращаемся к старой, проверенной логике: собираем последние токены из embd_inp
    // Это более стабильно для определения антипромптов, включая EOT-маркеры
    std::string last_output;

    // Собираем последние 50 символов из контекста (не токенов, а символов!)
    // Это даёт достаточно контекста для определения конца предложения
    int total_chars = 0;
    int start_index = (int)embd_inp.size() - 1;

    for (int i = start_index; i >= 0 && total_chars < 100; i--) {
        std::string piece = llama_token_to_piece(ctx_llama, embd_inp[i]);
        total_chars += utf8_length(piece);
        last_output = piece + last_output;
    }

    // Также добавляем текущий текст, который ещё не в embd_inp
    if (!text_to_speak.empty()) {
        last_output += text_to_speak;
    }


    int i_antiprompt = 0;
    last_output_has_username = false;  // Флаг наличия имени пользователя
    bool antiprompt_matched = false;  // Флаг, был ли найден антипромпт

    // Проходим по всем антипромптам
    for (std::string & antiprompt : antiprompts)
    {
        // ============================================================
        // ОБРАБОТКА НЕСКОЛЬКИХ ИМЁН ПЕРСОНАЖЕЙ ДЛЯ XTTS (multi_chars)
        // ============================================================
        if (params.multi_chars && last_output.size()>=4)
        {
            // Очищаем текст от различных знаков препинания
            last_output = ::replace(last_output, " ???", "");
            last_output = ::replace(last_output, " ??", "");
            last_output = ::replace(last_output, " ?", "");
            last_output = ::replace(last_output, " !!!", "");
            last_output = ::replace(last_output, " !!", "");
            last_output = ::replace(last_output, " !", "");
            last_output = ::replace(last_output, "!!!", "");
            last_output = ::replace(last_output, "!!", "");
            last_output = ::replace(last_output, " ...", "");
            last_output = ::replace(last_output, " .", "");
            last_output = ::replace(last_output, " ,", "");
            last_output = ::replace(last_output, "...", "");
            last_output = ::replace(last_output, "(", "");
            last_output = ::replace(last_output, ")", "");

            // Поиск нового персонажа
            std::smatch matches;
            std::regex r("\n([^:]*):", std::regex::icase | std::regex::optimize);

            if (std::regex_search(last_output, matches, r) && !matches.empty() &&
                matches.size() >= 2 && !matches[1].str().empty() &&
                matches[1].str() != params.person &&
                matches[1].str() != " \n"+params.person)
            {
                std::string current_voice_tmp = matches[1].str();
                current_voice_tmp = ::replace(current_voice_tmp, ":", "");
                current_voice_tmp = ::replace(current_voice_tmp, "\"", "");
                trim(current_voice_tmp);

                if (current_voice_tmp.size()>1 && current_voice_tmp.size()<30)
                {
                    current_voice = current_voice_tmp;
                    std::regex regEx("\n" + current_voice + ":");
                    text_to_speak = std::regex_replace(text_to_speak, regEx, "\n");
                }
            }
        }

        // ============================================================
        // ОБРАБОТКА СТОП-СЛОВ (АНТИПРОМПТОВ)
        // ============================================================
        // Проверяем, заканчивается ли текущий буфер last_output на антипромпт
        // ============================================================

        if (last_output.length() >= antiprompt.length())
        {
            std::string end_of_output = last_output.substr(last_output.length() - antiprompt.length());

            if (end_of_output == antiprompt)
            {
                // --------------------------------------------------------
                // 1. ПРОВЕРКА MIN_TOKENS (слишком короткий ответ)
                // --------------------------------------------------------
                if (params.min_tokens > 0 && tokens_in_reply < params.min_tokens) {
                    if (params.verbose) {
                        printf(" [ignoring antiprompt '%s', too short (%d < %d)] ",
                               antiprompt.c_str(), tokens_in_reply, params.min_tokens);
                    }
                    continue;  // Не останавливаемся, продолжаем генерацию
                }

                // --------------------------------------------------------
                // 2. СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ EOT
                //    Спецтокены <|eot_id|> и bot_message_suffix уже обработанычерез special_token_ids.
                // --------------------------------------------------------

                // --------------------------------------------------------
                // 3. ОБРАБОТКА ИМЕНИ ПОЛЬЗОВАТЕЛЯ (Друг:, Друг :)
                // --------------------------------------------------------
                bool is_user_name_antiprompt = (antiprompt == "\n" + params.person + chat_symb ||
                                                antiprompt == "\n" + params.person + " " + chat_symb);

                if (is_user_name_antiprompt)
                {
                    // Дополнительная проверка: если text_to_speak пустой, не останавливаемся
                    if (text_to_speak.empty() || text_to_speak.length() < 2) {
                        if (params.debug) {
                            printf("\n[DEBUG] User name antiprompt but text_to_speak empty - IGNORED\n");
                        }
                        i_antiprompt++;
                        continue;
                    }

                    // Ищем позицию антипромпта в last_output
                    size_t pos = last_output.rfind(antiprompt);

                    // Проверяем, что перед антипромптом есть \n или это начало строки
                    if (pos == 0 || (pos > 0 && last_output[pos-1] == '\n'))
                    {
                        bool is_at_end = (pos + antiprompt.length() >= last_output.length());

                        if (is_at_end) {
                            antiprompt_matched = true;
                            done = true;
                            if (params.debug) {
                                printf("\n[DEBUG] User name antiprompt at end - stopping\n");
                            }
                        } else {
                            if (params.debug) {
                                printf("\n[DEBUG] User name antiprompt at line start but more text follows - IGNORED\n");
                            }
                            i_antiprompt++;
                            continue;
                        }
                    }
                    else
                    {
                        if (params.debug) {
                            printf("\n[DEBUG] User name antiprompt in middle - IGNORED (continuing)\n");
                        }
                        i_antiprompt++;
                        continue;
                    }
                }
                else
                {
                    // --------------------------------------------------------
                    // 4. ОБРАБОТКА ОСТАЛЬНЫХ АНТИПРОМПТОВ (пользовательские --stop-words)
                    //    Примечание: \n больше не является антипромптом, поэтому
                    //    проверка на списки после \n не требуется.
                    // --------------------------------------------------------
                    antiprompt_matched = true;
                    done = true;

                    if (params.debug) {
                        printf("\n[DEBUG] Antiprompt '%s' matched - stopping\n", antiprompt.c_str());
                    }
                }

                // --------------------------------------------------------
                // 5. ВЫПОЛНЯЕМ ДЕЙСТВИЯ ПРИ ОСТАНОВКЕ
                // --------------------------------------------------------
                if (done)
                {
                    // Удаляем антипромпт из текста для озвучки
                    text_to_speak = ::replace(text_to_speak, antiprompt, "");

                    fflush(stdout);
                    need_to_save_session = true;

                    // Запоминаем, что это был первый антипромпт (имя пользователя)
                    if (i_antiprompt == 0)
                    {
                        last_output_has_username = true;
                        printf(" ");
                    }

                    // --------------------------------------------------------
                    // ПРОВЕРКА МИНИМАЛЬНОГО КОЛИЧЕСТВА ТОКЕНОВ
                    // --------------------------------------------------------
                    static int short_response_attempts = 0;

                    if (params.min_tokens && tokens_in_reply < params.min_tokens)
                    {
                        short_response_attempts++;

                        if (short_response_attempts > 5) {
                            if (params.verbose) {
                                printf("\n[WARN] Too many short responses (%d attempts), accepting as is\n",
                                       short_response_attempts);
                            }
                            short_response_attempts = 0;
                            if (params.debug) {
                                std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
                                printf("\n=====FULL text in embd (%zd tokens, %zd symbols)=====\n%s\n====END====\n",
                                       embd_inp.size(), full_dialog.size(), full_dialog.c_str());
                            }
                            break;
                        }

                        int symbols_to_delete = static_cast<int>(utf8_length(antiprompt) * 1) + 1;
                        const std::vector<llama_token> tokens_to_del = llama_tokenize(ctx_llama, antiprompt.c_str(), false);
                        int rollback_num = tokens_to_del.size() + 1;

                        if (rollback_num)
                        {
                            embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                            n_past = embd_inp.size();
                            n_session_consumed = n_past;
                            llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);

                            if (symbols_to_delete > utf8_length(text_to_speak))
                                text_to_speak = "";
                            else
                                text_to_speak = utf8_substr(text_to_speak, 0, utf8_length(text_to_speak)-symbols_to_delete);

                            temp_next = 1.8;
                            fflush(stdout);
                            printf("\b\b\b\b\b\b\b\b\b\b\b\b");
                            fflush(stdout);
                            done = false;

                            if (params.debug) {
                                printf("\n[DEBUG] Response too short (%d < %d), retry %d/5 with higher temp\n",
                                       tokens_in_reply, params.min_tokens, short_response_attempts);
                            }
                        }

                        if (params.debug)
                        {
                            std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
                            printf("\n=====FULL text in embd (%zd tokens, %zd symbols)=====\n%s\n====END====\n",
                                   embd_inp.size(), full_dialog.size(), full_dialog.c_str());
                        }
                    }
                    else
                    {
                        short_response_attempts = 0;
                        if (params.debug)
                        {
                            std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
                            printf("\n=====FULL text in embd (%zd tokens, %zd symbols)=====\n%s\n====END====\n",
                                   embd_inp.size(), full_dialog.size(), full_dialog.c_str());
                        }
                        break;
                    }
                }
            }
        }
        i_antiprompt++;
    }

    // ========== ФИНАЛЬНАЯ ПРОВЕРКА ==========
    if (antiprompt_matched && params.min_tokens > 0 && tokens_in_reply < params.min_tokens) {
        done = false;
        if (params.verbose) {
            printf(" [Safety: short response protected] ");
        }
    }
} // КОНЕЦ БЛОКА антипромптов

// ### ОБРАБОТКА АУДИОВХОДА И СИГНАЛОВ (VAD) ###
// ВНИМАНИЕ: проверка событий УДАЛЕНА из цикла генерации
// Она теперь выполняется только в главном цикле программы
// для максимальной скорости генерации токенов
}
            // Финальная часть предложения, если осталась
            // Очистка от спецтокенов перед TTS
            text_to_speak = ::replace(text_to_speak, "<|eot_id|>", "");
            text_to_speak = ::replace(text_to_speak, "<|start_header_id|>", "");
            text_to_speak = ::replace(text_to_speak, "<|end_header_id|>", "");
            text_to_speak = ::replace(text_to_speak, "<|im_end|>", "");
            text_to_speak = ::replace(text_to_speak, "<|im_start|>", "");
            text_to_speak = ::replace(text_to_speak, "</s>", "");
            text_to_speak = ::replace(text_to_speak, "<|endoftext|>", "");
            text_to_speak = ::replace(text_to_speak, "<|", "");
            text_to_speak = ::replace(text_to_speak, "|>", "");
            trim(text_to_speak);

            // Добавляем остаток в полный ответ
            if (!text_to_speak.empty()) {
                full_response_text += text_to_speak;
            }

            // Сохраняем ПОЛНЫЙ текст ответа для Regenerate
            {
                std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                g_last_tts_text = full_response_text;
            }

            if (!text_to_speak.empty())  // Если есть текст для озвучки
            {
                std::string text_to_speak_final = text_to_speak; // Создаём локальную копию

                try {
                    std::string voice_copy = current_voice;
                    std::string lang_copy = params.language;
                    std::string url_copy = params.xtts_url;
                    safe_thread_emplace(threads, [text_to_speak_final, voice_copy, lang_copy, url_copy]() {
                        send_tts_async(text_to_speak_final, voice_copy, lang_copy, url_copy);
                    });

                    text_to_speak = ""; // Очищаем оригинальную переменную
                }

                catch (const std::exception& ex) {
                    std::cerr << "[Exception]: Failed to send final TTS: " << ex.what() << '\n';
                }
            }
            // Безопасная очистка всех предыдущих потоков TTS — ВСЕ предыдущие потоки TTS должны быть завершены.
            // Это гарантирует, что при Regenerate/Reset/Exit не будет joinable-потоков в threads.
            // Используем swap + локальный вектор для безопасного join().
            {
                std::vector<std::thread> temp_threads;
                temp_threads.swap(threads); // ← ИСПРАВЛЕНО: swap вместо swape

                // Безопасное ожидание завершения потоков (упрощённый вариант без таймаута)
            for (auto& t : temp_threads) {
                if (t.joinable()) {
                    try {
                        t.join();
                    } catch (const std::exception& e) {
                        std::cerr << "[warn] exception joining thread: " << e.what() << std::endl;
                    } catch (...) {
                        std::cerr << "[warn] unknown exception joining thread\n";
                    }
                }
            }
                // temp_threads уничтожается здесь — все потоки гарантированно завершены или отсоединены
            }
            // Обработка прерывания генерации
            if (llama_interrupted.load() /*&& llama_interrupted_time - llama_start_time < 2.0*/)
            {
                1;  // Пустая операция (заглушка)
                //printf(" \n[continue speech] (%f)", (llama_interrupted_time - llama_start_time));  // Отладочный вывод
            }
            else
            {
                audio.clear();  // Очищаем аудио буфер
                //printf("\n [audio cleared fin]\n");  // Отладочный вывод
            }
            // Вывод статистики времени выполнения
            llama_end_time = get_current_time_ms();
            if (params.verbose)
            {
                // Рассчитываем временные метрики
                llama_time_input = llama_start_generation_time - llama_start_time;
                llama_time_output = llama_end_time - llama_start_generation_time;
                llama_time_total = llama_end_time - llama_start_time;

                // Выводим статистику по контексту и токенам
                printf("\n\n[Context: %d/%d. Tokens: %d in + %d out. Input %.3f s + output %.3f s = total: %.3f s]",
                        n_past, n_ctx, input_tokens_count, new_tokens,
                        llama_time_input, llama_time_output, llama_time_total);

                // Защита от деления на ноль при выводе скорости
                float input_speed = (llama_time_input > 0.001f) ? input_tokens_count / llama_time_input : 0.0f;
                float output_speed = (llama_time_output > 0.001f) ? new_tokens / llama_time_output : 0.0f;
                float total_speed = (llama_time_total > 0.001f) ? new_tokens / llama_time_total : 0.0f;

                printf("\n[Speed: input %.2f t/s + output %.2f t/s = total: %.2f t/s]\n",
                        input_speed, output_speed, total_speed);
            }
            // Сброс флагов и переменных
            llama_interrupted.store(0);         // Сбрасываем флаг прерывания
            llama_interrupted_time = 0.0;       // Сбрасываем время прерывания
            llama_start_generation_time = 0.0;  // Сбрасываем время начала генерации
           {                                    // Сброс под защитой мьютекса
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                g_hotkey_pressed = "";
            }         // Сбрасываем горячую клавишу

            // ===== ПРИГЛАШЕНИЕ ПОСЛЕ ОТВЕТА БОТА =====
            printf("\n");
#ifdef _WIN32
            set_console_color(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
#else
            printf("\033[32m");
#endif
            printf("%s%s ", params.person.c_str(), chat_symb.c_str());
            reset_console_color();
            fflush(stdout);
        }
    }
}

 // Завершение работы - очистка потоков
printf("Cleaning up TTS threads...\n");

// ДОБАВИТЬ: запрещаем создание новых потоков
g_shutting_down.store(true);
std::this_thread::sleep_for(std::chrono::milliseconds(100));

// ШАГ 1: Забираем все потоки из глобального вектора в локальный
std::vector<std::thread> local_threads;
{
    std::lock_guard<std::mutex> lock(g_threads_mutex);
    local_threads.swap(threads);
}

// ШАГ 2: Ждём завершения всех потоков БЕЗ блокировки
printf("Waiting for %zu TTS threads to finish...\n", local_threads.size());

for (auto& t : local_threads) {
    if (t.joinable()) {
        try {
            t.join();
        } catch (const std::exception& e) {
            fprintf(stderr, "Warning: Exception joining thread: %s\n", e.what());
            t.detach();
        } catch (...) {
            fprintf(stderr, "Warning: Unknown exception joining thread\n");
            t.detach();
        }
    }
}

printf("Cleanup complete.\n");

    // ### ЗАВЕРШЕНИЕ РАБОТЫ И ОСВОБОЖДЕНИЕ РЕСУРСОВ ###

    // Приостанавливаем аудио
    audio.pause();

    // ===== ВАЖНО: Сначала выводим статистику, потом освобождаем память =====
    // Выводим метрики Whisper
    whisper_print_timings(ctx_wsp);

    if (ctx_llama) {
        llama_perf_context_print(ctx_llama);
    }

    // ===== Теперь безопасно освобождаем все ресурсы =====
    // Освобождаем контекст Whisper
    whisper_free(ctx_wsp);

    // Освобождаем сэмплеры LLaMA.
    // llama_perf_sampler_print() выводит статистику и сама освобождает сэмплер.
    if (smpl) {
        llama_perf_sampler_print(smpl);
    }
    if (smpl_high_temp) {
        llama_sampler_free(smpl_high_temp);
    }
    // Освобождаем батч LLaMA
    llama_batch_free(batch);

    // Освобождаем контекст LLaMA
    llama_free(ctx_llama);

    // Освобождаем модель LLaMA
    llama_model_free(model_llama);

    // Освобождаем бэкенд LLaMA
    llama_backend_free();

    // Завершаем потоки ввода
    if (input_thread.joinable()) {
        input_thread.detach();
    }

    // Останавливаем поток горячих клавиш
    g_shortcut_thread_running.store(false);
    if (shortcut_thread.joinable()) {
        shortcut_thread.join();
    }

    return 0;
}

// Функция wmain - точка входа для Windows-приложений с поддержкой Unicode
// Преобразует аргументы командной строки из UTF-16 (Windows) в UTF-8 (Linux/Unix)
// и передаёт их в основную функцию run() для кроссплатформенной совместимости
#if _WIN32
int wmain(int argc, const wchar_t ** argv_UTF16LE) {
    console::init(true, true);
    atexit([]() { console::cleanup(); });
    std::vector<std::string> buffer(argc);
    std::vector<char*> argv_UTF8(argc);
    for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = &buffer[i][0];
    }
    return run(argc, argv_UTF8.data());
}
#else
// ### ГЛАВНАЯ ФУНКЦИЯ: main ###
// Инициализирует библиотеку libcurl для сетевых запросов и консоль для работы с Unicode,
// регистрирует функции очистки ресурсов и запускает основную логику программы через run()
int main(int argc, const char ** argv_UTF8) {
    // ИНИЦИАЛИЗАЦИЯ libcurl — ОДИН РАЗ ПРИ СТАРТЕ
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl" << std::endl;
        return 1;
    }
    // ИНИЦИАЛИЗАЦИЯ консоли
    console::init(true, true);
    // ОЧИСТКА ресурсов при выходе
    atexit([]() {
        console::cleanup();         // Очистка консоли
        curl_global_cleanup();      // Очистка libcurl
    });
    // ЗАПУСК основной логики
    return run(argc, argv_UTF8);
}
#endif