// Talk with AI
// ВНЕШНИЕ БИБЛИОТЕКИ ИИ (Whisper и LLaMA)
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

// ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И МЬЮТЕКСЫ
std::atomic<bool> g_is_interrupted{false};
// Эта переменная связывает система прослушивания и система озвучки
std::atomic<int> llama_interrupted{0};  // Изменяем на atomic<int>
std::queue<std::string> input_queue; // глобальная очередь ввода
std::mutex input_mutex; // Мьютекс для защиты доступа к input_queue
std::atomic<bool> keyboard_input_running{true}; // Атомарный флаг для безопасного доступа из разных потоков
std::string g_hotkey_pressed = ""; // Глобальная переменная для отслеживания нажатых горячих клавиш
std::mutex g_hotkey_pressed_mutex; // Мьютекс для защиты g_hotkey_pressed
std::mutex g_tts_mutex; // Мьютекс для защиты массивов TTS
std::mutex g_threads_mutex; // ← ДОБАВЛЯЕМ МЬЮТЕКС ДЛЯ ПОТОКОВ

// ФУНКЦИЯ ТОКЕНИЗАЦИИ ТЕКСТА
// Преобразует текст в последовательность токенов модели LLaMA
static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
const llama_model * model = llama_get_model(ctx);
const llama_vocab * vocab = llama_model_get_vocab(model);

    // Верхняя граница количества токенов (длина текста + BOS токен)
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    // Выполняем токенизацию текста
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) { // Если буфер оказался мал, увеличиваем его
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens); // Проверяем корректность результата
    } else {
        result.resize(n_tokens); // Устанавливаем точный размер
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
        
        // Дополнительная проверка: если не нашли ни одного числа
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
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 64;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;
	float vad_thold  = 0.0005f;
    float vad_start_thold  = 0.0003f;
    float vad_last_ms  = 1500;
    float freq_thold = 90.0f;
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
	std::string person      = "Друг";
    std::string bot_name    = "Эмма";
    std::string xtts_voice  = "Emma";
    std::string wake_cmd    = "";
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
    float top_k = 40;      
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
                params.top_k = std::stof(argv[++i]);
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
    fprintf(stderr, "  --top_k N                  [%-7.2f] top_k \n",                                    params.top_k);
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

// Потокобезопасное добавление задачи в вектор потоков
static void safe_thread_emplace(std::vector<std::thread>& threads_vec, std::function<void()> task) {
     std::scoped_lock lock(g_threads_mutex, g_tts_mutex); // Deadlock-free
    try {
        threads_vec.emplace_back(std::move(task));
    } catch (const std::exception& e) {
        std::cerr << "Ошибка создания потока: " << e.what() << std::endl;
    }
}

// Функция транскрибации аудио с использованием Whisper
static std::string transcribe(
    whisper_context* ctx,               // Контекст Whisper
    const whisper_params& params,       // Параметры транскрибации
    const std::vector<float>& pcmf32,   // Аудиоданные в формате float32
    const std::string& prompt_text,     // Текст промпта (не используется в текущей реализации)
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
    
    // Настройка промпта (в текущей реализации не используется)
    wparams.prompt_tokens = nullptr;
    wparams.prompt_n_tokens = 0;
    
    // Базовые параметры вывода
    wparams.print_progress = false;
    wparams.print_special = params.print_special;
    wparams.print_realtime = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate = params.translate;
    wparams.no_context = true;
    wparams.single_segment = true;

    // Настройка максимального количества токенов с проверкой лимитов модели
    {
        int model_text_ctx = static_cast<int>(whisper_n_text_ctx(ctx));
        int mt = (params.max_tokens > 0 ? params.max_tokens : 64);
        
        if (mt > model_text_ctx) {
            std::cerr << "Предупреждение: max_tokens (" << mt 
                      << ") превышает лимит модели (" << model_text_ctx 
                      << "), применяется лимит модели" << std::endl;
            mt = model_text_ctx;
        }
        wparams.max_tokens = mt;
    }

    // Настройка аудиоконтекста с проверкой лимитов модели
    wparams.audio_ctx = params.audio_ctx;
    int model_audio_ctx = static_cast<int>(whisper_n_audio_ctx(ctx));
    
    if (wparams.audio_ctx > model_audio_ctx) {
        std::cerr << "Предупреждение: audio_ctx (" << wparams.audio_ctx 
                  << ") превышает лимит модели (" << model_audio_ctx 
                  << "), применяется лимит модели" << std::endl;
        wparams.audio_ctx = model_audio_ctx;
    }

    // Настройка языка и потоков
    wparams.language = params.language.empty() ? nullptr : params.language.c_str();
    wparams.n_threads = params.n_threads;

    // Выполнение транскрипции
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Ошибка: Не удалось выполнить транскрипцию аудио" << std::endl;
        // Рассчёт времени выполнения даже при ошибке
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
        result += text;
        
        // Расчёт вероятности для сегмента
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);
            prob += token.p;
            ++prob_n;
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

// Разбивает строку на слова
static std::vector<std::string> get_words(const std::string& txt) {
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
    try {
        // Стандартный способ через C++17 <filesystem>
        return std::filesystem::temp_directory_path().string();
    } catch (const std::exception &e) {
        std::cerr << "[getTempDir] std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[getTempDir] Unknown exception while getting temp dir" << std::endl;
    }

#ifdef _WIN32
    // Fallback: WinAPI
    TCHAR path_buf[MAX_PATH] = {0};
    DWORD ret_val = GetTempPath(MAX_PATH, path_buf);

    if (ret_val == 0 || ret_val > MAX_PATH) {
        std::cerr << "[getTempDir] GetTempPath failed" << std::endl;
        return "";
    }

    #if defined(UNICODE) || defined(_UNICODE)
        try {
            // wide → UTF-8
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.to_bytes(path_buf);
        } catch (const std::exception &e) {
            std::cerr << "[getTempDir] UTF-8 conversion failed: " << e.what() << std::endl;
            return "";
        } catch (...) {
            std::cerr << "[getTempDir] Unknown error during UTF-8 conversion" << std::endl;
            return "";
        }
    #else
        return std::string(path_buf);
    #endif

#else
    // Fallback: POSIX
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
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' && ch != '\f' && ch != '\v' && ch != 0xA0;
    }));
}

// Убирает пробельные символы из конца строки
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' && ch != '\f' && ch != '\v' && ch != 0xA0;
    }).base(), s.end());
}

// Убирает пробельные символы с обеих сторон строки
inline void trim(std::string &s) {
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
                while (base < sanitizedInput.size() && (std::isspace((unsigned char)sanitizedInput[base]) || sanitizedInput[base] == ':' ))
                    ++base;
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

// Удаляет ведущие справа символы Unicode (UTF-8), совпадающие с любым из targetCharacter
std::string RemoveTrailingCharactersUtf8(const std::string& inputString, const std::u32string& targetCharacter) {
    // Преобразуем входную строку из UTF-8 в UTF-32 для корректной работы с символами
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::u32string u32_input = converter.from_bytes(inputString);

    // Ищем первую позицию с конца, где символ не содержится в targetCharacter
    auto lastNonTargetPosition = std::find_if(u32_input.rbegin(), u32_input.rend(), [&targetCharacter](char32_t ch) {
        return targetCharacter.find(ch) == std::u32string::npos;
    });

    // Преобразуем результат обратно в UTF-8 и возвращаем
    std::string result = converter.to_bytes(std::u32string(u32_input.begin(), lastNonTargetPosition.base()));
    return result;
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
    CURLcode res;
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

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
                result += static_cast<char>(c); // оставляем все остальные байты как есть (UTF-8 валиден)
        }
    }
    return result;
};
	// Настройка запроса
    struct curl_slist *headers = nullptr; // Выносим объявление перед try
    bool curl_freed = false; // Флаг, чтобы отследить очистку curl
    
    try {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

        std::ostringstream oss;
        bool firstParam = true;
        oss << "{";
        for (const auto& param : params) {
            if (!firstParam) oss << ",";
            oss << "\"" << escape_json(param.first) << "\":\"" << escape_json(param.second) << "\"";
            firstParam = false;
        }
        oss << "}";
        std::string jsonData = oss.str();
        fprintf(stdout, "send_curl_json: %s\n", jsonData.c_str());
		
		// Устанавливаем заголовки
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		
		// Устанавливаем тело запроса
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
		
		// Устанавливаем callback для получения данных
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Выполняем запрос без лишнего шума в консоли
        CURLcode res = curl_easy_perform(curl);
        
        // Оставляем проверку только для критических случаев в stderr
        if (res != CURLE_OK) {
            // Печатаем только если реально случилась беда, и то — коротко
            fprintf(stderr, " [TTS Error: %s]", curl_easy_strerror(res));
        }

		// Очищаем заголовки
        curl_slist_free_all(headers);
        headers = nullptr;
		
		// Очищаем curl
        curl_easy_cleanup(curl);
        curl_freed = true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        // Очищаем заголовки если они были созданы
        if (headers) {
            curl_slist_free_all(headers);
            headers = nullptr;
        }
        
        // Очищаем curl только если еще не очищен
        if (!curl_freed && curl) {
            curl_easy_cleanup(curl);
            curl_freed = true;
        }
        return "";
    }
    
    // curl уже очищен, ничего делать не нужно
    
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
    size_t i = 0;
    int chars = 0;
    const size_t ix = str.size();

    while (i < ix) {
        unsigned char c = static_cast<unsigned char>(str[i]);

        if (c <= 0x7F) {
            // ASCII
            ++i;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            if (i + 1 >= ix) throw std::runtime_error("Invalid UTF-8 sequence");
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            if (i + 2 >= ix) throw std::runtime_error("Invalid UTF-8 sequence");
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            if (i + 3 >= ix) throw std::runtime_error("Invalid UTF-8 sequence");
            i += 4;
        } else {
            throw std::runtime_error("Invalid UTF-8 sequence");
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
    if (leng == 0) return ""; // Пустая подстрока

    const size_t ix = str.size();
    size_t i = 0;      // индекс в байтах
    unsigned int chars = 0; // индекс в символах (codepoints)
    size_t min_byte_index = std::string::npos;
    size_t max_byte_index = std::string::npos;

    while (i < ix) {
        if (chars == start) min_byte_index = i;
        if (chars == start + leng) { max_byte_index = i; break; }

        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t step = 1;
        if (c <= 0x7F) {
            step = 1;
        } else if ((c & 0xE0) == 0xC0) {
            step = 2;
            if (i + 1 >= ix) return "";
        } else if ((c & 0xF0) == 0xE0) {
            step = 3;
            if (i + 2 >= ix) return "";
        } else if ((c & 0xF8) == 0xF0) {
            step = 4;
            if (i + 3 >= ix) return "";
        } else {
            return ""; 
        }

        i += step;
        ++chars;
    }

    if (max_byte_index == std::string::npos) max_byte_index = ix;
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
    for (const auto& token : embd) {
        std::string token_str = llama_token_to_piece(ctx_llama, token);
        ss += token_str;
    }
    return ss;
}

// Асинхронная функция для отправки текста в TTS (Text-to-Speech) сервис
// Все параметры передаются по значению для безопасности в многопоточном окружении.
// ВСЕ регулярные выражения компилируются ОДИН РАЗ при первом вызове функции.
// Использует оптимизированные regex и безопасную обработку UTF-8.
void send_tts_async(std::string text,
                    std::string speaker_wav = "Emma",
                    std::string language = "ru",
                    std::string tts_url = "http://localhost:8020/",
                    int reply_part = 0) {

    // Быстрая защита: если пусто — сразу выходим
    if (text.empty()) {
        return;
    }

    // 1) Унификация переводов строки и начальная обрезка
    //   - Все \r\n, \r, \n превращаем в пробел
    //   - Это позволяет не учитывать спецсимволы в дальнейших регулярках
    //   - После замены обязательно trim()
    try {
        static const std::regex re_newline(R"(\r\n|\r|\n)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_newline, " ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (newline normalization): %s\n", e.what());
        text = replace(text, "\r\n", " ");
        text = replace(text, "\r", " ");
        text = replace(text, "\n", " ");
    }
    trim(text);
    if (text.empty()) return;

    // Удаление HTML-тегов и базовая нормализация HTML-сущностей
    // ВАЖНО: НЕ удаляем глобально '=' и '/' — это ломает URL и обычный текст.
    try {
        static const std::regex re_html_tag(R"(<[^>]*>)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_html_tag, " ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (HTML removal): %s\n", e.what());
        text = replace(text, "<", " ");
        text = replace(text, ">", " ");
    }

    // Декодируем самые частые HTML-сущности, но очень локально и безопасно.
    text = replace(text, "&nbsp;", " ");
    text = replace(text, "&amp;", "&");
    text = replace(text, "&lt;", "<");
    text = replace(text, "&gt;", ">");
    text = replace(text, "&quot;", "\"");
    text = replace(text, "&#39;", "'");
    text = replace(text, "&apos;", "'");

    // Дополнительные HTML-сущности для лучшей обработки
    text = replace(text, "&#34;", "\"");
    text = replace(text, "&rsquo;", "'");
    text = replace(text, "&lsquo;", "'");
    text = replace(text, "&rdquo;", "\"");
    text = replace(text, "&ldquo;", "\"");
    text = replace(text, "&mdash;", "-");
    text = replace(text, "&ndash;", "-");
    text = replace(text, "&hellip;", "...");

    trim(text);
    if (text.empty()) return;

    // Обработка "умных" кавычек и тире в UTF-8
    // Безопасная замена без сложных проверок
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

    // Markdown: снимаем оформление, сохраняем полезное содержимое
    try {
        static const std::regex re_code_block(R"(```(.*?)```)", std::regex::ECMAScript);
        static const std::regex re_code_inline(R"(`([^`]*)`)", std::regex::ECMAScript);
        static const std::regex re_img_md(R"(!\[[^\]]*\]\([^)\s]+(?:\s+"[^"]*")?\))", std::regex::ECMAScript);
        static const std::regex re_link_md(R"(\[([^\]]*)\]\(([^)\s]+(?:\s+"[^"]*")?)\))", std::regex::ECMAScript);
        static const std::regex re_bold1(R"(\*\*([^*]+)\*\*)", std::regex::ECMAScript);
        static const std::regex re_bold2(R"(__([^_]+)__)", std::regex::ECMAScript);
        static const std::regex re_ital1(R"(\*([^*]+)\*)", std::regex::ECMAScript);
        static const std::regex re_ital2(R"(_([^_]+)_)", std::regex::ECMAScript);
        static const std::regex re_del(R"(~~([^~]+)~~)", std::regex::ECMAScript);
        static const std::regex re_multi_stars(R"(\*{2,})", std::regex::ECMAScript);
        static const std::regex re_multi_unders(R"(_{2,})", std::regex::ECMAScript);
        static const std::regex re_multi_tildes(R"(~{2,})", std::regex::ECMAScript);

        // Блоки и инлайн-код — оставляем содержимое
        text = std::regex_replace(text, re_code_block, "$1");
        text = std::regex_replace(text, re_code_inline, "$1");

        // Изображения — полностью в пробел
        text = std::regex_replace(text, re_img_md, " ");

        // Ссылки — оставляем URL
        text = std::regex_replace(text, re_link_md, "$2");

        // Снимаем жирный/курсив/зачёркнутый
        text = std::regex_replace(text, re_bold1, "$1");
        text = std::regex_replace(text, re_bold2, "$1");
        text = std::regex_replace(text, re_ital1, "$1");
        text = std::regex_replace(text, re_ital2, "$1");
        text = std::regex_replace(text, re_del, "$1");

        // Добиваем висячие маркеры
        text = std::regex_replace(text, re_multi_stars, " ");
        text = std::regex_replace(text, re_multi_unders, " ");
        text = std::regex_replace(text, re_multi_tildes, " ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (Markdown removal): %s\n", e.what());
        text = replace(text, "```", " ");
        text = replace(text, "`", " ");
        text = replace(text, "![", " ");
        text = replace(text, "](", " ");
        text = replace(text, "**", " ");
        text = replace(text, "__", " ");
        text = replace(text, "~~", " ");
    }
    trim(text);
    if (text.empty()) return;

    // Удаление маркеров списков в начале строки
    // Это важно для TTS, чтобы не зачитывались номера пунктов
    try {
        static const std::regex re_list_markers(
            R"(^\s*(\d+[\.\)]|[A-Za-zА-Яа-яЁё][\.\)]|[\-\*\+\>\|#]+)\s*)",
            std::regex::ECMAScript
        );
        text = std::regex_replace(text, re_list_markers, "");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (list markers removal): %s\n", e.what());
        // Простая замена наиболее частых маркеров
        if (text.size() > 2) {
            if (text[0] == '-' || text[0] == '*' || text[0] == '+' || text[0] == '#') {
                if (text[1] == ' ') {
                    text = text.substr(2);
                }
            }
            // Удаление цифр с точкой или скобкой в начале
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

    // Обработка скобок для создания искусственных пауз в речи
    // Содержимое скобок сохраняется, но вместо закрывающей скобки ставится запятая
    // Это создает естественную паузу при озвучивании
    try {
        // Обработка круглых скобок ()
        static const std::regex re_parens(R"(\(([^)]+)\))", std::regex::ECMAScript);
        text = std::regex_replace(text, re_parens, " $1, ");
        
        // Обработка квадратных скобок []
        static const std::regex re_square(R"(\[([^\]]+)\])", std::regex::ECMAScript);
        text = std::regex_replace(text, re_square, " $1, ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (brackets processing): %s\n", e.what());
        // Резервная обработка без запятых для безопасности
        text = replace(text, "(", " ");
        text = replace(text, ")", " ");
        text = replace(text, "[", " ");
        text = replace(text, "]", " ");
    }

    trim(text);
    if (text.empty()) return;

    // Удаление содержимого в {…} с поддержкой вложенности (итеративно)
    // Фигурные скобки полностью удаляются с содержимым
    try {
        static const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
        bool changed = true;
        while (changed) {
            changed = false;
            std::string t1 = std::regex_replace(text, re_curly, " ");
            if (t1 != text) {
                text.swap(t1);
                changed = true;
            }
        }
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (curly braces removal): %s\n", e.what());
        text = replace(text, "{", " ");
        text = replace(text, "}", " ");
    }

    trim(text);
    if (text.empty()) return;

    // Удаление одиночных "мусорных" символов и кавычек
    try {
        static const std::regex re_noise(R"([#\|\\])", std::regex::ECMAScript);
        static const std::regex re_quotes(R"(["'])", std::regex::ECMAScript);
        text = std::regex_replace(text, re_noise, " ");
        text = std::regex_replace(text, re_quotes, " ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (single-char removal): %s\n", e.what());
        text = replace(text, "#", " ");
        text = replace(text, "|", " ");
        text = replace(text, "\\", " ");
        text = replace(text, "\"", " ");
        text = replace(text, "'", " ");
    }
    trim(text);
    if (text.empty()) return;

    // Нормализация пунктуации: схлопываем повторы, многоточия в точку
    // Важно исправить некорректные последовательности для естественного звучания
    try {
        static const std::regex re_commas(R"(,{2,})", std::regex::ECMAScript);
        static const std::regex re_semis(R"(;{2,})", std::regex::ECMAScript);
        static const std::regex re_dashes(R"([\-–—]{2,})", std::regex::ECMAScript);
        static const std::regex re_bangs(R"(!{2,})", std::regex::ECMAScript);
        static const std::regex re_qmarks(R"(\?{2,})", std::regex::ECMAScript);
        static const std::regex re_all_dots(R"(\.{2,})", std::regex::ECMAScript); 
        static const std::regex re_ellipsis_spaces(R"(\s*\.\s*\.\s*\.\s*)", std::regex::ECMAScript);
        static const std::regex re_comma_before_stop(R"(\s*,\s*([.!?]))", std::regex::ECMAScript);
        static const std::regex re_leading_comma(R"(^\s*,\s*)", std::regex::ECMAScript);

        text = std::regex_replace(text, re_commas, ", ");
        text = std::regex_replace(text, re_semis, "; ");
        text = std::regex_replace(text, re_dashes, "- ");
        text = std::regex_replace(text, re_bangs, "!");
        text = std::regex_replace(text, re_qmarks, "? ");
        text = std::regex_replace(text, re_all_dots, ".");
        text = std::regex_replace(text, re_ellipsis_spaces, ".");
        text = std::regex_replace(text, re_comma_before_stop, "$1");
        text = std::regex_replace(text, re_leading_comma, "");

        // Дополнительная очистка: удаляем запятую после точки/вопроса/восклицания
        text = replace(text, ". ,", ". ");
        text = replace(text, "! ,", "! ");
        text = replace(text, "? ,", "? ");
        text = replace(text, "; ,", "; ");
        
        // Убираем двойные запятые, которые могли появиться
        while (text.find(", ,") != std::string::npos) {
            text = replace(text, ", ,", ", ");
        }
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (punctuation normalization): %s\n", e.what());
        while (text.find(",,") != std::string::npos) { text = replace(text, ",,", ","); }
        text = replace(text, ". ,", ". ");
        text = replace(text, "! ,", "! ");
        text = replace(text, "? ,", "? ");
    }
    trim(text);
    if (text.empty()) return;

    // Нормализуем последовательности пробелов до одного
    try {
        static const std::regex re_spaces(R"(\s+)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_spaces, " ");
    } catch (const std::regex_error& e) {
        fprintf(stderr, "Regex error (space normalization): %s\n", e.what());
        text = replace(text, "\t", " ");
        // (Резервный цикл обработки пробелов оставлен для надежности)
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

    // Удаляем префикс вида "Эмма: "
    if (text.find(speaker_wav + ":") == 0) {
        size_t pos = speaker_wav.length() + 1;
        if (pos < text.length() && text[pos] == ' ') pos++;
        text = text.substr(pos);
        trim(text);
    }

    // Финальная нормализация имени спикера
    // Безопасная обработка без сложных проверок, которые могут вызвать ошибки
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
                       "\"speaker_wav\":\"" + escape_json(speaker_wav) + "\", "
                       "\"reply_part\":" + std::to_string(reply_part) + "}";

    // Формируем URL и делаем запрос через cURL
    std::string full_url = tts_url + "tts_to_audio/";
    CURL* http_handle = curl_easy_init();
    if (http_handle) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        // Настройка таймаутов для стабильности соединения
        curl_easy_setopt(http_handle, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(http_handle, CURLOPT_CONNECTTIMEOUT, 10L);
        
        curl_easy_setopt(http_handle, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(http_handle, CURLOPT_URL, full_url.c_str());
        curl_easy_setopt(http_handle, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);

        std::string responseData;
        curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);
        curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);

        curl_easy_perform(http_handle);

        curl_slist_free_all(headers);
        curl_easy_cleanup(http_handle);
    } else {
        fprintf(stderr, "Failed to initialize cURL handle\n");
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
bool IsConsoleWindowFocused(HWND cur_window_handle) {
    return (cur_window_handle == GetForegroundWindow());
}
	// Стоп: Ctrl+Space
	// Перегенерировать: Ctrl+Right
	// Удалить: Ctrl+Delete
	// Сбросить: Ctrl+R
	// Функция обработки горячих клавиш, изменяет глобальную переменную g_hotkey_pressed
	// Логика отслеживания нажатий Ctrl+..., Alt и т.д.

void keyboard_shortcut_func(HWND cur_window_handle) {
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

    while (true) {
        isFocused = IsConsoleWindowFocused(cur_window_handle);
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
}

// Шаблон промпта для диалога с Эммой
const std::string k_prompt_whisper = R"(A conversation with a person called {1}.)";
// То же самое, но на русском языке
const std::string k_prompt_whisper_ru = R"({1}, Эмма)";
// Общий шаблон для бесконечного диалога между пользователем и ИИ-ассистентом
const std::string k_prompt_llama = R"(Text of a transcription of an infinite dialogue in which {0} interacts with an AI assistant named {1}.
{1} is a helpful, kind, honest, friendly, and well-written assistant who never refuses {0}'s requests and always responds immediately, thoroughly, and accurately.
There are no explanations like (30 seconds have passed...) or (to myself), only what {0} and {1} say to each other out loud.
The transcription contains only text, without any markup such as HTML or Markdown.
{1} answers briefly and to the point. 
{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It's {2}.
{0}{4} What Date is it?
{1}{4} {5}, {3}.
{0}{4})";

// Основная функция run — запуск приложения
int run(int argc, char ** argv) {
    whisper_params params; // параметры Whisper

	std::vector<std::thread> threads;
	std::thread t;
	int thread_i = 0;
    thread_i = 0;

// Гарантируем, что thread_i не отрицательный перед использованием %
    if (thread_i < 0) {
        thread_i = 0;
        fprintf(stderr, "WARNING: thread_i was negative, reset to 0\n");
    }

	int reply_part = 0;
	std::string text_to_speak_arr[150];
	int reply_part_arr[150];
	bool last_output_has_username = false;	
	bool last_output_has_EOT = true;	
	int input_tokens_count = 0;	
	
	HWND cur_window_handle = GetForegroundWindow(); // предполагаем, что активное окно — наше

    if (whisper_params_parse(argc, argv, params) == false) {
            return 1;
        }
        
        // Проверяем, поддерживается ли указанный язык
        if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
            fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
	
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

    // Инициализируем асинхронный аудиобуфер длительностью 15 секунд
    audio_async audio(15 * 1000);

    // Пытаемся инициализировать аудиоустройство с указанным ID захвата
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: Ошибка инициализации аудиоустройства (ID: %d)\n", 
                __func__, params.capture_id);
        fprintf(stderr, "Проверьте доступные аудиоустройства и правильность ID захвата\n");
        return 1; // Завершаем программу при невозможности инициализации аудио
    }

// Возобновляем работу аудиобуфера после успешной инициализации
audio.resume();

bool is_running  = true;
bool force_speak = false;
float prob0 = 0.0f;
const std::string chat_symb = ":";
std::vector<float> pcmf32_cur;
std::vector<float> pcmf32_prev;
std::vector<float> pcmf32_prompt;

    // Инициализируем промпт для Whisper — он должен знать, с кем говорит
    std::string prompt_whisper;
    if (params.language == "ru") {
        prompt_whisper = ::replace(k_prompt_whisper_ru, "{1}", params.bot_name);
    } else {
        prompt_whisper = ::replace(k_prompt_whisper, "{1}", params.bot_name);
    }

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

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // Получаем текущее время
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }
    {
        // Получаем текущий год
        std::string year_str;
		std::string ymd;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
			strftime(buf, sizeof(buf), "%Y-%m-%d", now);
            ymd = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
		prompt_llama = ::replace(prompt_llama, "{5}", ymd);
        }
    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

    llama_batch batch = llama_batch_init(2048, 0, 1); // <-- ВСЕГДА ИНИЦИАЛИЗИРУЕМ С n_tokens=0!
    fprintf(stdout, "llama_n_ctx %d", llama_n_ctx(ctx_llama));

    // Инициализация сэмплера
	const float top_k          = params.top_k;
	const float top_p          = params.top_p;
	const float min_p          = params.min_p;
	float temp                 = params.temp;                       
	const float repeat_penalty = params.repeat_penalty;						
    const int seed = 0;    
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler * smpl_high_temp = llama_sampler_chain_init(sparams);

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

    // --- поддержка ChatML при instruct_preset=ChatML ---
    if (params.instruct_preset == "ChatML") {
        // Формируем корректный ChatML формат, если он не применён ранее
        std::string chatml_prompt;

        // Добавляем system prompt, если он задан
        if (!params.prompt.empty()) {
            chatml_prompt += "<|im_start|>system\n" + params.prompt + "<|im_end|>\n";
        }

        // Добавляем user сообщение
        chatml_prompt += "<|im_start|>user\n" + prompt_llama + "<|im_end|>\n";

        // Добавляем начало блока assistant — модель продолжит отсюда
        chatml_prompt += "<|im_start|>assistant\n";

        // Заменяем оригинальный промпт на ChatML
        prompt_llama = chatml_prompt;
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
    const int wake_cmd_length = get_words(wake_cmd).size();
    const bool use_wake_cmd = wake_cmd_length > 0;
    if (use_wake_cmd) {
        printf("%s : the wake-up command is: '%s%s%s'\n", __func__, "\033[1m", wake_cmd.c_str(), "\033[0m");
    }
    printf("\n");
    printf("%s%s ", params.person.c_str(), chat_symb.c_str());
    fflush(stdout);

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

    srand(time(NULL)); // Инициализируем генератор случайных чисел
	int last_command_time = 0;
	int eot_antiprompt_id_1 = 0;
	int eot_antiprompt_id_2 = 0;
	std::string current_voice = params.xtts_voice;

// === УЛУЧШЕННЫЕ АНТИПРОМПТЫ (без проблемного \n) ===
std::vector<std::string> antiprompts = {
    params.person + chat_symb,      // "Друг:"
    params.person + " " + chat_symb, // "Друг :"
};

// Добавляем варианты с двоеточием для надежности (как в Mozer)
antiprompts.push_back(params.person + ":");
antiprompts.push_back(params.person + " :");

// Стоп-последовательность из инструкций
if (!params.instruct_preset_data["stop_sequence"].empty()) {
    antiprompts.push_back(params.instruct_preset_data["stop_sequence"]);
}

// Суффикс сообщения бота
if (!params.instruct_preset_data["bot_message_suffix"].empty())  
{
    antiprompts.push_back(params.instruct_preset_data["bot_message_suffix"]);
    eot_antiprompt_id_1 = antiprompts.size() - 1;
    
    // Страховка от странных тегов
    antiprompts.push_back("</end_of_turn>");
    eot_antiprompt_id_2 = antiprompts.size() - 1;
}

// Пользовательские стоп-слова (с фильтром коротких слов)
if (!params.stop_words.empty())
{
    size_t startIndex = 0;
    size_t endIndex = params.stop_words.find(';');
    
    if (endIndex == std::string::npos) {
        // Одно слово
        std::string word = params.stop_words;
        if (word.length() >= 2) {  // ← ФИЛЬТР: игнорируем слова короче 2 символов
            word = ::replace(word, "\\r", "\r");
            word = ::replace(word, "\\n", "\n");
            antiprompts.push_back(word);
        }
    } else {
        // Несколько слов через ";"
        while (startIndex < params.stop_words.size())
        {
            std::string word = params.stop_words.substr(startIndex, endIndex - startIndex);
            if (!word.empty())
            {
                // ← ФИЛЬТР: игнорируем слова короче 2 символов
                if (word.length() >= 2) {
                    word = ::replace(word, "\\r", "\r");
                    word = ::replace(word, "\\n", "\n");
                    antiprompts.push_back(word);
                }
            }
            startIndex = endIndex + 1;
            endIndex = params.stop_words.find(';', startIndex);
            if (endIndex == std::string::npos) {
                endIndex = params.stop_words.size();
            }
        }
    }
}

// Выводим итоговый список стоп-слов для отладки
printf("Llama stop words (%zu): ", antiprompts.size());
for (const auto &prompt : antiprompts) {
    if (prompt.length() < 10) {
        printf("'%s', ", prompt.c_str());
    } else {
        printf("'%.7s...', ", prompt.c_str());
    }
}
printf("\n");

	std::thread input_thread(input_thread_func);
	std::thread shortcut_thread([cur_window_handle]() {
        keyboard_shortcut_func(cur_window_handle);
    });
    
	printf("\nVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right), Delete(Ctrl+Delete), Reset(Ctrl+R)\n");
	if (params.push_to_talk) printf("Type anything or hold 'Alt' to speak:\n");
	else printf("Start speaking or typing:\n");
	printf("\n\n");
    printf("%s%s ", params.person.c_str(), chat_symb.c_str());
    fflush(stdout);

	int vad_result_prev = 2; // ended
	float speech_start_ms = 0;
	float speech_end_ms = 0;
	float speech_len = 0;
	int len_in_samples = 0;
    int64_t speech_start_sample = 0; // сохраняем номер сэмпла, когда началась речь
	std::string all_heard_pre;
	float llama_interrupted_time = 0.0;	
	llama_start_time = 0.0;
	float llama_start_generation_time = 0.0; // после оперативной обработки
	llama_end_time = 0.0;
	llama_time_total = 0.0;
    std::string user_typed = "";
    bool user_typed_this = false;
    // === Для Патча 3: буфер накопления ввода ===
static std::string input_accumulator = "";      // буфер накопления фраз
static float last_input_time = 0.0f;            // время последнего ввода (секунды)
const float INPUT_TIMEOUT_MS = 1.5f;            // 1.5 сек тишины = конец фразы

// ### ОСНОВНОЙ ЦИКЛ РАБОТЫ ПРИЛОЖЕНИЯ ###
    while (is_running) {
        // СБРОС СОСТОЯНИЯ ПРЕРЫВАНИЯ
        g_is_interrupted.store(false);
        llama_interrupted.store(0);
        // handle Ctrl + C
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }
        // задержка. попробуйте опустить?
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int64_t t_ms = 0;
            // === FIX: Объявляем text_heard раньше, чтобы он был виден в keyboard input ===
    static std::string text_heard = "";  // static — сохраняет значение между итерациями
		
        // === ПАТЧ 3: Буфер накопления ввода пользователя ===
// === ВАЖНО: Перед этим патчем, в начале функции run(), добавь эти переменные: ===
// static std::string input_accumulator = "";      // буфер накопления
// static float last_input_time = 0.0f;            // время последнего ввода
// const float INPUT_TIMEOUT_MS = 1.5f;            // 1.5 сек тишины = конец фразы

// keyboard input
user_typed_this = false;
console::set_display(console::reset);

// === ФИКС: Накопление ввода вместо мгновенной отправки ===
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
            if (!hk_copy.empty() && hk_copy != "Alt") { // ← Проверка и сброс в ОДНОЙ критической секции
                g_hotkey_pressed = ""; // ← Сбрасываем ТОЛЬКО если мы обработали событие
            }
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
                // СБРОС УЖЕ ВЫПОЛНЕН В КРИТИЧЕСКОЙ СЕКЦИИ ВЫШЕ. НЕ НУЖНО ПОВТОРНО БЛОКИРОВАТЬ!
            }
        }
        {
            // Получаем аудио из буфера длительностью step_ms (2000 мс), async — асинхронно
            audio.get(2000, pcmf32_cur); // step_ms, async
			// WHISPER_SAMPLE_RATE — частота дискретизации аудио для Whisper (16 кГц)
            // vad_last_ms — минимальная длина речевого сегмента для VAD (по умолчанию 1250 мс)
            // Вызываем VAD (Voice Activity Detection) для определения наличия речи в аудиосигнале
            // vad_simple_int возвращает:
            // 0 — тишина, 1 — начало речи, 2 — конец речи
            int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, 
                                            params.vad_thold, params.freq_thold, params.print_energy, 
                                            params.vad_start_thold);			

            // Если VAD обнаружил начало речи (vad_result == 1) и это новое начало (предыдущее не было началом)
            if (vad_result == 1 && params.vad_start_thold) // speech started
                {
                if (vad_result_prev != 1) // реальное начало речи
                    {					
                    // Запоминаем время начала речи
                    speech_start_ms = get_current_time_ms(); // float
                    // Сохраняем аудиофрагмент, который VAD только что проанализировал

            // Обновляем статус VAD
			vad_result_prev = 1;
					
            // Выполняем "прогревку" Whisper — небольшое распознавание для инициализации (это не основное распознавание)
            if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
                {
                    all_heard_pre = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms)); // warmup - try with small size audio
                    { // Сброс под защитой мьютекса
                        std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                        g_hotkey_pressed = "";
                    }

                }
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

                        // ⭐ ДОБАВЛЕНО: устанавливаем флаги прерывания
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
                //printf("%.3f found vad length: %.2f\n", get_current_time_ms(), speech_len);
                vad_result_prev = 2;
                // Сбрасываем время начала речи
                speech_start_ms = 0;
                // Пропускаем обработку, если длина речи нулевая и нет введённого пользователем текста
                if (!speech_len && !user_typed.size()) continue;
                // Добавляем небольшую "подушку" перед началом речи
                speech_len = speech_len + 0.3; // front padding
                // Устанавливаем минимальную длину речи (Whisper работает лучше с фразами дольше 1.1 секунды)
                if (speech_len < 1.10) speech_len = 1.10;
                // берём последние 10 сек из аудиобуфера целиком.
                // Это гарантирует захват ВСЕГО, что уместилось в последние 10 сек, включая начало фразы,
                // даже если VAD сработал с задержкой.
                audio.get(10000, pcmf32_cur); // Получаем последние 10000 мс (10 сек) аудио
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
                        if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
                        {
                            all_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper, prob0, t_ms)); // real transcribe
                            { // Сброс под защитой мьютекса
                                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                                g_hotkey_pressed = "";
                            }
                        }
                    }
                // логика обработки all_heard ---
                // Разделяем распознанный текст на команду пробуждения и основной текст
                const auto words = get_words(all_heard);
                std::string wake_cmd_heard;
                std::string text_heard;
                // Первые wake_cmd_length слов — это команда пробуждения
                for (int i = 0; i < (int) words.size(); ++i) {
                    if (i < wake_cmd_length) {
                        wake_cmd_heard += words[i] + " ";
                    } else {
                        text_heard += words[i] + " ";
                    }
                }

                // Выводим уровень энергии, если включён (для отладки)
                if (params.print_energy) fprintf(stdout, " [text_heard: (%s)]\n", text_heard.c_str());
                // Если используется команда пробуждения — проверяем её сходство с эталонной
                if (use_wake_cmd) {
                    const float sim = similarity(wake_cmd_heard, wake_cmd);
                    // Если сходство слишком низкое или текст пуст — игнорируем и очищаем аудиобуфер
                    if ((sim < 0.7f) || (text_heard.empty())) {
                        audio.clear();
                        continue;
                    }
                }

                // при необходимости дайте звуковую обратную связь о том, что текущий текст обрабатывается
                if (!params.heard_ok.empty()) {
                    speak_with_file(params.speak, params.heard_ok, params.speak_file, voice_id);
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
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"!");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U",");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U".");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"»");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"[");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"]");
				text_heard = RemoveTrailingCharactersUtf8(text_heard, U"\""); // удаление конечной кавычки
				if (text_heard[0] == '.') text_heard.erase(0, 1);
				if (text_heard[0] == '!') text_heard.erase(0, 1);
				if (text_heard[0] == '[') text_heard.erase(0, 1);
				trim(text_heard);
				
                // смягчённая фильтрация распознанного текста
                // Удаляем очевидный шум / стандартные завершающие фразы, НО НЕ удаляем одиночные символы и '*'.
                // Это даёт модели шанс среагировать на короткие фразы и односложные слова.
                if (
                    text_heard == "!" || text_heard == "." ||
                    text_heard == "Sil" || text_heard == "Bye" || text_heard == "Okay" || text_heard == "Okay." ||
                    text_heard == "Thank you." || text_heard == "Thank you" || text_heard == "Thanks." || text_heard == "Bye." ||
                    text_heard == "Thank you for listening." || text_heard == "Спасибо" || text_heard == "Пока" ||
                    text_heard == params.bot_name || text_heard == "*Звук!*" ||
                    text_heard.find("Редактор субтитров") != std::string::npos ||
                    text_heard.find("Спасибо за внимание") != std::string::npos ||
                    text_heard.find("Продолжение следует") != std::string::npos ||
                    text_heard.find("End of") != std::string::npos ||
                    text_heard.find("The End") != std::string::npos ||
                    text_heard.find("Translated by") != std::string::npos ||
                    text_heard.find("Thanks for watching") != std::string::npos ||
                    text_heard.find("Thank you for watching") != std::string::npos ||
                    text_heard.find("*click*") != std::string::npos ||
                    text_heard.find("Субтитры") != std::string::npos ||
                    text_heard.find("До свидания") != std::string::npos ||
                    text_heard.find("До новых встреч") != std::string::npos ||
                    text_heard.find("ПЕСНЯ") != std::string::npos ||
                    text_heard.find("Silence") != std::string::npos
                ) {
                    // оставляем это как «шум» и очищаем
                    text_heard = "";
                } else {
                    // Не удаляем короткие или односимвольные распознавания: даём модели шанс ответить.
                    // Небольшие дополнения: нормализуем пробельные символы, удаляем только длинный «мусор»
                    // (остальные случаи позволим дальше обрабатываться)
                }

				text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), ""); // trailing whitespace
				text_heard_trimmed = text_heard; // no periods or spaces
                trim(text_heard_trimmed);
				if (text_heard_trimmed[0] == '.') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[0] == '!') text_heard_trimmed.erase(0, 1);
				if (text_heard_trimmed[text_heard_trimmed.length() - 1] == '.' || text_heard_trimmed[text_heard_trimmed.length() - 1] == '!') text_heard_trimmed.erase(text_heard_trimmed.length() - 1, 1);
				trim(text_heard_trimmed);
				text_heard_trimmed = LowerCase(text_heard_trimmed); // not working right with utf and russian
                fflush(stdout);

                std::string user_command; // здесь будет храниться распознанная команда пользователя
				if (params.vad_start_thold)
                    {
                    // Пользователь закончил говорить, разрешаем воспроизведение через XTTS
                        allow_xtts_file(params.xtts_control_path, 1);
                    }
				
				// ВВОДНОЕ предложение TTS rand для мгновенного ответа
                if (params.xtts_intro)
                {
                    if (text_heard_trimmed.size())
                        {
                            rand_intro_text = tts_intros[rand() % tts_intros.size()];
                            // Сначала работаем с общими массивами под защитой мьютекса
                            std::string intro_for_tts;
                                {
                                    std::lock_guard<std::mutex> lock(g_tts_mutex);
                                    int idx = thread_i;
                                    if (idx < 0) idx = 0;
                                    if (idx >= 150) idx = 149;
                                    text_to_speak_arr[idx] = rand_intro_text;
                                    thread_i = (idx + 1) % 150;
                                    intro_for_tts = rand_intro_text; // Копируем ДЛЯ потока под защитой
                                }

                            // Теперь безопасно запускаем поток с локальной копией
                            if (!intro_for_tts.empty()) {
                                // Чистим завершенные потоки, чтобы вектор не рос бесконечно
                                for (auto it = threads.begin(); it != threads.end(); ) {
                                    if (it->joinable()) {
                                        it->detach(); // Отсоединяем, так как они сами завершатся при прерывании curl
                                        it = threads.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                                    threads.emplace_back([intro_for_tts, current_voice, params]() {
                                    send_tts_async(intro_for_tts, current_voice, params.language, params.xtts_url);
                                });
                            }
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
                                if (!past_prev_arr.empty()) {
                                    n_past_prev = past_prev_arr.back();
                                    past_prev_arr.pop_back();
                                } else {
                                    // безопасный fallback
                                    n_past_prev = 0;
                                }

                                int rollback_num = embd_inp.size()-n_past_prev;

                                if (rollback_num)
                                    {						
                                    // Удаляем последние токены из контекста
                                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                                    printf(" [regenerating %I32d tokens. Context: %zu]\n", rollback_num, embd_inp.size());
                                    n_past = embd_inp.size();
                                    n_session_consumed = n_past;
                                    // Удаляем последовательность 0 из KV-кэша (новый API)
                                    // Диапазон [embd_inp.size(), end)
                                    llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);
                                    // Восстанавливаем предыдущий запрос
                                    text_heard = text_heard_prev;
                                    text_heard_trimmed = "";								
                                    if (thread_i >= 0 && thread_i < 150) {
                                        text_to_speak_arr[thread_i] = "Regenerating";
                                        } else {
                                            thread_i = 0; // reset на всякий случай
                                            text_to_speak_arr[0] = "Regenerating";
                                    }								

                                    // Безопасное получение текста для повторного озвучивания
                                    std::string text_to_respeak_safe; // Локальная переменная вне мьютекса

                                    {
                                        // Все операции с общими массивами внутри одного мьютекса
                                        std::lock_guard<std::mutex> lock(g_tts_mutex);
                                        int prev_idx = (thread_i - 1 + 150) % 150;
                                        if (prev_idx < 0) prev_idx = 0;
                                        if (prev_idx >= 150) prev_idx = 149;
                                        text_to_respeak_safe = text_to_speak_arr[prev_idx];
                                        text_to_speak_arr[prev_idx] = ""; // Очищаем под защитой
                                    }

                                    //Теперь безопасно запускаем поток с локальной копией
                                    if (!text_to_respeak_safe.empty()) {
                                        threads.emplace_back([text_to_respeak_safe, current_voice, params]() {
                                            send_tts_async(text_to_respeak_safe, current_voice, params.language, params.xtts_url);
                                        });
                                    }
                                    thread_i = (thread_i + 1) % 150;
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
											
                            // Асинхронное воспроизведение "Deleted" через TTS
                            // Сохраняем текст "Deleted" в массив под защитой мьютекса
                            std::string text_for_deleted_tts = "Deleted";
                            std::string text_to_respeak_safe;
                            int previous_index = 0; // ← Объявляем ДО блока
                            {
                                std::lock_guard<std::mutex> lock(g_tts_mutex);
                                
                                // Сохраняем новый текст
                                text_to_speak_arr[thread_i] = text_for_deleted_tts;
                                
                                // Безопасно читаем предыдущее значение
                                previous_index = (thread_i - 1 + 150) % 150; // Используем previous_index вместо idx
                                text_to_respeak_safe = text_to_speak_arr[previous_index];
                                text_to_speak_arr[previous_index] = ""; // Очищаем под защитой
                                
                                thread_i = (thread_i + 1) % 150;
                            }
                            // Теперь можно использовать previous_index
                            if (!text_to_respeak_safe.empty()) {
                                threads.emplace_back([text_to_respeak_safe, current_voice, params]() {
                                    send_tts_async(text_to_respeak_safe, current_voice, params.language, params.xtts_url);
                                });
                            }

                            // Запускаем поток с локальными копиями
                            if (!text_to_respeak_safe.empty()) {
                                threads.emplace_back([text_to_respeak_safe, current_voice, params]() {
                                    send_tts_async(text_to_respeak_safe, current_voice, params.language, params.xtts_url);
                                });
                            }					
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
            // Удаляем всё, кроме начального промпта
            n_past_prev = past_prev_arr.front();
            past_prev_arr.clear();
            int rollback_num = embd_inp.size()-n_past_prev;
                if (rollback_num)
                            {
                                printf(" [Resetting context of %zd tokens.]\n", embd_inp.size());
                                // Обязательно освобождаем старый контекст перед созданием нового!
                                if (ctx_llama) {
                                    llama_free(ctx_llama);
                                }
                                // Пересоздаём контекст модели
                                ctx_llama = llama_init_from_model(model_llama, lcparams);

                // Токенизируем начальный промпт заново
                embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);
                 {										
                    batch.n_tokens = embd_inp.size();

                    for (int i = 0; i < batch.n_tokens; i++) {
                        batch.token[i]     = embd_inp[i];
                        batch.pos[i]       = i;
                        batch.n_seq_id[i]  = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i]    = i == batch.n_tokens - 1;
                    }
                }

                // Выполняем оценку начального промпта
                if (llama_decode(ctx_llama, batch)) {
                    fprintf(stderr, "%s : failed to decode\n", __func__);
                    return 1;
                }

                n_past = embd_inp.size();
                n_session_consumed = embd_inp.size();
                printf(" [Context is now %zu/%I32d tokens. n_past: %d]\n", embd_inp.size(), params.ctx_size, n_past);

                // Сбрасываем переменные
                text_heard = "";
                text_heard_trimmed = "";
                send_tts_async("Reset whole context", params.xtts_voice, params.language, params.xtts_url);
                new_command_allowed = 0;
            }
        }
        else 
            {
                // Если сбрасывать нечего — сообщаем об этом
                printf(" [Nothing to reset more]\n");			
                send_tts_async("Nothing to reset more", params.xtts_voice, params.language, params.xtts_url);
            }
    }
    audio.clear(); // Очищаем аудио-буфер
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
    //  Удобная обёртка для безопасного озвучивания коротких фраз (ПАТЧ: с защитой мьютексом)
        auto speak_safe = [&](const std::string& msg) {
            if (msg.empty()) return;
            // ЗАЩИЩАЕМ все операции с общими данными
            std::string msg_copy = msg;
            int current_reply_part;
            int current_idx;
            {
                std::lock_guard<std::mutex> lock(g_tts_mutex);
                // Безопасные операции под защитой мьютекса
                current_reply_part = reply_part++;
                current_idx = thread_i % 150; // Защита от выхода за границы
                // Сохраняем в массивы для совместимости с остальным кодом
                text_to_speak_arr[current_idx] = msg_copy;
                reply_part_arr[current_idx] = current_reply_part;
                // Обновляем индекс с циклическим буфером
                thread_i = (thread_i + 1) % 150;
            }
            
            try {
                threads.emplace_back([msg_copy, current_voice, params, current_reply_part]() {
                    send_tts_async(msg_copy, current_voice, params.language, params.xtts_url, current_reply_part);
                });
            } catch (const std::exception& e) {
                fprintf(stderr, "[google] TTS thread spawn failed: %s\n", e.what());
                // Откатываем изменения при ошибке (под защитой мьютекса)
                {
                    std::lock_guard<std::mutex> lock(g_tts_mutex);
                    text_to_speak_arr[current_idx] = "";
                    reply_part_arr[current_idx] = 0;
                    // Внимание: не откатываем thread_i и reply_part для простоты
                }
            }
        };
    // Достаём ключевые слова
    std::string q = ParseCommandAndGetKeyword(text_heard_trimmed, user_command);
    if (q.empty()) {
        fprintf(stdout, "[google] can't get keyword from: %s\n", text_heard_trimmed.c_str());
        speak_safe("Извините, не удалось понять, что именно вы хотите найти.");
        // ВАЖНО: не выходим из цикла генерации LLM жестким continue;
        // просто очищаем пользовательский ввод и дадим модели ответить дальше как обычно
        user_typed.clear();
        user_typed_this = false;
    } else {
            // Безопасная аудио-квитанция
            {
                std::string google_search_msg = "Ищу информацию по запросу: " + q;
                std::lock_guard<std::mutex> lock(g_tts_mutex);
                int idx = thread_i;
                if (idx < 0) idx = 0;
                if (idx >= 150) idx = 149;
                text_to_speak_arr[idx] = google_search_msg;
                thread_i = (idx + 1) % 150;
                // Запускаем TTS с локальной копией
                threads.emplace_back([google_search_msg, current_voice, params]() {
                    send_tts_async(google_search_msg, current_voice, params.language, params.xtts_url);
                });
            }

// Запрос к поисковому серверу
const std::string url = params.google_url + "google?q=" + UrlEncode(q);
std::string resp = send_curl(url);
    if (resp.empty()) {
        fprintf(stdout, "[google] empty response for (%s) — check backend\n", q.c_str());
        // Безопасное сообщение об ошибке
        {
            std::string error_msg = "Извините, не удалось найти информацию по запросу: " + q;
            std::lock_guard<std::mutex> lock(g_tts_mutex);
            int idx = thread_i;
            if (idx < 0) idx = 0;
            if (idx >= 150) idx = 149;
            text_to_speak_arr[idx] = error_msg;
            thread_i = (idx + 1) % 150;
            threads.emplace_back([error_msg, current_voice, params]() {
                send_tts_async(error_msg, current_voice, params.language, params.xtts_url);
            });
        }
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
                params.bot_name = q;
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
    int64_t speech_vad_start_ms = 0;
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
    
    if (last_output_has_username && !user_typed_this) // last model output has user name
        {
            text_heard.insert(0, 1, ' '); 
            text_heard_with_instruct.insert(0, 1, ' '); 
        }
            else if (!last_output_has_EOT) // no EOT
            {
                text_heard.insert(0, "\n"+params.person + chat_symb + " ");
                text_heard_with_instruct.insert(0, params.instruct_preset_data["bot_message_suffix"] +"\n"+ params.instruct_preset_data["user_message_prefix"]+"\n"+params.person + chat_symb + " ");
            }
                else // has EOT or no_instuct
                {
                    text_heard.insert(0, "\n"+params.person + chat_symb + " ");
                    text_heard_with_instruct.insert(0, "\n"+params.instruct_preset_data["user_message_prefix"]+"\n"+params.person + chat_symb + " ");
                }
    text_heard += "\n" + params.bot_name + chat_symb;
    text_heard_with_instruct += params.instruct_preset_data["user_message_suffix"]+"\n" + params.instruct_preset_data["bot_message_prefix"]+ "\n" + params.bot_name + chat_symb;

    if (user_typed_this) 
    {
        fprintf(stdout, "%s%s%s", "\033[1m", (params.bot_name + chat_symb).c_str(), "\033[0m");
        { 
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            g_hotkey_pressed = "";
        }
    }
    else fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");

    if (params.instruct_preset.size()) text_heard = text_heard_with_instruct; 
    fflush(stdout);
    int split_after = params.split_after;

    // ЕДИНСТВЕННАЯ ТОКЕНИЗАЦИЯ: сразу в embd
    embd = ::llama_tokenize(ctx_llama, text_heard, false); 
    input_tokens_count = embd.size();

    // Append the new input tokens to the session_tokens vector
    if (!path_session.empty()) {
        // Используем embd (актуальные токены), а не удаленный вектор tokens
        session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
    }

    // только защита от переполнения индекса
    if (thread_i >= 150) {
        thread_i = 0; // Мягкая ротация — не ломает логику, не вызывает join(), не мешает потокам
    }
    
    float temp_next = params.temp;
    int n_discard = 0;
    int n_left = 0;
    // text inference
    bool done = false;
    std::string text_to_speak;
    int new_tokens = 0;

    while (true) {
    // predict
	if (new_tokens > params.n_predict) break; // 64 default
		new_tokens++;
            if (embd.size() > 0) {
    		    if (n_past + (int) embd.size() > n_ctx) {
             
// === НОВЫЙ ПАТЧ 1: ОПТИМИЗИРОВАННАЯ РОТАЦИЯ КОНТЕКСТА ===
// Используем штатные методы llama.cpp для сдвига контекста,
// что должно быть быстрее и надежнее ручного управления токенами.
if (n_past + (int)embd.size() > n_ctx) {
    // Получаем вокабуляр для проверки BOS токена
    const llama_vocab * vocab_llama = llama_model_get_vocab(model_llama);
    
    // Безопасное вычисление n_left и n_discard с защитой от отрицательных значений
    const int n_left = std::max(0, n_past - n_keep);
    
    // Вычисляем, сколько токенов нужно выбросить. Используем ту же логику, что и была (n_left / 4),
    // но добавляем защиту от отрицательных значений и минимум в 1 токен.
    int n_discard = 0;
    if (n_left > 0) {
        n_discard = std::max(1, n_left / 4);
        // Убеждаемся, что n_discard не больше, чем n_left
        n_discard = std::min(n_discard, n_left);
    }
    
    // Флаг для отслеживания, был ли выполнен сдвиг
    bool context_updated = false;
    
    // Дополнительная защита: проверяем валидность диапазона для удаления
    if (n_discard > 0 && n_keep + n_discard <= n_past) {
        // Проверяем границы embd_inp для безопасности
        if (n_keep >= 0 && n_keep + n_discard <= (int)embd_inp.size()) {
            
            // 1. Удаляем диапазон [n_keep, n_keep + n_discard) из KV-кэша для последовательности 0.
            //    Используем llama_memory_seq_rm (как в вашем оригинальном коде)
            llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, n_keep, n_keep + n_discard);
            
            // 2. Сдвигаем оставшиеся токены (с n_keep + n_discard до n_past) влево на n_discard позиций.
            if (n_keep + n_discard < n_past) {
                llama_memory_seq_add(llama_get_memory(ctx_llama), 0, n_keep + n_discard, n_past, -n_discard);
            }
            
            // 3. Обновляем наш локальный буфер токенов, просто удаляя соответствующий диапазон.
            //    Это значительно быстрее, чем вручную собирать embd из обрезков.
            embd_inp.erase(embd_inp.begin() + n_keep, embd_inp.begin() + n_keep + n_discard);
            
            // 4. Синхронизируем session_tokens, если используется сессия.
            if (!path_session.empty() && !session_tokens.empty()) {
                size_t session_erase_start = std::min((size_t)n_keep, session_tokens.size());
                size_t session_erase_end = std::min((size_t)(n_keep + n_discard), session_tokens.size());
                if (session_erase_start < session_erase_end) {
                    session_tokens.erase(session_tokens.begin() + session_erase_start,
                                         session_tokens.begin() + session_erase_end);
                }
            }
            
            context_updated = true;
            printf(" [Context shifted: discarded %d tokens. New context size: %zu.]", 
                   n_discard, embd_inp.size());
        }
    }
    
    // Если сдвиг не удался (например, из-за невалидных параметров), используем безопасный fallback
    if (!context_updated) {
        printf(" [Context shift fallback - keeping last %d tokens.]", n_keep);
        
        // Просто оставляем последние n_keep токенов
        size_t new_size = std::min((size_t)std::max(0, n_keep), embd_inp.size());
        if (new_size < embd_inp.size()) {
            embd_inp.resize(new_size);
            
            // Синхронизируем session_tokens
            if (!path_session.empty() && !session_tokens.empty()) {
                size_t session_new_size = std::min((size_t)std::max(0, n_keep), session_tokens.size());
                session_tokens.resize(session_new_size);
            }
            
            // В вашей версии API нет прямой функции для полной очистки кэша,
            // поэтому используем существующий подход
            printf(" [Using fallback reset]");
        }
    }
    
    // ВАЖНО: Обновляем n_past после всех изменений
    n_past = (int)embd_inp.size();
    n_session_consumed = n_past;
    
    // Проверка и восстановление BOS-токена, если нужно
    if (vocab_llama) {
        // В вашей версии используется llama_token_bos, а не llama_vocab_bos
        const llama_token bos_token = llama_token_bos(vocab_llama);
        if (!embd_inp.empty() && embd_inp[0] != bos_token) {
            embd_inp.insert(embd_inp.begin(), bos_token);
            
            // Синхронизируем session_tokens
            if (!session_tokens.empty()) {
                session_tokens.insert(session_tokens.begin(), bos_token);
            }
            
            n_past = (int)embd_inp.size();
            n_session_consumed = n_past;
            printf(" [BOS token restored.]");
        }
    } else {
        fprintf(stderr, "WARNING: vocab_llama is null, cannot check BOS token\n");
    }
    
    printf(" [Final context size: %zu. n_past: %d]", embd_inp.size(), n_past);
    
    // Отключаем сессию после сдвига
    path_session = "";
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
            // Обнуляем logits для всего буфера — критично для новых версий llama.cpp
            for (int i = 0; i < 2048; ++i) {
                batch.logits[i] = false;
            }
            batch.n_tokens = static_cast<int>(embd.size());
            for (int i = 0; i < batch.n_tokens; ++i) {
                                                                            
                batch.token[i] = embd[i];
                batch.pos[i] = n_past + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == batch.n_tokens - 1);
            }
        }
// Выполняем декодирование
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
    if (!llama_start_generation_time) llama_start_generation_time = get_current_time_ms();
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
                id = llama_sampler_sample(smpl, ctx_llama, -1);  // Сэмплируем с нормальной температурой
            }
        // Если токен не является токеном окончания (EOS)
        if (id != llama_vocab_eos(vocab_llama)) {
            // Добавляем токен в контекст для следующей итерации
            embd.push_back(id);
            // если модель выводит только одиночную "*", заменяем на fallback-текст
out_token_str = llama_token_to_piece(ctx_llama, id);

            // Просто выводим токен как есть. Не нужно блокировать звездочки — это действия персонажа.
            text_to_speak += out_token_str;  
    printf("%s", out_token_str.c_str());
tokens_in_reply++;
            
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
        if (text_to_speak[text_len-1] == ',') n_comas++;  // Считаем запятые
        // Особая обработка апострофа в split_after
        if (new_tokens == split_after && params.split_after && text_to_speak[text_len-1] == '\'') 
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
                
                // Сигнализируем внешнему сервису
                allow_xtts_file(params.xtts_control_path, 0);
                done = true;
                { // Сброс под защитой мьютекса
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    g_hotkey_pressed = "";
                }
                // 4. Выход из цикла генерации
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
if (text_len >= 2 && new_tokens >=2 && !person_name_is_found && 
    (
        // Разбиение по split_after, если не апостроф
        (new_tokens == split_after && params.split_after && text_to_speak[text_len-1] != '\'') || 
        // Разбиение по различным знакам препинания
        text_to_speak[text_len-1] == '.' ||           // Точка
        // [FIX:] НЕ РАЗБИВАТЬ ПО СКОБКАМ — оставляем любые (...) на одной строке.
        // (удалены условия: text_to_speak[text_len-1] == '(' и == ')')
        // Запятая: только первая, после split_after, если включён split_after
        (text_to_speak[text_len-1] == ',' && n_comas==1 && new_tokens > split_after && params.split_after) || 
        // Тире после пробела
        // (text_to_speak[text_len-2] == ' ' && text_to_speak[text_len-1] == '-') ||  
        text_to_speak[text_len-1] == '?' ||    // Вопросительный знак
        text_to_speak[text_len-1] == '!' ||    // Восклицательный знак
        // text_to_speak[text_len-1] == ';' || // Точка с запятой
        // text_to_speak[text_len-1] == ':' || // Двоеточие
        text_to_speak[text_len-1] == '\n'      // Новая строка
    )
)
{
    // Если идёт процесс перевода, добавляем текст в буфер перевода
    if (translation_is_going == 1) 
    {
        translation_full += text_to_speak;  // Накапливаем текст для перевода
        //fprintf(stdout, " translation_full: (%s)\n", translation_full.c_str());  // Отладочный вывод
    }
    // Подготовка текста для TTS: заменяем кавычки и антипромпты
    text_to_speak = ::replace(text_to_speak, "\"", "'");
    text_to_speak = ::replace(text_to_speak, antiprompts[0], ""); // Удаляем имя пользователя

    // Удаляем имя бота из текста для TTS — он должен быть ТОЛЬКО на экране
    std::string bot_prefix = params.bot_name + ":";
    if (!text_to_speak.empty() && text_to_speak.substr(0, bot_prefix.size()) == bot_prefix) {
        text_to_speak = text_to_speak.substr(bot_prefix.size());
    }
    
    // Если есть текст для озвучки (первая или средняя часть предложения)
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
// XTTS в отдельных потоках
// Сохраняем текст и номер части ответа для асинхронной обработки
int current_reply_part = reply_part++; // захватываем значение ДО инкремента
int current_idx; // индекс в массиве для этого текста
    {
        // БЛОКИРУЕМ доступ к глобальным массивам
        std::lock_guard<std::mutex> lock(g_tts_mutex);
        current_idx = thread_i; // сохраняем текущий индекс
        text_to_speak_arr[current_idx] = text_to_speak; // записываем текст в массив
        reply_part_arr[current_idx] = current_reply_part; // записываем номер части
        thread_i = (thread_i + 1) % 150; // увеличиваем индекс с циклическим буфером
    }

try 
    {
        // Захватываем ВСЁ по значению — никаких глобальных массивов!
        threads.emplace_back([text_to_speak, current_voice, params, current_reply_part]() {
            send_tts_async(text_to_speak, current_voice, params.language, params.xtts_url, current_reply_part);
        });
        // Очищаем локальную переменную
        text_to_speak = "";
        // Если задержка перед XTTS включена, делаем паузу
        // Это помогает ускорить инференс xtts
        if (params.sleep_before_xtts) 
            std::this_thread::sleep_for(std::chrono::milliseconds(params.sleep_before_xtts));
        // Проверяем уровень энергии, если пользователь говорит
        // (не вызывает распознавание whisper, только громкий шум останавливает всё)
        if (!params.push_to_talk || (params.push_to_talk && g_hotkey_pressed == "Alt"))
        {
            // Получаем аудио данные (неблокирующий вызов, 2000 мс)
            audio.get(2000, pcmf32_cur);
            // Проверяем активность голоса (VAD - Voice Activity Detection)
            int vad_result = ::vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE, params.vad_last_ms, 
            params.vad_thold, params.freq_thold, params.print_energy, 
            params.vad_start_thold);

            // Если обнаружена активность голоса
            if (vad_result == 1) {
                if (speech_vad_start_ms == 0) {
                    speech_vad_start_ms = get_current_time_ms() * 1000; // Фиксируем начало звука
                }
                
                // Проверяем длительность: прерываем только если звук длится > 250 мс
                if ((get_current_time_ms() * 1000) - speech_vad_start_ms > 250) {
                    printf(" [Speech interruption confirmed!]\n");
                    llama_interrupted.store(1);
                    g_is_interrupted.store(true);
                    allow_xtts_file(params.xtts_control_path, 0);
                    done = true;
                    break;
                }
            } else {
                // Если звук пропал раньше чем через 250мс — это был шум, сбрасываем таймер
                speech_vad_start_ms = 0;
            }
        }
    }
                // Обработка исключений при создании потока
                catch (const std::exception& ex) {
                    // Выводим сообщение об ошибке создания потока
                    std::cerr << "[Exception]: Failed to push_back mid thread: " << ex.what() << '\n';
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
    std::string last_output;  // Буфер для последних выводимых токенов
    // Собираем последние 10 токенов из контекста плюс текущий токен
    for (int i = embd_inp.size() - 10; i < (int) embd_inp.size(); i++) {
        last_output += llama_token_to_piece(ctx_llama, embd_inp[i]);
    }
    last_output += llama_token_to_piece(ctx_llama, embd[0]);  // Добавляем текущий токен
    int i_antiprompt = 0;
    last_output_has_username = false;  // Флаг наличия имени пользователя
    last_output_has_EOT = false;       // Флаг наличия конца текста
    
    bool antiprompt_matched = false;  // Флаг, был ли найден антипромпт
    
    // Проходим по всем антипромптам
    for (std::string & antiprompt : antiprompts) 
    {
        // Обработка нескольких имён персонажей для XTTS
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
        
        // Обработка стоп-слов
        if (last_output.length() > antiprompt.length() && 
            last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) 
        {
            antiprompt_matched = true;  // Запоминаем, что антипромпт найден
            
            done = true;  // Предварительно устанавливаем флаг завершения
            // Удаляем антипромпт из текста для озвучки
            text_to_speak = ::replace(text_to_speak, antiprompt, "");
            fflush(stdout);
            need_to_save_session = true;
            
            // Если это первый антипромпт (обычно имя пользователя)
            if (i_antiprompt == 0) 
            {
                last_output_has_username = true;
                printf(" ");
            }
            // Если это антипромпт конца текста (EOT)
            else if (i_antiprompt == eot_antiprompt_id_1 || i_antiprompt == eot_antiprompt_id_2) 
            {
                last_output_has_EOT = true;							
            }
            
            // Если антипромпт является суффиксом сообщения бота или тегом конца
            if (antiprompt == params.instruct_preset_data["bot_message_suffix"] || antiprompt == "</end_of_turn>" ) 
            {
                std::string backspaces(antiprompt.length(), '\b');
                std::string spaces(antiprompt.length(), ' ');
                fflush(stdout);
                printf("%s", backspaces.c_str());
                printf("%s", spaces.c_str());
                printf("%s", backspaces.c_str());
                printf("\n");
                fflush(stdout);
            }
            
            // Проверка минимального количества токенов в ответе
            if (params.min_tokens && tokens_in_reply < params.min_tokens)
            {
                int symbols_to_delete = static_cast<int>(utf8_length(antiprompt) * 1) + 1;
                const std::vector<llama_token> tokens_to_del = llama_tokenize(ctx_llama, antiprompt.c_str(), false);
                int rollback_num = tokens_to_del.size() + 1;
                
                if (rollback_num)
                {		
                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                    n_past = embd_inp.size();
                    n_session_consumed = n_past;
                    llama_memory_seq_rm(llama_get_memory(ctx_llama), 0, embd_inp.size(), -1);
                    
                    if (symbols_to_delete > utf8_length(text_to_speak)) text_to_speak = "";
                    else text_to_speak = utf8_substr(text_to_speak, 0, utf8_length(text_to_speak)-symbols_to_delete);
                    
                    temp_next = 1.8;
                    fflush(stdout);
                    printf("\b\b\b\b\b\b\b\b\b\b\b\b");
                    fflush(stdout);
                    done = false;  // Продолжаем генерацию
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
                if (params.debug)
                {
                    std::string full_dialog = emb_to_str(ctx_llama, embd_inp);
                    printf("\n=====FULL text in embd (%zd tokens, %zd symbols)=====\n%s\n====END====\n", 
                           embd_inp.size(), full_dialog.size(), full_dialog.c_str());
                }
                break;
            }
        }
        i_antiprompt++;
    } // КОНЕЕЦ ЦИКЛА for
    
    // ========== ДОБАВЛЕННАЯ ФИНАЛЬНАЯ ПРОВЕРКА ==========
    // СТРАХОВКА: если антипромпт был найден, но ответ слишком короткий,
    // принудительно продолжаем генерацию (перезаписываем done)
    if (antiprompt_matched && params.min_tokens > 0 && tokens_in_reply < params.min_tokens) {
        done = false;  // Гарантированно продолжаем
        if (params.verbose) {
            printf(" [Safety: short response protected] ");
        }
    }
    // ========== КОНЕЦ ДОБАВЛЕННОЙ ПРОВЕРКИ ==========
    
} // КОНЕЦ БЛОКА антипромптов
// ### ОБРАБОТКА АУДИОВХОДА И СИГНАЛОВ (VAD) ###
            // Проверяем SDL события (ввод с клавиатуры, закрытие окна и т.д.)
            is_running = sdl_poll_events();
                // Если приложение не запущено (закрыто), выходим из цикла
                if (!is_running) {
                    break;
                }
            }
            // Финальная часть предложения, если осталась
            text_to_speak = ::replace(text_to_speak, "\"", "'");
            if (text_to_speak.size())  // Если есть текст для озвучки
            {
                int current_reply_part_final;
                int current_idx_final;
                std::string text_to_speak_final = text_to_speak; // Создаём локальную копию
                // ВСЕ операции с разделяемыми данными в одной критической секции
                {
                    std::lock_guard<std::mutex> lock(g_tts_mutex);
                    current_idx_final = thread_i;
                    current_reply_part_final = reply_part;
                    // Сохраняем данные в массивы
                    text_to_speak_arr[current_idx_final] = text_to_speak_final;
                    reply_part_arr[current_idx_final] = current_reply_part_final;
                    // Атомарно обновляем счетчики
                    reply_part++;
                    thread_i = (thread_i + 1) % 150;
                }
                try 
                {		
                    // Захватываем локальные копии, а не оригинальные переменные
                    threads.emplace_back([text_to_speak_final, current_voice, params, current_reply_part_final]() {
                        send_tts_async(text_to_speak_final, current_voice, params.language, params.xtts_url, current_reply_part_final);
                    });
                    
                    text_to_speak = ""; // Очищаем оригинальную переменную
                }
                catch (const std::exception& ex) {
                    std::cerr << "[Exception]: Failed to emplace fin thread: " << ex.what() << '\n'; 
                    
                    // Откатываем изменения под защитой мьютекса
                    {
                        std::lock_guard<std::mutex> lock(g_tts_mutex);
                        text_to_speak_arr[current_idx_final] = "";
                        reply_part_arr[current_idx_final] = 0;
                    }
                    // Внимание: не откатываем reply_part и thread_i для простоты
                }
            }
            // Безопасная очистка всех предыдущих потоков TTS — ВСЕ предыдущие потоки TTS должны быть завершены.
            // Это гарантирует, что при Regenerate/Reset/Exit не будет joinable-потоков в threads.
            // Используем swap + локальный вектор для безопасного join().
            {
                std::vector<std::thread> temp_threads;
                temp_threads.swap(threads); // Перемещаем ВСЕ потоки из threads → temp_threads. threads теперь пуст!
            // безопасное ожидание завершения потоков с таймаутом
            // Теперь безопасно ждём завершения всех старых потоков
            for (auto& t : temp_threads) {
                if (t.joinable()) {
                    try {
                        // Пытаемся подождать завершения не более 5 секунд
                        using namespace std::chrono_literals;
                        auto start = std::chrono::steady_clock::now();
                        while (true) {
                            // Проверяем, не завершился ли поток уже
                            if (!t.joinable()) break;
                            // Если прошло больше 5 секунд — выходим и отсоединяем поток
                            if (std::chrono::steady_clock::now() - start > 5s) {
                                std::cerr << "[warn] join timeout exceeded — detaching thread\n";
                                t.detach();
                                break;
                            }
                            // Пробуем join с малым сном (вместо блокировки навсегда)
                            std::this_thread::sleep_for(100ms);
                            // Поток успевает завершиться — теперь можно безопасно join
                            if (t.joinable()) t.join();
                            break;
                        }
                    } catch (...) {
                        // Игнорируем исключения: главное — не позволить std::terminate()
                        std::cerr << "[warn] exception caught while joining thread\n";
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
            llama_end_time = get_current_time_ms();  // Получаем время окончания
            if (params.verbose)  // Если включён подробный вывод
            {
                // Рассчитываем временные метрики
                llama_time_input = llama_start_generation_time - llama_start_time;    // Время ввода
                llama_time_output = llama_end_time - llama_start_generation_time;     // Время вывода
                llama_time_total = llama_end_time - llama_start_time;                 // Общее время
                // Выводим статистику по контексту и токенам
                printf("\n\n[Context: %d/%d. Tokens: %d in + %d out. Input %.3f s + output %.3f s = total: %.3f s]", 
                        n_past, n_ctx, input_tokens_count, new_tokens, 
                        llama_time_input, llama_time_output, llama_time_total);
                
                // Выводим скорость обработки
                printf("\n[Speed: input %.2f t/s + output %.2f t/s = total: %.2f t/s]\n", 
                        input_tokens_count/llama_time_input, new_tokens/llama_time_output, new_tokens/llama_time_total);
            }
            // Сброс флагов и переменных
            llama_interrupted.store(0);         // Сбрасываем флаг прерывания
            llama_interrupted_time = 0.0;       // Сбрасываем время прерывания
            llama_start_generation_time = 0.0;  // Сбрасываем время начала генерации
           {                                    // Сброс под защитой мьютекса
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                g_hotkey_pressed = "";
            }                                   // Сбрасываем горячую клавишу
        }
    }
}
    // Завершение работы - очистка потоков
    printf("Cleaning up TTS threads...\n");
    {
        // Очищаем все потоки TTS перед завершением
        std::lock_guard<std::mutex> lock(g_threads_mutex);
        for (auto& t : threads) {
            if (t.joinable()) {
                try {
                    t.join();
                } catch (const std::exception& e) {
                    fprintf(stderr, "Warning: Exception joining thread: %s\n", e.what());
                    t.detach();  // Если join не удается, отсоединяем
                } catch (...) {
                    fprintf(stderr, "Warning: Unknown exception joining thread\n");
                    t.detach();  // Если join не удается, отсоединяем
                }
            }
        }
        threads.clear();
    }
    printf("Cleanup complete.\n");
    // ### ЗАВЕРШЕНИЕ РАБОТЫ И ОСВОБОЖДЕНИЕ РЕСУРСОВ ###
    // Завершение работы программы
    audio.pause();  // Приостанавливаем аудио (останавливаем запись с микрофона)
    // Выводим временные метрики работы Whisper (время распознавания речи)
    whisper_print_timings(ctx_wsp);
    // Освобождаем контекст Whisper (очищаем память, занятую моделью распознавания речи)
    whisper_free(ctx_wsp);
    // Выводим статистику производительности сэмплера LLaMA
    llama_perf_sampler_print(smpl);
    // Выводим статистику производительности контекста LLaMA
    llama_perf_context_print(ctx_llama);
    // Освобождаем сэмплер LLaMA (очищаем память)
    llama_sampler_free(smpl);
    // Освобождаем батч (очищаем память, выделенную под батч-обработку)
    llama_batch_free(batch);
    // Освобождаем контекст LLaMA (очищаем память, занятую моделью генерации текста)
    llama_free(ctx_llama);
    // Освобождаем бэкенд LLaMA (завершаем работу библиотеки LLaMA)
    llama_backend_free();
    // Поток ввода висит на readline/cin, его нельзя завершить извне.
    // Вместо join() делаем detach(), чтобы программа могла закрыться немедленно.
    // ОС сама очистит ресурсы при завершении процесса.
    if (input_thread.joinable()) {
        input_thread.detach(); 
    }
    // Ожидаем завершения потока обработки горячих клавиш
    // Метод join() блокирует основной поток до завершения shortcut_thread
    shortcut_thread.join(); // поток обработки горячих клавиш
    // Возвращаем 0 - успешное завершение программы
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