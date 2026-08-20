// ========================================================================
// talk-llama.cpp — Голосовой ассистент на базе Whisper + LLaMA + XTTS
// ВЕРСИЯ 6.1 — ПОЛНЫЙ РЕФАКТОРИНГ С ЧАТОВЫМ РЕЖИМОМ
// ========================================================================
//
// ИЗМЕНЕНИЯ ВЕРСИИ 6.1:
//   1. ВОССТАНОВЛЕН PARTIAL DISPLAY — текст появляется постепенно (каждые 3 сек)
//   2. ИСПРАВЛЕНА КОМАНДА RESET — безопасное пересоздание контекста
//   3. УЛУЧШЕН АНТИПРОМПТЫ — защита от самодиалога
//   4. ЧИСТЫЙ UI — статусы в конце строки, серый цвет, без лишних строк
//   5. ПОЛНЫЕ КОММЕНТАРИИ — каждый блок и переменная описаны
//   6. УДАЛЁН МЁРТВЫЙ КОД — неиспользуемые переменные убраны
//
// ========================================================================
// 1. ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
// ========================================================================

#include "common-sdl.h"      // SDL для захвата аудио с микрофона
#include "common.h"          // Утилиты проекта (замена строк, trim)
#include "common-whisper.h"  // Интеграция с Whisper (VAD, high_pass_filter)
#include "whisper.h"         // Основная библиотека распознавания речи
#include "llama.h"           // Основная библиотека генерации текста

// Системные библиотеки C++
#include <chrono>            // Таймеры, замер длительности
#include <cstdio>            // printf, fprintf
#include <cassert>           // GGML_ASSERT внутри llama.cpp
#include <fstream>           // Чтение prompt-file, instruct-preset JSON
#include <regex>             // Парсинг команд, очистка текста для TTS
#include <sstream>           // stringstream для парсинга float-списков
#include <functional>        // std::function для лямбд в потоках TTS
#include <string>            // std::string
#include <thread>            // std::thread — VAD-монитор, стриминг, TTS
#include <vector>            // std::vector — токены, аудио-буферы
#include <stdexcept>         // Исключения при парсинге аргументов
#include <mutex>             // std::mutex, std::lock_guard
#include <atomic>            // std::atomic — флаги состояний между потоками
#include <iostream>          // std::cerr для ошибок парсинга
#include <algorithm>         // std::find_if, std::transform
#include <cctype>            // std::tolower, std::isspace (C-стиль)
#include <locale>            // std::locale для tolower
#include <clocale>           // setlocale (если потребуется)
#include <codecvt>           // wstring_convert для WinAPI GetTempPath
#include <queue>             // std::queue — очередь ввода с клавиатуры
#include <unordered_set>     // Хэш-множество для is_hallucination
#include <ctype.h>           // isspace, isdigit, isalpha (C-стиль)
#include <map>               // std::map — instruct_preset_data
#include <iterator>          // istreambuf_iterator для чтения файлов
#include <ctime>             // time(), localtime, strftime
#include <filesystem>        // std::filesystem::temp_directory_path (C++17)
#include <random>            // mt19937 для случайных TTS-интро
#include <condition_variable>// std::condition_variable для reset
#include <cmath>             // fabsf для VAD, fabs для tensor-split

// Пользовательские модули (консоль)
#include "console.h"         // Ввод с клавиатуры (readline), цветной вывод
#include "console.cpp"       // Подключаем .cpp напрямую (единый translation unit)

// Сетевые библиотеки
#include <curl/curl.h>       // HTTP POST к XTTS-серверу, GET к Google-поиску
#include "json.hpp"          // nlohmann/json для парсинга instruct-preset

// Заголовки ОС
#ifdef _WIN32
#include <Windows.h>         // WinAPI для мгновенной записи семафора
#include <fileapi.h>
#else
#include <fcntl.h>           // POSIX: open/write/fsync для семафора
#include <unistd.h>
#endif

// ========================================================================
// 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И МЬЮТЕКСЫ
// ========================================================================

// [2.1] Состояния бота
// IDLE — слушает, GENERATING — отвечает, INTERRUPTED — прерван.
// Атомарные переменные позволяют потокам читать/писать без мьютексов.
enum class BotState : uint8_t {
    IDLE = 0,
    GENERATING,
    INTERRUPTED
};
std::atomic<BotState> g_bot_state{ BotState::IDLE };

// [2.2] Состояние UI — управляет правами на печать в консоль.
// WHY: Разделяем ответственность: g_bot_state управляет логикой TTS/распознавания,
//      а g_ui_state управляет ИСКЛЮЧИТЕЛЬНО правами на печать в консоль.
//      Это устраняет гонку потоков, когда [IDLE] затирает префикс «Эмма →».
enum class UiState : uint8_t {
    UI_IDLE = 0,         // Ждём речь; audio_input имеет право показать [IDLE]
    UI_ACCUMULATING,     // Пользователь говорит; строка у audio_input
    UI_HANDOFF,          // Текст ушёл в LLaMA; НИКТО не печатает [IDLE]
    UI_USER_LINE,        // run() печатает реплику пользователя
    UI_BOT_LINE          // run() печатает ответ бота; audio_input молчит
};
std::atomic<UiState> g_ui_state{ UiState::UI_IDLE };

// [2.3] Причина прерывания генерации
enum class InterruptReason : uint8_t {
    NONE = 0,
    VAD_SPEECH,          // Пользователь начал говорить (barge-in)
    HOTKEY_STOP,         // Нажата Ctrl+Space
    HOTKEY_ALT,          // Нажата Alt (push-to-talk)
    MANUAL_STOP          // Команда "стоп"
};
std::atomic<InterruptReason> g_interrupt_reason{ InterruptReason::NONE };
std::atomic<bool> g_interrupt_processed{ false };
std::atomic<bool> g_shutting_down{ false };

// [2.4] Управление TTS-запросами
std::atomic<bool> g_cancel_tts_requests{ false };  // Отмена HTTP-запросов к XTTS
std::string g_xtts_control_file_path = "";         // Путь к файлу-семафору
std::mutex g_xtts_control_mutex;

// [2.5] Очередь ввода с клавиатуры
std::queue<std::string> input_queue;
std::mutex input_mutex;
std::atomic<bool> keyboard_input_running{ true };

// [2.6] Горячие клавиши
std::string g_hotkey_pressed = "";
std::mutex g_hotkey_pressed_mutex;

// [2.7] Управление потоками TTS
std::mutex g_threads_mutex;

// [2.8] Регенерация: храним последний озвученный текст для повтора
std::string g_last_tts_text = "";
std::mutex g_last_tts_mutex;

// [2.9] Флаги управления потоками
std::atomic<bool> g_shortcut_thread_running{ true };
std::mutex g_llama_mutex;
std::atomic<bool> g_verbose_mode{ false };

// [2.10] Мьютекс на консольный вывод
// WHY: Без него printf из audio_input_thread и основного цикла перемешиваются.
std::mutex g_console_mutex;
const int THREAD_CLEANUP_INTERVAL = 5;

// [2.11] Аккумулятор распознанного текста
std::string g_accumulated_text;
std::mutex g_text_accumulator_mutex;

// [2.12] Отдельное хранение отображаемого текста (чат-режим)
std::string g_display_text;
std::mutex g_display_mutex;
std::atomic<bool> g_message_fixed{ false };

// [2.13] Лимит аккумулятора в токенах (вычисляется из ctx_size)
std::atomic<int> g_soft_limit_tokens{ 2560 };

// [2.14] Флаг работы audio_input_thread
std::atomic<bool> g_audio_thread_running{ false };

// [2.15] Передача текста из audio_input_thread в основной цикл
std::atomic<bool> g_pending_llm_request{ false };
std::string g_pending_llm_text;
std::mutex g_pending_llm_mutex;

// [2.16] Буфер для мягкого прерывания TTS
std::string g_interrupted_tts_buffer;
std::mutex g_interrupted_tts_mutex;

// [2.17] Управление сбросом контекста (reset)
// WHY: Предотвращает гонку потоков при пересоздании ctx_llama.
std::atomic<bool> g_reset_in_progress{ false };
std::mutex g_reset_mutex;
std::condition_variable g_reset_cv;


// [2.18] Промпт для Whisper (контекстная подсказка)
std::string g_whisper_prompt = "";

// ========================================================================
// [ПАТЧ] FORWARD DECLARATIONS
// ========================================================================
// WHY: Эти функции используются до своего определения в коде.
//      Без объявлений компилятор выдаёт C3861.
std::string getTempDir();
int utf8_length(const std::string& str);

// ========================================================================
// 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ========================================================================

// [3.1] Токенизация текста для LLaMA
static std::vector<llama_token> llama_tokenize(
    struct llama_context* ctx,
    const std::string& text,
    bool add_bos) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    int n_tokens = static_cast<int>(text.length()) + (add_bos ? 1 : 0);
    std::vector<llama_token> result(n_tokens);

    n_tokens = llama_tokenize(vocab, text.data(), text.length(),
        result.data(), result.size(), add_bos, false);

    if (n_tokens < 0) {
        result.resize(static_cast<size_t>(-n_tokens));
        int check = llama_tokenize(vocab, text.data(), text.length(),
            result.data(), result.size(), add_bos, false);
        if (check != -n_tokens) {
            fprintf(stderr, "Warning: token count mismatch after resize\n");
        }
        result.resize(static_cast<size_t>(n_tokens));
    }
    else {
        result.resize(static_cast<size_t>(n_tokens));
    }
    return result;
}

// [3.2] Преобразование токена обратно в строку
static std::string llama_token_to_piece(
    const struct llama_context* ctx,
    llama_token token) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<char> result(8);
    const int n_tokens = llama_token_to_piece(vocab, token,
        result.data(), result.size(), 0, false);

    if (n_tokens < 0) {
        result.resize(static_cast<size_t>(-n_tokens));
        int check = llama_token_to_piece(vocab, token,
            result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    }
    else {
        result.resize(static_cast<size_t>(n_tokens));
    }
    return std::string(result.data(), result.size());
}

// [3.3] Парсинг списка float через запятую (для --tensor-split)
std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> result;
    if (s.empty()) {
        std::cerr << "Error: Empty input string for float list." << std::endl;
        return result;
    }

    std::stringstream ss(s);
    std::string item;

    try {
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                item.erase(0, item.find_first_not_of(' '));
                item.erase(item.find_last_not_of(' ') + 1);
                if (!item.empty()) {
                    result.push_back(std::stof(item));
                }
            }
        }
        if (result.empty()) {
            std::cerr << "Warning: No valid float numbers found in string: '"
                << s << "'" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing float list from '" << s << "': "
            << e.what() << '\n';
        result.clear();
    }
    return result;
}

// [3.4] Текущее время в секундах (float)
float get_current_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        ) / 1000.0f;
}

// [3.5] Потокобезопасное добавление задачи в вектор потоков TTS
static void safe_thread_emplace(std::vector<std::thread>& threads_vec,
    std::function<void()> task) {
    if (g_shutting_down.load()) {
        return;
    }
    std::scoped_lock lock(g_threads_mutex);
    try {
        threads_vec.emplace_back(std::move(task));
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка создания потока: " << e.what() << std::endl;
    }
}

// [3.6] Очистка завершённых потоков TTS
static void cleanup_finished_threads(std::vector<std::thread>& threads_vec) {
    if (threads_vec.empty()) return;

    std::scoped_lock lock(g_threads_mutex);
    for (auto it = threads_vec.begin(); it != threads_vec.end(); ) {
        if (it->joinable()) {
            ++it;
        }
        else {
            it = threads_vec.erase(it);
        }
    }
    if (threads_vec.size() > 50) {
        size_t to_erase = threads_vec.size() - 25;
        threads_vec.erase(threads_vec.begin(), threads_vec.begin() + to_erase);
        if (g_verbose_mode.load()) {
            fprintf(stderr, "[Threads] Обрезано до %zu\n", threads_vec.size());
        }
    }
}

// ========================================================================
// 4. СТРУКТУРА ПАРАМЕТРОВ КОМАНДНОЙ СТРОКИ
// ========================================================================

struct whisper_params {
    // Производительность
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_gpu_layers = 999;
    bool use_gpu = true;
    bool flash_attn = false;
    int main_gpu = 0;
    std::string split_mode = "none";
    std::vector<float> tensor_split;

    // Аудиозахват
    int32_t voice_ms = 10000;
    int32_t capture_id = -1;

    // Whisper
    int32_t max_tokens = 64;
    int32_t audio_ctx = 0;

    // VAD
    float vad_thold = 0.6f;
    float vad_start_thold = 0.000270f;
    float vad_last_ms = 1250.0f;
    float freq_thold = 90.0f;

    // Прерывание
    int32_t interrupt_check_ms = 50;
    int32_t interrupt_threshold_ms = 100;

    // LLaMA
    int32_t ctx_size = 2048;
    int32_t batch_size = 64;
    int32_t n_predict = 64;
    int32_t min_tokens = 0;
    float temp = 0.9f;
    int32_t top_k = 40;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repeat_penalty = 1.10f;
    int repeat_last_n = 256;
    int n_keep = 128;
    bool safe_context_shift = true;
    bool allow_newline = false;
    bool seqrep = false;

    // TTS
    std::string xtts_voice = "Эмма";
    std::string xtts_url = "http://localhost:8020/";
    std::string xtts_control_path = "xtts_play_allowed.txt";
    bool xtts_intro = false;
    int sleep_before_xtts = 0;
    int split_after = 0;

    // Имена
    std::string person = "Друг";
    std::string bot_name = "Эмма";
    std::string wake_cmd = "";
    std::string heard_ok = "";

    // Язык
    std::string language = "ru";
    bool translate = false;

    // Модели
    std::string model_wsp = "whisper-ggml-medium-q4_0.bin";
    std::string model_llama = "saiga_yandexgpt_8b_Q5_K.gguf";

    // Прочие
    bool speed_up = false;
    bool print_special = false;
    bool print_energy = false;
    bool debug = false;
    bool no_timestamps = true;
    bool verbose_prompt = false;
    bool verbose = false;
    bool multi_chars = false;
    bool push_to_talk = false;
    std::string speak = "speak";
    std::string speak_file = "to_speak.txt";
    std::string google_url = "http://localhost:8003/";
    std::string prompt = "";
    std::string instruct_preset = "";
    std::string fname_out = "";
    std::string path_session = "";
    std::string stop_words = "";

    // Instruct-пресет (данные из JSON)
    std::map<std::string, std::string> instruct_preset_data = {
        {"system_prompt_prefix", ""},
        {"system_prompt_suffix", ""},
        {"user_message_prefix", ""},
        {"user_message_suffix", ""},
        {"bot_message_prefix", ""},
        {"bot_message_suffix", ""},
        {"stop_sequence", ""}
    };
};

// ========================================================================
// 5. ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
// ========================================================================

void whisper_print_usage(int argc, char** argv, const whisper_params& params);

bool whisper_params_parse(int argc, char** argv, whisper_params& params) {
    params.tensor_split.clear();

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        try {
            if (arg == "-h" || arg == "--help") {
                whisper_print_usage(argc, argv, params);
                return false;
            }
            else if (arg == "-t" || arg == "--threads") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_threads = std::stoi(argv[++i]);
            }
            else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_gpu_layers = std::stoi(argv[++i]);
            }
            else if (arg == "-ng" || arg == "--no-gpu") { params.use_gpu = false; }
            else if (arg == "-fa" || arg == "--flash-attn") { params.flash_attn = true; }
            else if (arg == "--main-gpu") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.main_gpu = std::stoi(argv[++i]);
            }
            else if (arg == "--split-mode") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.split_mode = argv[++i];
            }
            else if (arg == "--tensor-split") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                std::string tensor_split_str = argv[++i];
                if (tensor_split_str.empty()) {
                    std::cerr << "Error: empty tensor-split list" << std::endl;
                    return false;
                }
                params.tensor_split = parse_float_list(tensor_split_str);
                if (params.tensor_split.empty()) {
                    std::cerr << "Error: failed to parse tensor-split list: '"
                        << tensor_split_str << "'" << std::endl;
                    return false;
                }
                float sum = 0.0f;
                for (float val : params.tensor_split) {
                    if (val < 0.0f || val > 1.0f) {
                        std::cerr << "Error: tensor-split values must be between 0.0 and 1.0, got: "
                            << val << std::endl;
                        return false;
                    }
                    sum += val;
                }
                if (fabs(sum - 1.0f) > 0.001f) {
                    std::cerr << "Warning: tensor-split values sum to " << sum
                        << " (expected ~1.0)" << std::endl;
                }
            }
            else if (arg == "-vms" || arg == "--voice-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.voice_ms = std::stoi(argv[++i]);
            }
            else if (arg == "-c" || arg == "--capture") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.capture_id = std::stoi(argv[++i]);
            }
            else if (arg == "-mt" || arg == "--max-tokens") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.max_tokens = std::stoi(argv[++i]);
            }
            else if (arg == "-ac" || arg == "--audio-ctx") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.audio_ctx = std::stoi(argv[++i]);
            }
            else if (arg == "-vth" || arg == "--vad-thold") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.vad_thold = std::stof(argv[++i]);
            }
            else if (arg == "-vths" || arg == "--vad-start-thold") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.vad_start_thold = std::stof(argv[++i]);
            }
            else if (arg == "-vlm" || arg == "--vad-last-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.vad_last_ms = std::stof(argv[++i]);
            }
            else if (arg == "-fth" || arg == "--freq-thold") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.freq_thold = std::stof(argv[++i]);
            }
            else if (arg == "--interrupt-check-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.interrupt_check_ms = std::stoi(argv[++i]);
            }
            else if (arg == "--interrupt-threshold-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.interrupt_threshold_ms = std::stoi(argv[++i]);
            }
            else if (arg == "--ctx_size") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.ctx_size = std::stoi(argv[++i]);
            }
            else if (arg == "-b" || arg == "--batch-size") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.batch_size = std::stoi(argv[++i]);
            }
            else if (arg == "-n" || arg == "--n_predict") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_predict = std::stoi(argv[++i]);
            }
            else if (arg == "--temp") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.temp = std::stof(argv[++i]);
            }
            else if (arg == "--top_k") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.top_k = std::stoi(argv[++i]);
            }
            else if (arg == "--top_p") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.top_p = std::stof(argv[++i]);
            }
            else if (arg == "--min_p") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.min_p = std::stof(argv[++i]);
            }
            else if (arg == "--repeat_penalty") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.repeat_penalty = std::stof(argv[++i]);
            }
            else if (arg == "--repeat_last_n") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.repeat_last_n = std::stoi(argv[++i]);
            }
            else if (arg == "--n_keep") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_keep = std::stoi(argv[++i]);
            }
            else if (arg == "--min-tokens") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.min_tokens = std::stoi(argv[++i]);
            }
            else if (arg == "--allow-newline") { params.allow_newline = true; }
            else if (arg == "--seqrep") { params.seqrep = true; }
            else if (arg == "--xtts-voice") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.xtts_voice = argv[++i];
            }
            else if (arg == "--xtts-url") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.xtts_url = argv[++i];
            }
            else if (arg == "--xtts-control-path") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.xtts_control_path = argv[++i];
            }
            else if (arg == "--xtts-intro") { params.xtts_intro = true; }
            else if (arg == "--sleep-before-xtts") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.sleep_before_xtts = std::stoi(argv[++i]);
            }
            else if (arg == "--split-after") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.split_after = std::stoi(argv[++i]);
            }
            else if (arg == "-p" || arg == "--person") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.person = argv[++i];
            }
            else if (arg == "-bn" || arg == "--bot-name") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.bot_name = argv[++i];
            }
            else if (arg == "-w" || arg == "--wake-command") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.wake_cmd = argv[++i];
            }
            else if (arg == "-ho" || arg == "--heard-ok") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.heard_ok = argv[++i];
            }
            else if (arg == "-l" || arg == "--language") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.language = argv[++i];
            }
            else if (arg == "-tr" || arg == "--translate") { params.translate = true; }
            else if (arg == "-mw" || arg == "--model-whisper") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.model_wsp = argv[++i];
            }
            else if (arg == "-ml" || arg == "--model-llama") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.model_llama = argv[++i];
            }
            else if (arg == "-su" || arg == "--speed-up") { params.speed_up = true; }
            else if (arg == "-ps" || arg == "--print-special") { params.print_special = true; }
            else if (arg == "-pe" || arg == "--print-energy") { params.print_energy = true; }
            else if (arg == "--debug") { params.debug = true; }
            else if (arg == "-vp" || arg == "--verbose-prompt") { params.verbose_prompt = true; }
            else if (arg == "--verbose") { params.verbose = true; g_verbose_mode.store(true); }
            else if (arg == "--multi-chars") { params.multi_chars = true; }
            else if (arg == "--push-to-talk") { params.push_to_talk = true; }
            else if (arg == "-s" || arg == "--speak") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.speak = argv[++i];
            }
            else if (arg == "-sf" || arg == "--speak-file") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.speak_file = argv[++i];
            }
            else if (arg == "--google-url") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.google_url = argv[++i];
            }
            else if (arg == "--stop-words") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.stop_words = argv[++i];
            }
            else if (arg == "--instruct-preset") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.instruct_preset = argv[++i];
            }
            else if (arg == "--prompt-file") {
                if (i + 1 >= argc) { whisper_print_usage(argc, argv, params); return false; }
                std::ifstream file(argv[++i]);
                if (!file.is_open()) {
                    std::cerr << "Failed to open prompt file: " << argv[i] << std::endl;
                    return false;
                }
                std::copy(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>(),
                    std::back_inserter(params.prompt));
                if (!params.prompt.empty() && params.prompt.back() == '\n') {
                    params.prompt.pop_back();
                }
            }
            else if (arg == "--session") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.path_session = argv[++i];
            }
            else if (arg == "-f" || arg == "--file") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.fname_out = argv[++i];
            }
            else {
                fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
                whisper_print_usage(argc, argv, params);
                return false;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing argument: " << e.what() << std::endl;
            whisper_print_usage(argc, argv, params);
            return false;
        }
    }
    return true;
}

// [5.1] Справка по параметрам (--help)
void whisper_print_usage(int /*argc*/, char** argv, const whisper_params& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n", params.voice_ms);
    fprintf(stderr, "  --interrupt-check-ms N     [%-7d] how often to check mic during generation (ms)\n", params.interrupt_check_ms);
    fprintf(stderr, "  --interrupt-threshold-ms N [%-7d] how much speech to interrupt generation (ms)\n", params.interrupt_threshold_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n", params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice activity detection threshold\n", params.vad_thold);
    fprintf(stderr, "  -vths N,  --vad-start-thold N [%-7.6f] vad min level to start, 0: off\n", params.vad_start_thold);
    fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7.2f] vad min silence after speech, ms\n", params.vad_last_ms);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up       [%-7s] speed up audio by x2 (not working)\n", params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n", params.print_energy ? "true" : "false");
    fprintf(stderr, "  --debug                    [%-7s] print debug info\n", params.debug ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n", params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  --verbose                  [%-7s] print speed and debug info\n", params.verbose ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn     [%-7s] flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt)\n", params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n", params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n", params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n", params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n", params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n", params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n", params.speak.c_str());
    fprintf(stderr, "  -sf FILE, --speak-file     [%-7s] file to pass to TTS\n", params.speak_file.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n", " ");
    fprintf(stderr, "  --instruct-preset TEXT     [%-7s] instruct preset to use without .json\n", " ");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n", params.fname_out.c_str());
    fprintf(stderr, "  --ctx_size N               [%-7d] size of the prompt context\n", params.ctx_size);
    fprintf(stderr, "  -b N,     --batch-size N   [%-7d] size of input batch\n", params.batch_size);
    fprintf(stderr, "  -n N,     --n_predict N    [%-7d] max number of tokens to predict\n", params.n_predict);
    fprintf(stderr, "  --temp N                   [%-7.2f] temperature\n", params.temp);
    fprintf(stderr, "  --top_k N                  [%-7d] top_k\n", params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p\n", params.top_p);
    fprintf(stderr, "  --min_p N                  [%-7.2f] min_p\n", params.min_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repeat_penalty\n", params.repeat_penalty);
    fprintf(stderr, "  --repeat_last_n N          [%-7d] repeat_last_n\n", params.repeat_last_n);
    fprintf(stderr, "  --n_keep N                 [%-7d] keep first n_tokens after context_shift\n", params.n_keep);
    fprintf(stderr, "  --main-gpu N               [%-7d] main GPU id, starting from 0\n", params.main_gpu);
    fprintf(stderr, "  --split-mode NAME          [%-7s] GPU split mode: 'none' or 'layer'\n", params.split_mode.c_str());
    fprintf(stderr, "  --tensor-split NAME        [    ] tensor split, list of floats: 0.5,0.5\n");
    fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n", params.xtts_voice.c_str());
    fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts/silero server URL, with trailing slash\n", params.xtts_url.c_str());
    fprintf(stderr, "  --xtts-control-path FNAME  [%-7s] not used anymore\n", params.xtts_control_path.c_str());
    fprintf(stderr, "  --xtts-intro               [%-7s] xtts instant short random intro like Hmmm\n", params.xtts_intro ? "true" : "false");
    fprintf(stderr, "  --sleep-before-xtts        [%-7d] sleep llama inference before xtts, ms\n", params.sleep_before_xtts);
    fprintf(stderr, "  --google-url TEXT          [%-7s] langchain google-serper server URL, with /\n", params.google_url.c_str());
    fprintf(stderr, "  --allow-newline            [%-7s] allow new line in llama output\n", params.allow_newline ? "true" : "false");
    fprintf(stderr, "  --multi-chars              [%-7s] xtts will use same wav name as in llama output\n", params.multi_chars ? "true" : "false");
    fprintf(stderr, "  --push-to-talk             [%-7s] hold Alt to speak\n", params.push_to_talk ? "true" : "false");
    fprintf(stderr, "  --seqrep                   [%-7s] sequence repetition penalty\n", params.seqrep ? "true" : "false");
    fprintf(stderr, "  --split-after N            [%-7d] split after first n tokens for tts\n", params.split_after);
    fprintf(stderr, "  --min-tokens N             [%-7d] min new tokens to output\n", params.min_tokens);
    fprintf(stderr, "  --stop-words TEXT          [%-7s] llama stop words separated by ;\n", params.stop_words.c_str());
    fprintf(stderr, "\n");
}

// ========================================================================
// 6. СТРОКОВЫЕ УТИЛИТЫ
// ========================================================================

// [6.1] Удаление пробелов в начале строки
inline void ltrim(std::string& s) {
    if (s.empty()) return;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
            && ch != '\f' && ch != '\v' && ch != 0xA0; // 0xA0 = неразрывный пробел UTF-8
        }));
}

// [6.2] Удаление пробелов в конце строки
inline void rtrim(std::string& s) {
    if (s.empty()) return;
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
            && ch != '\f' && ch != '\v' && ch != 0xA0;
        }).base(), s.end());
}

// [6.3] Удаление пробелов с обоих концов
inline void trim(std::string& s) {
    if (s.empty()) return;
    rtrim(s);
    ltrim(s);
}

// [6.4] Проверка, является ли символ пунктуационным
bool IsPunctuationMark(char c) {
    switch (static_cast<unsigned char>(c)) {
    case ',': case '.': case '?': case '!': return true;
    default: return false;
    }
}

// [6.5] Удаление пунктуационных знаков из строки
std::string StripPunctuationMarks(const std::string& text) {
    std::string cleanText;
    for (const auto& c : text) {
        if (!IsPunctuationMark(c)) {
            cleanText += c;
        }
    }
    return cleanText;
}

// [6.6] Перевод строки в нижний регистр с корректной обработкой UTF-8 кириллицы
std::string LowerCase(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            // ASCII: стандартный tolower
            result += static_cast<char>(std::tolower(c));
            ++i;
        }
        else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            // 2-байтовый UTF-8 (кириллица)
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            if (c == 0xD0 && c2 >= 0x90 && c2 <= 0xAF) {
                // А-Я → а-я: сдвиг второго байта на +0x20
                result += static_cast<char>(0xD0);
                result += static_cast<char>(c2 + 0x20);
            }
            else if (c == 0xD0 && c2 == 0x81) {
                // Ё → ё: замена 0xD0 0x81 на 0xD1 0x91
                result += static_cast<char>(0xD1);
                result += static_cast<char>(0x91);
            }
            else {
                result += text[i];
                result += text[i + 1];
            }
            i += 2;
        }
        else {
            // 3+ байтовые символы (эмодзи, CJK) — копируем как есть
            size_t len = 1;
            if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            for (size_t j = 0; j < len && i + j < text.size(); ++j) {
                result += text[i + j];
            }
            i += len;
        }
    }
    return result;
}

// [6.7] Замена всех вхождений подстроки
std::string string_replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

// [6.8] Удаление хвостовых символов (ASCII)
std::string RemoveTrailingCharacters(const std::string& inputString,
    const char targetCharacter) {
    auto lastNonTargetPosition = std::find_if(
        inputString.rbegin(), inputString.rend(),
        [targetCharacter](auto ch) { return ch != targetCharacter; });
    return std::string(inputString.begin(), lastNonTargetPosition.base());
}

// [6.9] Удаление хвостовых символов (UTF-8)
std::string RemoveTrailingCharactersUtf8(const std::string& inputString,
    const std::string& targetCharacters) {
    if (inputString.empty()) {
        return inputString;
    }
    size_t pos = inputString.length();
    while (pos > 0) {
        // Находим начало последнего UTF-8 символа
        size_t char_start = pos - 1;
        while (char_start > 0 &&
            (static_cast<unsigned char>(inputString[char_start]) & 0xC0) == 0x80) {
            char_start--;
        }
        std::string last_char = inputString.substr(char_start, pos - char_start);
        bool should_remove = false;
        if (last_char.size() == 1) {
            char c = last_char[0];
            for (char target : targetCharacters) {
                if (c == target) {
                    should_remove = true;
                    break;
                }
            }
        }
        if (!should_remove) {
            break;
        }
        pos = char_start;
    }
    return inputString.substr(0, pos);
}

// [6.10] Количество UTF-8 символов в строке
int utf8_length(const std::string& str) {
    if (str.empty()) return 0;
    size_t i = 0;
    int chars = 0;
    const size_t ix = str.size();
    while (i < ix) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        if (c <= 0x7F) { ++i; }
        else if ((c & 0xE0) == 0xC0) { if (i + 1 >= ix) return chars; i += 2; }
        else if ((c & 0xF0) == 0xE0) { if (i + 2 >= ix) return chars; i += 3; }
        else if ((c & 0xF8) == 0xF0) { if (i + 3 >= ix) return chars; i += 4; }
        else { ++i; }
        ++chars;
    }
    return chars;
}

// [6.11] Подстрока по индексам UTF-8 символов
std::string utf8_substr(const std::string& str, unsigned int start,
    unsigned int leng) {
    if (leng == 0) return "";
    const size_t ix = str.size();
    size_t i = 0;
    unsigned int chars = 0;
    size_t min_byte_index = std::string::npos;
    size_t max_byte_index = std::string::npos;
    while (i < ix) {
        if (chars == start) min_byte_index = i;
        if (chars == start + leng) {
            max_byte_index = i;
            break;
        }
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t step = 1;
        if (c <= 0x7F) { step = 1; }
        else if ((c & 0xE0) == 0xC0) { step = 2; if (i + 1 >= ix) return ""; }
        else if ((c & 0xF0) == 0xE0) { step = 3; if (i + 2 >= ix) return ""; }
        else if ((c & 0xF8) == 0xF0) { step = 4; if (i + 3 >= ix) return ""; }
        else { return ""; }
        i += step;
        ++chars;
    }
    if (max_byte_index == std::string::npos) max_byte_index = ix;
    if (min_byte_index == std::string::npos || max_byte_index > ix) return "";
    return str.substr(min_byte_index, max_byte_index - min_byte_index);
}

// [6.12] Транслитерация английских букв в русские
std::string translit_en_ru(const std::string& str) {
    static const std::unordered_map<char, std::string> tbl = {
        {'a', u8"а"}, {'b', u8"б"}, {'c', u8"ц"}, {'d', u8"д"}, {'e', u8"е"},
        {'f', u8"ф"}, {'g', u8"г"}, {'h', u8"х"}, {'i', u8"и"}, {'j', u8"й"},
        {'k', u8"к"}, {'l', u8"л"}, {'m', u8"м"}, {'n', u8"н"}, {'o', u8"о"},
        {'p', u8"п"}, {'q', u8"к"}, {'r', u8"р"}, {'s', u8"с"}, {'t', u8"т"},
        {'u', u8"у"}, {'v', u8"в"}, {'w', u8"в"}, {'x', u8"кс"}, {'y', u8"й"},
        {'z', u8"з"},
        {'A', u8"А"}, {'B', u8"Б"}, {'C', u8"Ц"}, {'D', u8"Д"}, {'E', u8"Е"},
        {'F', u8"Ф"}, {'G', u8"Г"}, {'H', u8"Х"}, {'I', u8"И"}, {'J', u8"Й"},
        {'K', u8"К"}, {'L', u8"Л"}, {'M', u8"М"}, {'N', u8"Н"}, {'O', u8"О"},
        {'P', u8"П"}, {'Q', u8"К"}, {'R', u8"Р"}, {'S', u8"С"}, {'T', u8"Т"},
        {'U', u8"У"}, {'V', u8"В"}, {'W', u8"В"}, {'X', u8"Кс"}, {'Y', u8"Й"},
        {'Z', u8"З"}
    };
    std::string out;
    out.reserve(str.size() * 2);
    for (size_t i = 0; i < str.size();) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        if (c < 0x80) {
            auto it = tbl.find(static_cast<char>(c));
            if (it != tbl.end()) { out += it->second; }
            else { out.push_back(static_cast<char>(c)); }
            ++i;
        }
        else {
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else { out.push_back(static_cast<char>(c)); ++i; continue; }
            if (i + len <= str.size()) { out.append(str.data() + i, len); i += len; }
            else { out.append(str.data() + i, str.size() - i); break; }
        }
    }
    return out;
}

// ========================================================================
// КОНЕЦ ЧАСТИ 1/3
// ПРОДОЛЖЕНИЕ В ЧАСТИ 2/3
// ========================================================================

// ========================================================================
// ЧАСТЬ 2/3 — ФУНКЦИИ TTS, VAD, WHISPER, АУДИО-ПОТОК
// ========================================================================

// ========================================================================
// 7. УПРАВЛЕНИЕ СЕМАФОРОМ TTS
// ========================================================================

// [7.1] Мгновенная запись байта «0» или «1» в файл семафора
// WHY: std::ofstream слишком медленный для real-time прерываний.
//      WinAPI WriteFile + FlushFileBuffers или POSIX open + write + fsync
//      гарантируют, что Python-сервер XTTS увидит изменение мгновенно.
static void write_semaphore_instant(const std::string& filepath, bool allowed) {
    const char* value = allowed ? "1" : "0";

#ifdef _WIN32
    HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        NULL, CREATE_ALWAYS, FILE_FLAG_WRITE_THROUGH, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD bytesWritten;
        WriteFile(hFile, value, 1, &bytesWritten, NULL);
        FlushFileBuffers(hFile);
        CloseHandle(hFile);
    }
    else if (g_verbose_mode.load()) {
        std::cerr << "[WARN] write_semaphore_instant: не удалось открыть "
            << filepath << std::endl;
    }
#else
    int fd = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        write(fd, value, 1);
        fsync(fd);
        close(fd);
    }
    else if (g_verbose_mode.load()) {
        std::cerr << "[WARN] write_semaphore_instant: не удалось открыть "
            << filepath << std::endl;
    }
#endif
}

// [7.2] Обёртка для записи семафора
static void allow_xtts_file(std::string& path, int xtts_play_allowed) {
    if (g_xtts_control_file_path.empty()) {
        std::string temp_path = getTempDir();
        if (temp_path.empty()) {
            std::cerr << "[CRITICAL] allow_xtts_file: не удалось получить "
                "временную директорию." << std::endl;
            return;
        }
#if __cplusplus >= 201703L
        std::filesystem::path p(temp_path);
        g_xtts_control_file_path = (p / "xtts_play_allowed.txt").string();
#else
        g_xtts_control_file_path = temp_path;
        if (g_xtts_control_file_path.back() != '/' &&
            g_xtts_control_file_path.back() != '\\')
            g_xtts_control_file_path += '/';
        g_xtts_control_file_path += "xtts_play_allowed.txt";
#endif
    }

    path = g_xtts_control_file_path;

    std::lock_guard<std::mutex> lock(g_xtts_control_mutex);
    write_semaphore_instant(g_xtts_control_file_path, xtts_play_allowed == 1);

    if (g_verbose_mode.load()) {
        fprintf(stderr, "[Semaphore] Записано %d в %s\n",
            xtts_play_allowed, g_xtts_control_file_path.c_str());
    }
}

// ========================================================================
// 8. ПОЛУЧЕНИЕ ВРЕМЕННОЙ ДИРЕКТОРИИ
// ========================================================================

// [8.1] Получение временной директории (кроссплатформенно)
std::string getTempDir() {
    try {
        auto temp_path = std::filesystem::temp_directory_path();
        if (!temp_path.empty()) {
            return temp_path.string();
        }
    }
    catch (const std::exception& e) {
        if (g_verbose_mode.load()) {
            std::cerr << "[getTempDir] filesystem exception: " << e.what() << std::endl;
        }
    }
    catch (...) {
        if (g_verbose_mode.load()) {
            std::cerr << "[getTempDir] неизвестное исключение filesystem" << std::endl;
        }
    }
#ifdef _WIN32
    TCHAR path_buf[MAX_PATH] = { 0 };
    DWORD ret_val = GetTempPath(MAX_PATH, path_buf);
    if (ret_val == 0 || ret_val > MAX_PATH) {
        if (g_verbose_mode.load()) {
            std::cerr << "[getTempDir] GetTempPath failed, error: "
                << GetLastError() << std::endl;
        }
        return "";
    }
    if (path_buf[0] == 0) {
        return "";
    }
#if defined(UNICODE) || defined(_UNICODE)
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string result = converter.to_bytes(path_buf);
        if (!result.empty() && (result.back() == '\\' || result.back() == '/')) {
            result.pop_back();
        }
        return result;
    }
    catch (...) {
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
    const char* tmpdir = std::getenv("TMPDIR");
    if (tmpdir && tmpdir[0] != '\0') {
        return std::string(tmpdir);
    }
    return "/tmp";
#endif
}

// ========================================================================
// 9. URL-КОДИРОВАНИЕ И CURL-ЗАПРОСЫ
// ========================================================================

// [9.1] URL-кодирование строки
std::string UrlEncode(const std::string& str) {
    CURL* curl = curl_easy_init();
    if (curl) {
        char* encodedUrl = curl_easy_escape(curl, str.c_str(),
            static_cast<int>(str.length()));
        std::string escapedUrl;
        if (encodedUrl) {
            escapedUrl.assign(encodedUrl);
            curl_free(encodedUrl);
        }
        curl_easy_cleanup(curl);
        return escapedUrl;
    }
    return {};
}

// [9.2] Callback для записи данных, полученных через CURL
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    if (userp) {
        ((std::string*)userp)->append(static_cast<char*>(contents), realsize);
    }
    return realsize;
}

// [9.3] Callback прогресса TTS-запросов
static int progress_callback(void* /*clientp*/,
    curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
    curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    if (g_cancel_tts_requests.load()) {
        return 1; // Прерываем запрос
    }
    return 0;
}

// [9.4] Отправка JSON-данных на сервер
std::string send_curl_json(const std::string& url,
    const std::map<std::string, std::string>& params) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

    struct CurlDeleter {
        void operator()(CURL* c) const { if (c) curl_easy_cleanup(c); }
    };
    std::unique_ptr<CURL, CurlDeleter> curl_guard(curl);

    struct SlistDeleter {
        void operator()(curl_slist* s) const { if (s) curl_slist_free_all(s); }
    };
    std::unique_ptr<curl_slist, SlistDeleter> headers_guard(nullptr);

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
            default:   result += static_cast<char>(c);
            }
        }
        return result;
    };

    try {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

        std::ostringstream oss;
        bool firstParam = true;
        oss << "{";
        for (const auto& param : params) {
            if (!firstParam) oss << ", ";
            std::string escaped_key = escape_json(param.first);
            std::string escaped_value = escape_json(param.second);
            if (!escaped_key.empty()) {
                oss << "\"" << escaped_key << "\": \"" << escaped_value << "\"";
            }
            else {
                if (g_verbose_mode.load()) {
                    fprintf(stderr, "Warning: skipping empty JSON key\n");
                }
                continue;
            }
            firstParam = false;
        }
        oss << "}";
        std::string jsonData = oss.str();
        if (jsonData.size() <= 2) {
            if (g_verbose_mode.load()) {
                fprintf(stderr, "Warning: generated empty JSON, using fallback\n");
            }
            jsonData = "{}";
        }

        curl_slist* headers = curl_slist_append(nullptr,
            "Content-Type: application/json");
        headers_guard.reset(headers);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        CURLcode res = curl_easy_perform(curl);
        (void)res;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return "";
    }
    return readBuffer;
}

// [9.5] Выполнение GET-запроса
std::string send_curl(std::string url) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

// ========================================================================
// 10. VAD-ФУНКЦИЯ
// ========================================================================

// [10.1] VAD с возвратом состояний 0/1/2
// 0 — речь продолжается (энергия выше фона, но ниже порога старта)
// 1 — начало речи (энергия превысила vad_start_thold)
// 2 — окончание речи (энергия упала ниже фона)
static int vad_simple_int(std::vector<float>& pcmf32, int sample_rate,
    int last_ms, float vad_thold, float freq_thold,
    bool verbose, float vad_start_thold = 0.000270f) {
    const int n_samples = static_cast<int>(pcmf32.size());
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        return 0; // Недостаточно сэмплов — считаем, что речи нет
    }

    // High-pass фильтр: убирает низкочастотный гул (вентиляторы, сеть 50 Гц)
    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
        energy_all += fabsf(pcmf32[i]);
        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr, "[VAD] energy_all: %f, energy_last: %f, thold: %f\n",
            energy_all, energy_last, vad_thold);
    }

    // Абсолютный порог: энергия выше vad_start_thold → начало речи
    if (vad_start_thold > 0.0f && energy_last > vad_start_thold) {
        if (verbose) printf("[VAD] speech started (energy > %f)\n", vad_start_thold);
        return 1;
    }

    // Относительный порог: энергия выше фона → речь продолжается
    if (energy_last > vad_thold * energy_all) {
        return 0;
    }

    // Энергия упала → конец речи
    if (verbose) printf("[VAD] speech ended (energy < %f)\n", vad_thold * energy_all);
    return 2;
}

// ========================================================================
// 11. ТРАНСКРИБАЦИЯ WHISPER И ФИЛЬТРАЦИЯ ГАЛЛЮЦИНАЦИЙ
// ========================================================================

// [11.1] Основная функция распознавания речи
static std::string transcribe(
    whisper_context* ctx,
    const whisper_params& params,
    const std::vector<float>& pcmf32,
    const std::string& prompt_text,
    float& prob,
    int64_t& t_ms) {
    prob = 0.0f;
    t_ms = 0;

    if (!ctx) {
        std::cerr << "Ошибка: контекст Whisper не инициализирован" << std::endl;
        return "";
    }
    if (pcmf32.empty()) {
        if (params.verbose) {
            std::cerr << "Ошибка: входные аудиоданные пусты" << std::endl;
        }
        return "";
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Промпт помогает модели понимать контекст: кто говорит, на каком языке
    std::vector<whisper_token> prompt_tokens_vec;
    if (!prompt_text.empty()) {
        prompt_tokens_vec.resize(prompt_text.size() + 1);
        int n_tokens = whisper_tokenize(ctx, prompt_text.c_str(),
            prompt_tokens_vec.data(),
            static_cast<int>(prompt_tokens_vec.size()));
        if (n_tokens > 0) {
            prompt_tokens_vec.resize(static_cast<size_t>(n_tokens));
            wparams.prompt_tokens = prompt_tokens_vec.data();
            wparams.prompt_n_tokens = static_cast<int>(prompt_tokens_vec.size());
        }
        else {
            wparams.prompt_tokens = nullptr;
            wparams.prompt_n_tokens = 0;
        }
    }
    else {
        wparams.prompt_tokens = nullptr;
        wparams.prompt_n_tokens = 0;
    }

    wparams.print_progress = false;
    wparams.print_special = params.print_special;
    wparams.print_realtime = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.no_timestamps = params.no_timestamps;
    wparams.translate = params.translate;

    // КРИТИЧНО: no_context=true — каждый вызов независим, нет петель галлюцинаций
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.token_timestamps = false;
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;
    wparams.temperature = 0.2f;
    wparams.temperature_inc = 0.0f;
    wparams.length_penalty = 0.5f;
    wparams.entropy_thold = 2.4f;
    wparams.max_len = 96;

    {
        int model_text_ctx = static_cast<int>(whisper_n_text_ctx(ctx));
        int mt = (params.max_tokens > 0) ? params.max_tokens : 64;
        if (mt > model_text_ctx) {
            if (params.verbose) {
                std::cerr << "Предупреждение: max_tokens (" << mt
                    << ") превышает лимит модели (" << model_text_ctx
                    << "), применяется лимит модели" << std::endl;
            }
            mt = model_text_ctx;
        }
        wparams.max_tokens = mt;
    }

    wparams.audio_ctx = params.audio_ctx;
    int model_audio_ctx = static_cast<int>(whisper_n_audio_ctx(ctx));
    if (wparams.audio_ctx > model_audio_ctx) {
        if (params.verbose) {
            std::cerr << "Предупреждение: audio_ctx (" << wparams.audio_ctx
                << ") превышает лимит модели (" << model_audio_ctx
                << "), применяется лимит модели" << std::endl;
        }
        wparams.audio_ctx = model_audio_ctx;
    }

    wparams.language = params.language.empty() ? nullptr : params.language.c_str();
    wparams.n_threads = params.n_threads;

    if (whisper_full(ctx, wparams, pcmf32.data(),
        static_cast<int>(pcmf32.size())) != 0) {
        if (params.verbose) {
            std::cerr << "Ошибка: не удалось выполнить транскрипцию аудио" << std::endl;
        }
        const auto t_end = std::chrono::high_resolution_clock::now();
        t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text != nullptr) {
            result += text;
        }
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);
            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= static_cast<float>(prob_n);
    }
    else {
        prob = 0.0f;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = t_end - t_start;
    if (duration.count() < 0) {
        t_ms = 0;
    }
    else {
        t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    return result;
}

// [11.2] Фильтрация галлюцинаций Whisper
// WHY: Проверяет текст на мусорные фразы и отбрасывает его.
static bool is_hallucination(const std::string& text) {
    // Точные совпадения — полный мусор
    static const std::unordered_set<std::string> exact_matches = {
        "", "!", ".", "Sil", "Bye", "Okay", "Okay.", "Thank you.", "Thank you",
        "Thanks.", "Bye.", "Thank you for listening.", "К", "Спасибо", "Пока",
        "Звук!", "Р", "You're", "you're", "You're not", "See?", "you", "You",
        "Yeah", "Well", "Hey", "Oh", "Right", "Real", "Huh", "I", "I'm", "В",
        "а", "У-у-у!", "Привет!", "Здравствуйте!", "Пока.", "Пока!", "Спасибо.",
        "До свидания.", "Thank you guys.", "Я передам вам.",
        // [ПАТЧ] Новые мусорные фразы
        "молочи", "сочивающие", "направленный", "офигенно", 
        "трендишь", "тарантиш", "отвыка", "отвык", "пеной",
        "поднявший", "вкалывала", "минерал", "купила",
        "звук", "музыка", "шум", "тишина", "пауза", "смех",
        "аплодисменты", "кашель", "вздох", "скрип", "стук"
    };
    if (exact_matches.count(text)) return true;

    // Частичные совпадения
    static const std::vector<std::string> substrings = {
        "Редактор субтитров", "можешь это сделать", "Как дела?",
        "Добро пожаловать", "Спасибо за внимание", "Будьте здоровы",
        "Продолжение следует", "End of", "The End", "THE END",
        "I can't believe it happened.",
        "All right.", "Hello, everyone.", "The film was made", "Translated by",
        "Thanks for watching", "buzz", "What do you have to say?", "Action!",
        "\"Okay\"?",
        "Badass music", "The second part of the video", "Thank you for watching",
        "click", "Субтитры", "До новых встреч", "ПЕСНЯ", "Silence",
        "*звук!", "Пока, ребята.", "звук шума", "СТУК", "Спасибо за просмотр",
        "СПОКОЙНАЯ МУЗЫКА", "ЗВОНОК В ДВЕРЬ", "Прошу прощения.", "звук реверса",
        "Мы с вами поздравляем вас с праздником!",
        "реклама", "заставка", "интро", "аутро"
    };
    for (const auto& sub : substrings) {
        if (text.find(sub) != std::string::npos) return true;
    }
    return false;
}

// ========================================================================
// 12. ПАРСИНГ КОМАНД
// ========================================================================

// [12.1] Извлечение ключевого слова после команды «google» или «call»
std::string ParseCommandAndGetKeyword(std::string textHeardTrimmed,
    const std::string& command = "google") {
    textHeardTrimmed = StripPunctuationMarks(textHeardTrimmed);
    std::string sanitizedInput = textHeardTrimmed;
    std::size_t pos = 0;
    bool startsWithPrefix = false;

    // Слова-паразиты, которые Whisper добавляет к командам
    static const std::unordered_set<std::string> please_needles = {
        "can you hear me", "Can you hear me", "Are you here", "are you here",
        "Do you hear me", "do you hear me", "Пожалуйста", "пожалуйста",
        "Позови", "позови", "ты тут", "Ты тут", "ты здесь", "Ты здесь",
        "ты меня слышишь", "Ты меня слышишь", "ты слышишь меня", "Ты слышишь меня",
        "Hey", "hey", "please", "Please", "can you", "Can you", "let's", "Let's",
        "What do you think", "Что ты думаешь", "что ты думаешь",
        "Что ты об этом думаешь", "что ты об этом думаешь"
    };
    std::string result_param = "";
    for (const auto& prefix : please_needles) {
        sanitizedInput = string_replace_all(sanitizedInput, prefix, "");
    }
    trim(sanitizedInput);

    // Для команды «google» ищем русские префиксы
    if (command == "google") {
        static const std::unordered_set<std::string> prefixNeedles = {
            "Погугли", "погугли", "гугли", "гугл", "угли", "углe",
            "По гугле", "По угли"
        };
        for (const auto& prefix : prefixNeedles) {
            if (sanitizedInput.size() >= prefix.size() &&
                sanitizedInput.compare(0, prefix.length(), prefix) == 0) {
                size_t base = prefix.length();
                while (base < sanitizedInput.size()) {
                    unsigned char ch = static_cast<unsigned char>(sanitizedInput[base]);
                    if (std::isspace(ch) || ch == ':') { ++base; }
                    else { break; }
                }
                pos = base;
                startsWithPrefix = true;
                break;
            }
        }
    }

    // Если префикс не найден — ищем саму команду в строке
    if (!startsWithPrefix) {
        size_t found = sanitizedInput.find(command);
        if (found != std::string::npos) {
            size_t base = found + command.size();
            while (base < sanitizedInput.size() &&
                (std::isspace(static_cast<unsigned char>(sanitizedInput[base])) ||
                    sanitizedInput[base] == ':')) { ++base; }
            pos = base;
        }
        else {
            size_t foundCall = sanitizedInput.find("Call");
            if (foundCall != std::string::npos) {
                size_t base = foundCall + 4;
                while (base < sanitizedInput.size() &&
                    (std::isspace(static_cast<unsigned char>(sanitizedInput[base])) ||
                        sanitizedInput[base] == ':')) { ++base; }
                pos = base;
            }
            else { pos = 0; }
        }
    }

    // Для команды «call» — нормализация русских имён (падежи)
    if (command == "call") {
        trim(sanitizedInput);

        // Замена ю→а/я (Васю→Вася, Петю→Петя)
        if (sanitizedInput.size() >= 2) {
            bool utf8_rule_applied = false;
            const size_t len = sanitizedInput.size();
            auto safeReplace = [&](size_t p, const std::string& from,
                const std::string& to) -> bool {
                if (p + from.length() <= len) {
                    if (sanitizedInput.compare(p, from.length(), from) == 0) {
                        sanitizedInput.replace(p, from.length(), to);
                        return true;
                    }
                }
                return false;
                };
            utf8_rule_applied = safeReplace(len - 2, "\xD1\x8E", "\xD0\xB0") ||
                utf8_rule_applied;
            utf8_rule_applied = safeReplace(len - 2, "\xD1\x8E", "\xD1\x8F") ||
                utf8_rule_applied;
            if (utf8_rule_applied) {
                trim(sanitizedInput);
            }
        }

        // Regex-замены для нормализации падежей
        if (sanitizedInput.size() >= 2) {
            thread_local const std::regex re_male_genitive_ogo_ego(
                R"((.+)([оe]го)$)", std::regex_constants::icase);
            thread_local const std::regex re_male_u(
                R"((.+)у$)", std::regex_constants::icase);
            thread_local const std::regex re_male_a(
                R"((.+)а$)", std::regex_constants::icase);
            thread_local const std::regex re_male_om(
                R"((.+)ом$)", std::regex_constants::icase);
            thread_local const std::regex re_male_em(
                R"((.+)ем$)", std::regex_constants::icase);
            thread_local const std::regex re_male_yu(
                R"((.+)ю$)", std::regex_constants::icase);
            thread_local const std::regex re_male_yem(
                R"((.+)еем$)", std::regex_constants::icase);
            thread_local const std::regex re_female_e(
                R"((.+)е$)", std::regex_constants::icase);
            thread_local const std::regex re_female_oj(
                R"((.+)ой$)", std::regex_constants::icase);
            thread_local const std::regex re_female_y(
                R"((.+)ы$)", std::regex_constants::icase);
            thread_local const std::regex re_female_i(
                R"((.+)и$)", std::regex_constants::icase);
            thread_local const std::regex re_female_ej(
                R"((.+)ей$)", std::regex_constants::icase);
            thread_local const std::regex re_female_yu(
                R"((.+)ю$)", std::regex_constants::icase);
            thread_local const std::regex re_female_instr_lyubov(
                R"((.+)ью$)", std::regex_constants::icase);
            thread_local const std::regex re_female_dat_lyubov(
                R"((.+)и$)", std::regex_constants::icase);

            sanitizedInput = std::regex_replace(sanitizedInput,
                re_male_genitive_ogo_ego, "$1");
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
            sanitizedInput = std::regex_replace(sanitizedInput,
                re_female_instr_lyubov, "$1ь");
            sanitizedInput = std::regex_replace(sanitizedInput,
                re_female_dat_lyubov, "$1ь");
        }
        trim(sanitizedInput);
        textHeardTrimmed = sanitizedInput;
    }

    result_param = textHeardTrimmed.substr(pos);
    return result_param;
}

// ========================================================================
// 13. ПРЕОБРАЗОВАНИЕ URL В РЕЧЬ
// ========================================================================

// [13.1] Преобразование URL в читаемый текст для TTS
std::string url_to_speech(const std::string& url) {
    std::string result;
    std::string clean_url = url;
    clean_url = string_replace_all(clean_url, "https://", "");
    clean_url = string_replace_all(clean_url, "http://", "");
    clean_url = string_replace_all(clean_url, "www.", "");
    size_t qpos = clean_url.find('?');
    if (qpos != std::string::npos) clean_url = clean_url.substr(0, qpos);
    size_t hpos = clean_url.find('#');
    if (hpos != std::string::npos) clean_url = clean_url.substr(0, hpos);
    if (!clean_url.empty() && clean_url.back() == '/') {
        clean_url.pop_back();
    }
    std::vector<std::string> parts;
    std::string current;
    for (char c : clean_url) {
        if (c == '.' || c == '/' || c == '-' || c == ' ') {
            if (!current.empty()) { parts.push_back(current); current.clear(); }
            if (c == '.') parts.push_back("dot");
            else if (c == '/') parts.push_back("slash");
            else if (c == '-') parts.push_back("dash");
            else if (c == ' ') parts.push_back("underscore");
        }
        else { current += c; }
    }
    if (!current.empty()) parts.push_back(current);
    for (size_t i = 0; i < parts.size(); ++i) {
        if (!result.empty()) result += " ";
        result += parts[i];
    }
    return result;
}

// ========================================================================
// 14. ОЧИСТКА ТЕКСТА ДЛЯ TTS
// ========================================================================

// [14.1] Удаляет маркеры эмоций, многоточия и спецсимволы
static std::string clean_text_for_tts(const std::string& text) {
    std::string result = text;
    try {
        result = std::regex_replace(result, std::regex(R"(\*[^*]+\*)"), " ");
        result = std::regex_replace(result, std::regex(R"(\[[^\]]+\])"), " ");
        result = std::regex_replace(result, std::regex(R"(\.{2,})"), ". ");
        result = std::regex_replace(result, std::regex(R"([#*_~`>])"), " ");
        result = std::regex_replace(result, std::regex(R"(\s+)"), " ");
        result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
        result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load()) {
            fprintf(stderr, "[clean_text_for_tts] regex error: %s\n", e.what());
        }
    }
    return result;
}

// ========================================================================
// 15. АСИНХРОННАЯ ОТПРАВКА ТЕКСТА В TTS
// ========================================================================

// [15.1] Отправляет текст в XTTS-сервер через HTTP POST с JSON-телом.
// Выполняется в отдельном потоке, чтобы не блокировать генерацию LLaMA.
void send_tts_async(std::string text,
    std::string speaker_wav = "Эмма",
    std::string language = "ru",
    std::string tts_url = "http://localhost:8020/") {
    if (text.empty()) {
        return;
    }

    // Очистка от маркеров эмоций и мусора
    text = clean_text_for_tts(text);
    if (text.empty()) return;

    // [Защита числовых паттернов]
    std::vector<std::pair<std::string, std::string>> protected_patterns;

    static const std::regex re_time(
        R"(\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?\b)",
        std::regex::ECMAScript);
    static const std::regex re_date_dots(
        R"(\b(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\.(\d{4})\b)",
        std::regex::ECMAScript);
    static const std::regex re_date_iso(
        R"(\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b)",
        std::regex::ECMAScript);
    static const std::regex re_date_slash(
        R"(\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b)",
        std::regex::ECMAScript);
    static const std::regex re_decimal(
        R"(\b\d+[.,]\d+\b(?![\w-]))", std::regex::ECMAScript);
    static const std::regex re_percent(
        R"(\b\d+(?:[.,]\d+)?\s*%)", std::regex::ECMAScript);
    static const std::regex re_phone(
        R"(\+?[\d\s\-\(\)]{7,})", std::regex::ECMAScript);
    static const std::regex re_url(
        R"(https?://[^\s]+)", std::regex::ECMAScript);
    static const std::regex re_email(
        R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        std::regex::ECMAScript);

    // Защита времени: 15:30, 15:30:45
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_time);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "TIME" +
                std::to_string(protected_patterns.size()) + "TIME";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (time protection): %s\n", e.what());
    }

    // Защита дат: 31.12.2025
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_dots);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "DATE" +
                std::to_string(protected_patterns.size()) + "DATE";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (date dots): %s\n", e.what());
    }

    // Защита дат ISO: 2025-12-31
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_iso);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "DATE" +
                std::to_string(protected_patterns.size()) + "DATE";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (date iso): %s\n", e.what());
    }

    // Защита дат slash: 12/31/2025
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_date_slash);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "DATE" +
                std::to_string(protected_patterns.size()) + "DATE";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (date slash): %s\n", e.what());
    }

    // Защита десятичных дробей: 3.14, 0,5
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_decimal);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "DEC" +
                std::to_string(protected_patterns.size()) + "DEC";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (decimal): %s\n", e.what());
    }

    // Защита процентов: 50%, 12.5%
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_percent);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "PCT" +
                std::to_string(protected_patterns.size()) + "PCT";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (percent): %s\n", e.what());
    }

    // Защита валют: 100$, $50, 1000 руб.
    static const std::regex re_currency(
        R"(\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*[$€£¥₽]|\b[$€£¥₽]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)",
        std::regex::ECMAScript);
    try {
        std::string processed;
        auto it_currency = std::sregex_iterator(text.begin(), text.end(), re_currency);
        auto end_currency = std::sregex_iterator();
        size_t last_pos = 0;
        for (auto i = it_currency; i != end_currency; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "CUR" +
                std::to_string(protected_patterns.size()) + "CUR";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (currency): %s\n", e.what());
    }

    // Защита дробей: 1/2, 1 1/2
    static const std::regex re_fraction(
        R"(\b\d+\s*/\s*\d+\b|\b\d+\s+\d+\s*/\s*\d+\b)",
        std::regex::ECMAScript);
    try {
        std::string processed;
        auto it_fraction = std::sregex_iterator(text.begin(), text.end(), re_fraction);
        auto end_fraction = std::sregex_iterator();
        size_t last_pos = 0;
        for (auto i = it_fraction; i != end_fraction; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "FRAC" +
                std::to_string(protected_patterns.size()) + "FRAC";
            std::string frac_value = match.str();
            frac_value = std::regex_replace(frac_value,
                std::regex(R"(\s+)"), " ");
            frac_value = std::regex_replace(frac_value,
                std::regex(R"(\s*/\s*)"), "/");
            protected_patterns.emplace_back(marker, frac_value);
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (fraction): %s\n", e.what());
    }

    // Защита телефонов: +7 (123) 456-78-90
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_phone);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "PHONE" +
                std::to_string(protected_patterns.size()) + "PHONE";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (phone): %s\n", e.what());
    }

    // Защита URL
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_url);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "URL" +
                std::to_string(protected_patterns.size()) + "URL";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (URL): %s\n", e.what());
    }

    // Защита email
    try {
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_email);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "EMAIL" +
                std::to_string(protected_patterns.size()) + "EMAIL";
            protected_patterns.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (email): %s\n", e.what());
    }

    // [Защита точек] IP, аббревиатуры, инициалы
    std::vector<std::pair<std::string, std::string>> protected_dots;
    try {
        static const std::regex re_ip(
            R"(\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b)",
            std::regex::ECMAScript);
        std::string processed;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re_ip);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            processed += text.substr(last_pos, match.position() - last_pos);
            std::string marker = "IP" +
                std::to_string(protected_dots.size()) + "IP";
            protected_dots.emplace_back(marker, match.str());
            processed += marker;
            last_pos = match.position() + match.length();
        }
        processed += text.substr(last_pos);
        text = processed;

        // Алгоритмическая защита остальных точек
        struct DotInfo {
            size_t pos;
            std::string before;
            std::string after;
        };
        std::vector<DotInfo> dots;
        for (size_t i = 0; i < text.length(); ++i) {
            if (text[i] == '.') {
                DotInfo info;
                info.pos = i;
                size_t start = (i > 20) ? i - 20 : 0;
                info.before = text.substr(start, i - start);
                info.after = text.substr(i + 1,
                    std::min((size_t)10, text.length() - i - 1));
                dots.push_back(info);
            }
        }
        std::string new_text;
        last_pos = 0;
        for (const auto& dot : dots) {
            bool protect = false;

            // Критерий 1: часть числа (дата, версия)
            if (!dot.before.empty() && !dot.after.empty()) {
                char prev_char = dot.before.back();
                char next_char = dot.after[0];
                if (std::isdigit(static_cast<unsigned char>(prev_char)) &&
                    std::isdigit(static_cast<unsigned char>(next_char))) {
                    protect = true;
                }
            }

            // Критерий 2: часть аббревиатуры — «т.д.», «P.S.»
            if (!protect && !dot.before.empty() && !dot.after.empty()) {
                char prev_char = dot.before.back();
                if (std::isalpha(static_cast<unsigned char>(prev_char))) {
                    size_t after_pos = 0;
                    while (after_pos < dot.after.length() &&
                        std::isspace(static_cast<unsigned char>(
                            dot.after[after_pos]))) {
                        after_pos++;
                    }
                    if (after_pos < dot.after.length() &&
                        std::isalpha(static_cast<unsigned char>(
                            dot.after[after_pos]))) {
                        protect = true;
                    }
                }
            }

            // Критерий 3: инициалы — «А.С. Пушкин»
            if (!protect && !dot.before.empty()) {
                size_t next_dot = text.find('.', dot.pos + 1);
                if (next_dot != std::string::npos &&
                    next_dot - dot.pos <= 4) {
                    bool valid = true;
                    for (size_t j = dot.pos + 1; j < next_dot; ++j) {
                        if (!std::isalpha(static_cast<unsigned char>(text[j])) &&
                            !std::isspace(static_cast<unsigned char>(text[j]))) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) protect = true;
                }
            }

            // Критерий 4: после точки нет заглавной буквы
            if (!protect && !dot.after.empty()) {
                size_t after_pos = 0;
                while (after_pos < dot.after.length() &&
                    (std::isspace(static_cast<unsigned char>(
                        dot.after[after_pos])) ||
                        dot.after[after_pos] == ')' ||
                        dot.after[after_pos] == ']' ||
                        dot.after[after_pos] == '}' ||
                        dot.after[after_pos] == '"' ||
                        dot.after[after_pos] == '\'')) {
                    after_pos++;
                }
                if (after_pos < dot.after.length()) {
                    char next_char = dot.after[after_pos];
                    if (std::islower(static_cast<unsigned char>(next_char)) ||
                        std::isdigit(static_cast<unsigned char>(next_char)) ||
                        next_char == '.' || next_char == ',' ||
                        next_char == ';' || next_char == ':' ||
                        next_char == '?' || next_char == '!') {
                        protect = true;
                    }
                }
            }

            // Критерий 5: после точки сразу запятая или другой знак
            if (!protect && !dot.after.empty()) {
                char next_char = dot.after[0];
                if (next_char == ',' || next_char == ';' ||
                    next_char == ':' || next_char == ')' ||
                    next_char == ']' || next_char == '}') {
                    protect = true;
                }
            }

            new_text += text.substr(last_pos, dot.pos - last_pos);
            if (protect) {
                std::string marker = "DOT" +
                    std::to_string(protected_dots.size()) + "DOT";
                protected_dots.emplace_back(marker, ".");
                new_text += marker;
            }
            else {
                new_text += ".";
            }
            last_pos = dot.pos + 1;
        }
        new_text += text.substr(last_pos);
        text = new_text;
    }
    catch (const std::exception& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Error in dot protection: %s\n", e.what());
    }

    // [Базовая очистка]
    try {
        static const std::regex re_newline(R"(\r\n|\r|\n)",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_newline, " ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (newline): %s\n", e.what());
        text = string_replace_all(text, "\r\n", " ");
        text = string_replace_all(text, "\r", " ");
        text = string_replace_all(text, "\n", " ");
    }
    trim(text);
    if (text.empty()) return;

    // Удаление HTML-тегов
    try {
        static const std::regex re_html_tag(R"(<[^>]*>)",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_html_tag, " ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (HTML): %s\n", e.what());
        text = string_replace_all(text, "<", " ");
        text = string_replace_all(text, ">", " ");
    }

    // Декодирование HTML-сущностей
    text = string_replace_all(text, "&nbsp;", " ");
    text = string_replace_all(text, "&amp;", "&");
    text = string_replace_all(text, "&lt;", "<");
    text = string_replace_all(text, "&gt;", ">");
    text = string_replace_all(text, "&quot;", "\"");
    text = string_replace_all(text, "&#39;", "'");
    text = string_replace_all(text, "&apos;", "'");
    text = string_replace_all(text, "&#34;", "\"");
    text = string_replace_all(text, "&rsquo;", "'");
    text = string_replace_all(text, "&lsquo;", "'");
    text = string_replace_all(text, "&rdquo;", "\"");
    text = string_replace_all(text, "&ldquo;", "\"");
    text = string_replace_all(text, "&mdash;", "-");
    text = string_replace_all(text, "&ndash;", "-");
    text = string_replace_all(text, "&hellip;", "...");

    // Защита от бесконечных замен
    int max_iterations = 10;
    for (int iter = 0; iter < max_iterations; iter++) {
        std::string prev_text = text;
        text = string_replace_all(text, "&amp;lt;", "<");
        text = string_replace_all(text, "&amp;gt;", ">");
        text = string_replace_all(text, "&amp;quot;", "\"");
        text = string_replace_all(text, "&amp;amp;", "&");
        if (text == prev_text) break;
    }
    trim(text);
    if (text.empty()) return;

    // Обработка «умных» кавычек и тире
    text = string_replace_all(text, "\xE2\x80\x9C", "\"");
    text = string_replace_all(text, "\xE2\x80\x9D", "\"");
    text = string_replace_all(text, "\xE2\x80\x98", "'");
    text = string_replace_all(text, "\xE2\x80\x99", "'");
    text = string_replace_all(text, "\xE2\x80\x93", "-");
    text = string_replace_all(text, "\xE2\x80\x94", "-");
    text = string_replace_all(text, "\xC2\xA0", " ");
    text = string_replace_all(text, "\xE2\x80\xA6", "...");
    trim(text);
    if (text.empty()) return;

    // [Нормализация эмоций и выделений]
    try {
        {
            static const std::regex re_double_star(R"(\*\*([^*]+)\*\*)",
                std::regex::ECMAScript);
            text = std::regex_replace(text, re_double_star, "$1,");
        }
        {
            static const std::regex re_star(R"(\*([^*]+)\*)",
                std::regex::ECMAScript);
            text = std::regex_replace(text, re_star, "$1,");
        }
        {
            static const std::regex re_parens(R"(\(([^)]+)\))",
                std::regex::ECMAScript);
            text = std::regex_replace(text, re_parens, "$1,");
        }
        {
            static const std::regex re_brackets(R"(\[([^\]]+)\])",
                std::regex::ECMAScript);
            text = std::regex_replace(text, re_brackets, "$1,");
        }
        text = string_replace_all(text, "*", "");
        text = string_replace_all(text, "(", "");
        text = string_replace_all(text, ")", "");
        text = string_replace_all(text, "[", "");
        text = string_replace_all(text, "]", "");

        static const std::regex re_comma_space(R"(,([^\s!?.,:;]))",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_comma_space, ", $1");

        static const std::regex re_double_space("  +",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_double_space, " ");

        if (!text.empty() && text[0] == ',') {
            text.erase(0, 1);
            if (!text.empty() && text[0] == ' ') {
                text.erase(0, 1);
            }
        }
        {
            static const std::regex re_spaces(R"(\s+)",
                std::regex::ECMAScript);
            text = std::regex_replace(text, re_spaces, " ");
        }
        trim(text);
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (emotion): %s\n", e.what());
        text = string_replace_all(text, "*", " ");
        text = string_replace_all(text, "(", " ");
        text = string_replace_all(text, ")", " ");
        text = string_replace_all(text, "[", " ");
        text = string_replace_all(text, "]", " ");
        text = string_replace_all(text, "  ", " ");
        trim(text);
    }

    // [Удаление Markdown]
    try {
        static const std::regex re_code_block(R"(```(.*?)```)",
            std::regex::ECMAScript);
        static const std::regex re_code_inline(R"(`([^`]*)`)",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_code_block, "$1");
        text = std::regex_replace(text, re_code_inline, "$1");

        static const std::regex re_bold2(R"(__([^_]+)__)",
            std::regex::ECMAScript);
        static const std::regex re_ital2(R"(_([^_]+)_)",
            std::regex::ECMAScript);
        static const std::regex re_del(R"(~~([^~]+)~~)",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_bold2, "$1");
        text = std::regex_replace(text, re_ital2, "$1");
        text = std::regex_replace(text, re_del, "$1");

        static const std::regex re_multi_unders(R"(_{2,})",
            std::regex::ECMAScript);
        static const std::regex re_multi_tildes(R"(~{2,})",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_multi_unders, " ");
        text = std::regex_replace(text, re_multi_tildes, " ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (Markdown): %s\n", e.what());
        text = string_replace_all(text, "```", " ");
        text = string_replace_all(text, "`", " ");
        text = string_replace_all(text, "__", " ");
        text = string_replace_all(text, "~~", " ");
    }
    trim(text);
    if (text.empty()) return;

    // [Удаление маркеров списков]
    try {
        static const std::regex re_list_markers(
            R"(^\s*(\d+[\.\)]|[A-Za-zА-Яа-яЁё][\.\)]|[\-\*\+\>\|#]+)\s*)",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_list_markers, "");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (list markers): %s\n", e.what());
        if (text.size() > 2) {
            if (text[0] == '-' || text[0] == '*' ||
                text[0] == '+' || text[0] == '#') {
                if (text[1] == ' ') text = text.substr(2);
            }
            else if (std::isdigit(static_cast<unsigned char>(text[0]))) {
                size_t i = 1;
                while (i < text.size() &&
                    std::isdigit(static_cast<unsigned char>(text[i]))) i++;
                if (i < text.size() && (text[i] == '.' || text[i] == ')')) {
                    if (i + 1 < text.size() && text[i + 1] == ' ') {
                        text = text.substr(i + 2);
                    }
                    else {
                        text = text.substr(i + 1);
                    }
                }
            }
        }
    }
    trim(text);
    if (text.empty()) return;

    // [Обработка кавычек и сокращений]
    try {
        auto make_utf8 = [](std::initializer_list<unsigned int> bytes) -> std::string {
            std::string s;
            s.reserve(bytes.size());
            for (unsigned int b : bytes) {
                s.push_back(static_cast<char>(b));
            }
            return s;
        };

        std::vector<std::pair<std::string, std::string>> saved_contractions;
        static const std::regex re_contractions("\\b\\w+'\\w+\\b",
            std::regex::ECMAScript);
        std::string protected_text;
        auto words_begin = std::sregex_iterator(text.begin(), text.end(),
            re_contractions);
        auto words_end = std::sregex_iterator();
        size_t last_pos = 0;
        for (auto i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            protected_text += text.substr(last_pos,
                match.position() - last_pos);
            std::string marker = "@@CONTR" +
                std::to_string(saved_contractions.size()) + "@@";
            saved_contractions.push_back({ marker, match.str() });
            protected_text += marker;
            last_pos = match.position() + match.length();
        }
        protected_text += text.substr(last_pos);
        text = protected_text;

        static const std::regex re_quotes_double("\"([^\"]*)\"",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_double, "$1");

        static const std::regex re_quotes_single("'([^']*)'",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_single, "$1");

        static const std::string Q_LAQUO   = make_utf8({0xC2, 0xAB});
        static const std::string Q_RAQUO   = make_utf8({0xC2, 0xBB});
        static const std::string Q_LDQUO   = make_utf8({0xE2, 0x80, 0x9C});
        static const std::string Q_RDQUO   = make_utf8({0xE2, 0x80, 0x9D});
        static const std::string Q_LSQUO   = make_utf8({0xE2, 0x80, 0x98});
        static const std::string Q_RSQUO   = make_utf8({0xE2, 0x80, 0x99});
        static const std::string Q_LDBL9   = make_utf8({0xE2, 0x80, 0x9E});
        static const std::string Q_LSGL9   = make_utf8({0xE2, 0x80, 0x9A});
        static const std::string Q_LSAQUO  = make_utf8({0xE2, 0x80, 0xB9});
        static const std::string Q_RSAQUO  = make_utf8({0xE2, 0x80, 0xBA});
        static const std::string Q_LCORNER = make_utf8({0xE3, 0x80, 0x8C});
        static const std::string Q_RCORNER = make_utf8({0xE3, 0x80, 0x8D});
        static const std::string Q_LDLCORN = make_utf8({0xE3, 0x80, 0x8E});
        static const std::string Q_RDLCORN = make_utf8({0xE3, 0x80, 0x8F});

        static const std::regex re_quotes_angle1(
            Q_LAQUO + "(.*?)" + Q_RAQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_angle1, "$1");

        static const std::regex re_quotes_angle2(
            Q_RAQUO + "(.*?)" + Q_LAQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_angle2, "$1");

        static const std::regex re_quotes_german_double(
            Q_LDBL9 + "(.*?)" + Q_LDQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_german_double, "$1");

        static const std::regex re_quotes_german_single(
            Q_LSGL9 + "(.*?)" + Q_RSQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_german_single, "$1");

        static const std::regex re_quotes_french_double(
            Q_LSAQUO + "(.*?)" + Q_RSAQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_french_double, "$1");

        static const std::regex re_quotes_french_single(
            Q_RSAQUO + "(.*?)" + Q_LSAQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_french_single, "$1");

        static const std::regex re_quotes_jp_double(
            Q_LCORNER + "(.*?)" + Q_RCORNER, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_jp_double, "$1");

        static const std::regex re_quotes_jp_single(
            Q_LDLCORN + "(.*?)" + Q_RDLCORN, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_jp_single, "$1");

        static const std::regex re_quotes_polish(
            Q_LDBL9 + "(.*?)" + Q_RDQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_polish, "$1");

        static const std::regex re_quotes_swedish_double(
            Q_LDQUO + "(.*?)" + Q_RDQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_swedish_double, "$1");

        static const std::regex re_quotes_swedish_single(
            Q_LSQUO + "(.*?)" + Q_RSQUO, std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_swedish_single, "$1");

        text = string_replace_all(text, Q_LAQUO, "");
        text = string_replace_all(text, Q_RAQUO, "");
        text = string_replace_all(text, Q_LDBL9, "");
        text = string_replace_all(text, Q_LDQUO, "");
        text = string_replace_all(text, Q_LSGL9, "");
        text = string_replace_all(text, Q_RSQUO, "");
        text = string_replace_all(text, Q_LSAQUO, "");
        text = string_replace_all(text, Q_RSAQUO, "");
        text = string_replace_all(text, Q_LCORNER, "");
        text = string_replace_all(text, Q_RCORNER, "");
        text = string_replace_all(text, Q_LDLCORN, "");
        text = string_replace_all(text, Q_RDLCORN, "");
        text = string_replace_all(text, Q_RDQUO, "");
        text = string_replace_all(text, Q_LSQUO, "");

        for (const auto& p : saved_contractions) {
            text = string_replace_all(text, p.first, p.second);
        }

        static const std::regex re_spaces("\\s+", std::regex::ECMAScript);
        text = std::regex_replace(text, re_spaces, " ");
        trim(text);
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (quotes): %s\n", e.what());
        auto fb = [](std::initializer_list<unsigned int> bytes) -> std::string {
            std::string s;
            for (unsigned int b : bytes) s.push_back(static_cast<char>(b));
            return s;
        };
        text = string_replace_all(text, fb({0xC2, 0xAB}), "");
        text = string_replace_all(text, fb({0xC2, 0xBB}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x9E}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x9C}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0xB9}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0xBA}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x9D}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x99}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x98}), "");
        text = string_replace_all(text, fb({0xE2, 0x80, 0x9A}), "");

        static const std::regex re_spaces_fallback("\\s+",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_spaces_fallback, " ");
        trim(text);
    }

    // [Ссылки]
    try {
        static const std::regex re_link_md(
            R"(\[([^\]]*)\]\(([^)\s]+)\))", std::regex::ECMAScript);
        static const std::regex re_bare_url(
            R"(https?://[^\s<>]+|www\.[^\s<>]+)", std::regex::ECMAScript);

        std::string result1;
        auto it1 = std::sregex_iterator(text.begin(), text.end(), re_link_md);
        auto end1 = std::sregex_iterator();
        size_t last_pos = 0;
        for (auto i = it1; i != end1; ++i) {
            std::smatch match = *i;
            result1 += text.substr(last_pos, match.position() - last_pos);
            std::string link_text = match[1].str();
            std::string url = match[2].str();
            if (link_text.length() > 2 &&
                link_text != "ссылка" && link_text != "link") {
                result1 += link_text + ", ";
            }
            else {
                result1 += url_to_speech(url) + ", ";
            }
            last_pos = match.position() + match.length();
        }
        result1 += text.substr(last_pos);
        text = result1;

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
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (links): %s\n", e.what());
        text = string_replace_all(text, "[", " ");
        text = string_replace_all(text, "]", " ");
        text = string_replace_all(text, "(", " ");
        text = string_replace_all(text, ")", " ");
    }

    // [Изображения]
    try {
        static const std::regex re_img_md(
            R"(!\[([^\]]*)\]\([^)]+\))", std::regex::ECMAScript);
        text = std::regex_replace(text, re_img_md, "$1, ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (images): %s\n", e.what());
    }

    // [Удаление фигурных скобок]
    try {
        static const std::regex re_curly(R"(\{[^{}]*\})",
            std::regex::ECMAScript);
        bool changed = true;
        int max_iterations_local = 100;
        int iteration = 0;
        while (changed && iteration < max_iterations_local) {
            changed = false;
            iteration++;
            std::string t1 = std::regex_replace(text, re_curly, " ");
            if (t1 != text) {
                text.swap(t1);
                changed = true;
            }
        }
        if (iteration >= max_iterations_local && g_verbose_mode.load()) {
            fprintf(stderr,
                "Warning: Too many iterations removing curly braces\n");
        }
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (curly): %s\n", e.what());
        text = string_replace_all(text, "{", " ");
        text = string_replace_all(text, "}", " ");
    }
    trim(text);
    if (text.empty()) return;

    // [Удаление мусорных символов]
    try {
        static const std::regex re_noise(R"([#\|\\])",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_noise, " ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (noise): %s\n", e.what());
        text = string_replace_all(text, "#", " ");
        text = string_replace_all(text, "|", " ");
        text = string_replace_all(text, "\\", " ");
    }
    trim(text);
    if (text.empty()) return;

    // [XTTS-специфичные замены]
    text = string_replace_all(text, ";", ",");
    text = string_replace_all(text, "\"", "");

    try {
        static const std::regex re_space_before_excl(R"(\s+(!))",
            std::regex::ECMAScript);
        static const std::regex re_space_before_ques(R"(\s+(\?))",
            std::regex::ECMAScript);
        static const std::regex re_space_before_dot(R"(\s+(\.))",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_space_before_excl, "$1");
        text = std::regex_replace(text, re_space_before_ques, "$1");
        text = std::regex_replace(text, re_space_before_dot, "$1");
    }
    catch (const std::regex_error& e) {
        text = string_replace_all(text, " !", "!");
        text = string_replace_all(text, " ?", "?");
        text = string_replace_all(text, " .", ".");
    }

    // [Схлопывание повторов знаков препинания]
    try {
        static const std::regex re_bangs(R"(!{2,})",
            std::regex::ECMAScript);
        static const std::regex re_qmarks(R"(\?{2,})",
            std::regex::ECMAScript);
        text = std::regex_replace(text, re_bangs, "!");
        text = std::regex_replace(text, re_qmarks, "?");

        text = string_replace_all(text, ". ,", ". ");
        text = string_replace_all(text, "! ,", "! ");
        text = string_replace_all(text, "? ,", "? ");

        while (text.find(", ,") != std::string::npos) {
            text = string_replace_all(text, ", ,", ", ");
        }
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (punctuation): %s\n", e.what());
    }
    trim(text);
    if (text.empty()) return;

    // [Нормализация пробелов]
    try {
        static const std::regex re_spaces(R"(\s+)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_spaces, " ");
    }
    catch (const std::regex_error& e) {
        if (g_verbose_mode.load())
            fprintf(stderr, "Regex error (spaces): %s\n", e.what());
        std::string temp;
        bool last_was_space = false;
        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!last_was_space) { temp += ' '; last_was_space = true; }
            }
            else {
                temp += c;
                last_was_space = false;
            }
        }
        text = temp;
    }
    trim(text);
    if (text.empty()) return;

    // [Восстановление защищённых паттернов]
    for (const auto& p : protected_patterns) {
        text = string_replace_all(text, p.first, p.second);
    }
    for (const auto& p : protected_dots) {
        text = string_replace_all(text, p.first, p.second);
    }

    // [Удаление префикса спикера]
    std::string prefix = speaker_wav + ":";
    if (text.size() >= prefix.size() && text.find(prefix) == 0) {
        size_t pos = prefix.size();
        if (pos < text.length() && text[pos] == ' ') {
            pos++;
        }
        if (pos <= text.length()) {
            text = text.substr(pos);
            trim(text);
        }
    }

    // [Нормализация имени спикера]
    speaker_wav = string_replace_all(speaker_wav, ":", "");
    speaker_wav = string_replace_all(speaker_wav, "\\", "");
    speaker_wav = string_replace_all(speaker_wav, "\r", "");
    speaker_wav = string_replace_all(speaker_wav, "\"", "");
    speaker_wav = string_replace_all(speaker_wav, "/", "_");
    speaker_wav = string_replace_all(speaker_wav, "<", "_");
    speaker_wav = string_replace_all(speaker_wav, ">", "_");
    speaker_wav = string_replace_all(speaker_wav, "|", "_");
    speaker_wav = string_replace_all(speaker_wav, "?", "_");
    speaker_wav = string_replace_all(speaker_wav, "*", "_");
    trim(speaker_wav);
    if (speaker_wav.size() < 2) speaker_wav = "default";

    // Финальная зачистка
    text = string_replace_all(text, "\r\n", " ");
    text = string_replace_all(text, "\r", " ");
    text = string_replace_all(text, "\n", " ");
    trim(text);
    if (text.empty()) return;

    // [Отправка JSON-запроса в XTTS]
    auto escape_json = [](const std::string& s) -> std::string {
        std::string result;
        result.reserve(s.size());
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
                if (c >= 32 && c != 127)
                    result += static_cast<char>(c);
                else {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned int>(c));
                    result += buf;
                }
            }
        }
        return result;
    };

    std::string data = "{\"text\":\"" + escape_json(text) + "\", "
        "\"language\":\"" + escape_json(language) + "\", "
        "\"speaker_wav\":\"" + escape_json(speaker_wav) + "\"}";

    std::string full_url = tts_url + "tts_to_audio/";

    CURL* http_handle = curl_easy_init();
    if (http_handle) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(http_handle, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(http_handle, CURLOPT_CONNECTTIMEOUT, 3L);
        curl_easy_setopt(http_handle, CURLOPT_FAILONERROR, 1L);

        // Callback прогресса для отмены запросов при barge-in
        curl_easy_setopt(http_handle, CURLOPT_XFERINFOFUNCTION,
            progress_callback);
        curl_easy_setopt(http_handle, CURLOPT_XFERINFODATA, nullptr);
        curl_easy_setopt(http_handle, CURLOPT_NOPROGRESS, 0L);

        curl_easy_setopt(http_handle, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(http_handle, CURLOPT_URL, full_url.c_str());
        curl_easy_setopt(http_handle, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);

        std::string responseData;
        curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);
        curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);

        CURLcode res = curl_easy_perform(http_handle);
        (void)res;

        curl_slist_free_all(headers);
        curl_easy_cleanup(http_handle);
    }
    else {
        if (g_verbose_mode.load()) {
            fprintf(stderr, "Failed to initialize cURL handle\n");
        }
    }
}

// ========================================================================
// 16. ВВОД С КЛАВИАТУРЫ
// ========================================================================

// [16.1] Поток для чтения пользовательского ввода с клавиатуры
void input_thread_func() {
    std::string line;
    std::string buffer;
    bool found_another_line = true;

    while (keyboard_input_running) {
        do {
            found_another_line = console::readline(line, true);
            buffer += line;
            if (!line.empty() && line.back() == '\n') {
                break;
            }
        } while (found_another_line);

        trim(buffer);
        if (!buffer.empty()) {
            std::lock_guard<std::mutex> lock(input_mutex);
            input_queue.push(buffer);
            buffer = "";
        }
    }
}

// ========================================================================
// 17. ГОРЯЧИЕ КЛАВИШИ
// ========================================================================

// [17.1] Проверка, фокусировано ли окно консоли (только Windows)
bool IsConsoleWindowFocused() {
#ifdef _WIN32
    HWND console_window = GetConsoleWindow();
    if (console_window == NULL) {
        return false;
    }
    HWND foreground_window = GetForegroundWindow();
    if (foreground_window == NULL) {
        return false;
    }
    return (console_window == foreground_window);
#else
    return true;
#endif
}

// [17.2] Функция обработки горячих клавиш
void keyboard_shortcut_func() {
#ifdef _WIN32
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

    {
        std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
        g_hotkey_pressed = "";
    }

    while (g_shortcut_thread_running.load()) {
        isFocused = IsConsoleWindowFocused();
        if (isFocused) {
            b_ctr_space = (GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                (GetAsyncKeyState(VK_SPACE) & 0x8000);
            b_ctr_right = (GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                (GetAsyncKeyState(VK_RIGHT) & 0x8000);
            b_ctr_delete = (GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                (GetAsyncKeyState(VK_DELETE) & 0x8000);
            b_ctr_r = (GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                (GetAsyncKeyState('R') & 0x8000);
            b_alt = GetAsyncKeyState(VK_MENU) & 0x8000;

            // Обработка Alt (Push-to-Talk)
            if (b_alt) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
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
                    printf("\b");
                    fflush(stdout);
                    printf("\n[Stop]\n");
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

            // Обработка Ctrl+Right (Regenerate)
            if (b_ctr_right && !b_ctr_right_prev) {
                if (!b_ctr_right_processed) {
                    fflush(stdout);
                    printf("\b");
                    fflush(stdout);
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    g_hotkey_pressed = "Ctrl+Right";
                    b_ctr_right_processed = true;
                }
            }
            else if (!b_ctr_right && b_ctr_right_prev && b_ctr_right_processed) {
                b_ctr_right_processed = false;
            }

            // Обработка Ctrl+Delete (Delete)
            if (b_ctr_delete && !b_ctr_delete_prev) {
                if (!b_ctr_delete_processed) {
                    fflush(stdout);
                    printf("\b");
                    fflush(stdout);
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    g_hotkey_pressed = "Ctrl+Delete";
                    b_ctr_delete_processed = true;
                }
            }
            else if (!b_ctr_delete && b_ctr_delete_prev && b_ctr_delete_processed) {
                b_ctr_delete_processed = false;
            }

            // Обработка Ctrl+R (Reset)
            if (b_ctr_r && !b_ctr_r_prev) {
                if (!b_ctr_r_processed) {
                    fflush(stdout);
                    printf("\b\b");
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
#else
    // На Linux горячие клавиши не поддерживаются
    while (g_shortcut_thread_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
#endif
}

// ========================================================================
// КОНЕЦ ЧАСТИ 2/3
// ПРОДОЛЖЕНИЕ В ЧАСТИ 3/3
// ========================================================================

// ========================================================================
// ЧАСТЬ 3/3 — АУДИО-ПОТОК, ОСНОВНАЯ ФУНКЦИЯ run(), ТОЧКИ ВХОДА
// ========================================================================

// ========================================================================
// 18. ПЕРЕРИСОВКА СТРОКИ ПОЛЬЗОВАТЕЛЯ (ЧАТОВЫЙ РЕЖИМ)
// ========================================================================

// [18.1] Единая функция для обновления строки пользователя в консоли
// WHY: Централизует отрисовку, добавляет цвета и статусы в конце строки.
static void redraw_user_message(const std::string& person_name,
                                const std::string& text,
                                const std::string& status) {
    std::lock_guard<std::mutex> lock(g_console_mutex);
    // \033[32m — зелёный, \033[90m — серый, \033[K — очистка до конца строки
    printf("\r\033[32m%s \xe2\x86\x92\033[0m %s \033[90m%s\033[0m\033[K",
        person_name.c_str(), text.c_str(), status.c_str());
    fflush(stdout);
}

// ========================================================================
// 19. ШАБЛОНЫ ПРОМПТОВ
// ========================================================================

// [19.1] Промпт для Whisper (используется как контекстная подсказка)
const std::string k_prompt_whisper_ru =
R"({1}: разговор с голосовым ассистентом. Распознавай только речь {0}.)";

const std::string k_prompt_whisper =
R"({1}: conversation with voice assistant. Recognize only {0}'s speech.)";

// [19.2] Промпт для LLaMA с запретом самодиалога
// WHY: Явно запрещаем боту генерировать диалог между пользователем и собой.
const std::string k_prompt_llama = R"({1} — дружелюбный и умный помощник. {1} отвечает кратко, по делу, только текстом. Без скобок, звёздочек и других спецсимволов.
{1} НИКОГДА не генерирует диалог между собой и {0}. {1} отвечает ТОЛЬКО на последнюю реплику {0} и заканчивает свой ответ.
{0}{4} Привет, {1}!
{1}{4} Привет! Как дела?
{0}{4} Который час?
{1}{4} Сейчас {2}.
{0}{4} Какая сегодня дата?
{1}{4} {5}, {3}.
{0}{4})";

// ========================================================================
// 20. ЕДИНЫЙ ПОТОК АУДИО-ВВОДА (ЧАТОВЫЙ РЕЖИМ v2.0)
// ========================================================================

// [20.1] Основной поток обработки аудио с микрофона
// Архитектура VAD-автомата:
//   SILENCE → vad_simple_int() == 1 → SPEECH_START → warmup
//   SPEECH  → vad_simple_int() == 0 → SPEECH_CONTINUE
//   SPEECH  → vad_simple_int() == 2 → SPEECH_END → transcribe → accumulator
//   SPEECH  → duration > max_segment → FORCE_SEGMENT → transcribe → accumulator
//
// [ПАТЧ] PARTIAL DISPLAY: каждые 3 секунды показывает промежуточный текст.
// [ПАТЧ] ДЕДУПЛИКАЦИЯ: проверяет, не дублируется ли последний сегмент.
// [ПАТЧ] МЯГКОЕ ПРЕРЫВАНИЕ: не очищает буфер TTS при barge-in.
void audio_input_thread_func(
    whisper_context* ctx_wsp,
    llama_context* ctx_llama,
    const whisper_params& params,
    audio_async& audio_ref,
    const std::string& person_name) {
    g_audio_thread_running.store(true);

    // Состояние VAD-автомата
    bool speech_active = false;
    float speech_start_ms = 0.0f;
    float last_speech_end_ms = get_current_time_ms();
    int consecutive_barge_in = 0;

    // [ПАТЧ] Partial display
    bool partial_enabled = true;
    const float partial_interval_ms = 3000.0f;   // 3 секунды между обновлениями
    float last_partial_time = 0.0f;

    // [ПАТЧ] Флаг для предотвращения мерцания [IDLE]
    bool idle_shown = false;

    // Таймаут тишины — не может быть меньше 1500 мс
    float silence_timeout_ms = params.vad_last_ms;
    if (silence_timeout_ms < 1500.0f) silence_timeout_ms = 1500.0f;

    // Принудительная разбивка: если пользователь говорит дольше 12 секунд
    const float max_segment_ms = 12000.0f;

    // Промпт Whisper (глобальный, может быть установлен из run())
    std::string prompt_whisper = g_whisper_prompt;

    // Переменные для отображения
    std::string current_display_text;
    bool is_first_segment = true;

    if (g_verbose_mode.load()) {
        fprintf(stderr, "[AudioInput] Поток запущен (чат-режим): silence=%.0f мс, "
            "max_segment=%.0f мс, soft_limit=%d токенов\n",
            silence_timeout_ms, max_segment_ms, g_soft_limit_tokens.load());
    }

    while (!g_shutting_down.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (g_shutting_down.load()) break;

        // [ПАТЧ] Проверка сброса контекста (reset)
        if (g_reset_in_progress.load()) {
            std::unique_lock<std::mutex> lock(g_reset_mutex);
            g_reset_cv.wait(lock, []{ return !g_reset_in_progress.load(); });
            continue;
        }

        float current_time = get_current_time_ms();

        // Забираем 2 секунды аудио для VAD-анализа
        std::vector<float> pcmf32_cur;
        audio_ref.get(2000, pcmf32_cur);
        if (pcmf32_cur.empty()) continue;

        BotState state = g_bot_state.load();

        // === РЕЖИМ GENERATING: только barge-in ===
        if (state == BotState::GENERATING) {
            int vad_result = vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE,
                static_cast<int>(params.vad_last_ms), params.vad_thold,
                params.freq_thold, params.print_energy, params.vad_start_thold);

            if (vad_result == 1) {
                consecutive_barge_in++;
                if (consecutive_barge_in >= 3) {
                    // Блокируем TTS через файл-семафор
                    std::string dummy;
                    allow_xtts_file(dummy, 0);

                    // [ПАТЧ] МЯГКОЕ ПРЕРЫВАНИЕ: НЕ отменяем текущие TTS-запросы
                    // g_cancel_tts_requests.store(true);

                    g_interrupt_reason.store(InterruptReason::VAD_SPEECH);
                    g_interrupt_processed.store(true);
                    g_bot_state.store(BotState::INTERRUPTED);

                    audio_ref.clear();
                    consecutive_barge_in = 0;

                    {
                        std::lock_guard<std::mutex> lock(g_console_mutex);
                        printf("\n\033[90m[Прервано]\033[0m\n");
                        fflush(stdout);
                    }
                    if (params.verbose) {
                        printf("\n[AudioInput] Barge-in: речь обнаружена!\n");
                    }
                }
            }
            else {
                consecutive_barge_in = 0;
            }
            continue;
        }

        // Пропускаем, если бот не в IDLE
        if (state != BotState::IDLE) continue;

        int vad_result = vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE,
            static_cast<int>(params.vad_last_ms), params.vad_thold,
            params.freq_thold, params.print_energy, params.vad_start_thold);

        // --- НАЧАЛО РЕЧИ ---
        if (vad_result == 1 && !speech_active) {
            speech_active = true;
            speech_start_ms = current_time;
            is_first_segment = true;
            current_display_text.clear();
            g_message_fixed.store(false);
            last_partial_time = current_time;  // Сброс таймера partial

            g_ui_state.store(UiState::UI_ACCUMULATING);
            idle_shown = false;

            // Показываем [SPEECH] сразу
            {
                std::lock_guard<std::mutex> lock(g_console_mutex);
                printf("\n\033[32m%s \xe2\x86\x92\033[0m \033[33m[SPEECH]\033[0m\033[K",
                    person_name.c_str());
                fflush(stdout);
            }

            // Warmup: прогрев GPU (результат игнорируем)
            float prob_warmup = 0.0f;
            int64_t t_warmup = 0;
            transcribe(ctx_wsp, params, pcmf32_cur, prompt_whisper,
                prob_warmup, t_warmup);

            // Блокируем TTS (пользователь говорит — бот должен молчать)
            std::string dummy;
            allow_xtts_file(dummy, 0);

            if (g_verbose_mode.load()) {
                fprintf(stderr, "[AudioInput] Начало речи (%.3f)\n", current_time);
            }
        }

        // --- РЕЧЬ ПРОДОЛЖАЕТСЯ ---
        if (speech_active && (vad_result == 0 || vad_result == 1)) {
            float speech_duration = current_time - speech_start_ms;

            // Принудительная разбивка длинного сегмента
            if (speech_duration > max_segment_ms) {
                std::vector<float> pcmf32_seg;
                audio_ref.get(params.voice_ms, pcmf32_seg);
                if (!pcmf32_seg.empty()) {
                    float prob = 0.0f;
                    int64_t t_ms = 0;
                    std::string seg_text = transcribe(ctx_wsp, params, pcmf32_seg,
                        prompt_whisper, prob, t_ms);
                    trim(seg_text);

                    // [ПАТЧ] ДЕДУПЛИКАЦИЯ: проверяем, не дублируется ли текст
                    if (!seg_text.empty() && !is_hallucination(seg_text) && prob > 0.3f) {
                        bool is_duplicate = false;
                        {
                            std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                            if (!g_accumulated_text.empty() && !seg_text.empty()) {
                                std::string last_part = g_accumulated_text;
                                if (last_part.length() > 50) {
                                    last_part = last_part.substr(last_part.length() - 50);
                                }
                                if (seg_text.length() >= last_part.length() &&
                                    seg_text.substr(0, last_part.length()) == last_part) {
                                    is_duplicate = true;
                                }
                            }
                        }
                        if (!is_duplicate) {
                            std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                            if (!g_accumulated_text.empty()) g_accumulated_text += " ";
                            g_accumulated_text += seg_text;
                        }
                    }
                }
                speech_start_ms = current_time;
                if (g_verbose_mode.load()) {
                    fprintf(stderr, "[AudioInput] Принудительная разбивка (%.0f мс)\n",
                        speech_duration);
                }
            }

            // [ПАТЧ] PARTIAL DISPLAY: показываем промежуточный текст каждые 3 секунды
            if (partial_enabled && (current_time - last_partial_time) > partial_interval_ms) {
                last_partial_time = current_time;

                // Забираем последние 5 секунд аудио для частичной транскрипции
                std::vector<float> pcmf32_partial;
                audio_ref.get(5000, pcmf32_partial);
                if (!pcmf32_partial.empty()) {
                    float prob_partial = 0.0f;
                    int64_t t_partial = 0;
                    std::string partial_text = transcribe(ctx_wsp, params, pcmf32_partial,
                                                          prompt_whisper, prob_partial, t_partial);
                    trim(partial_text);

                    // Собираем полный накопленный текст + частичный
                    std::string full_display;
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        full_display = g_accumulated_text;
                    }
                    if (!full_display.empty() && !partial_text.empty()) {
                        full_display += " ";
                    }
                    full_display += partial_text;

                    // [ПАТЧ] ПОЛНЫЙ ТЕКСТ — БЕЗ ОБРЕЗКИ (clip_for_console не используется)
                    char status_buf[64];
                    snprintf(status_buf, sizeof(status_buf), "[SPEECH %.1fs]",
                        current_time - speech_start_ms);
                    redraw_user_message(person_name, full_display, status_buf);

                    // Сохраняем в глобальный display для других частей кода
                    {
                        std::lock_guard<std::mutex> lock(g_display_mutex);
                        g_display_text = full_display;
                    }
                }
            }
        }

        // --- КОНЕЦ РЕЧИ ---
        if (vad_result == 2 && speech_active) {
            speech_active = false;
            float speech_len = current_time - speech_start_ms;
            last_speech_end_ms = current_time;

            // Разрешаем TTS (пользователь замолчал)
            std::string dummy;
            allow_xtts_file(dummy, 1);

            if (speech_len > 0.5f) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                // Забираем весь сегмент из кольцевого буфера
                std::vector<float> pcmf32_full;
                audio_ref.get(params.voice_ms, pcmf32_full);

                // Паддинг 300 мс тишины в конец сегмента
                const int padding = (WHISPER_SAMPLE_RATE * 300) / 1000;
                pcmf32_full.insert(pcmf32_full.end(), padding, 0.0f);

                if (!pcmf32_full.empty()) {
                    float prob = 0.0f;
                    int64_t t_ms = 0;

                    // ОДИН вызов Whisper, no_context=true
                    std::string text = transcribe(ctx_wsp, params, pcmf32_full,
                        prompt_whisper, prob, t_ms);
                    trim(text);

                    // [ПАТЧ] ДЕДУПЛИКАЦИЯ + фильтр галлюцинаций
                    if (!text.empty() && !is_hallucination(text) && prob > 0.3f) {
                        bool is_duplicate = false;
                        {
                            std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                            if (!g_accumulated_text.empty() && !text.empty()) {
                                std::string last_part = g_accumulated_text;
                                if (last_part.length() > 50) {
                                    last_part = last_part.substr(last_part.length() - 50);
                                }
                                if (text.length() >= last_part.length() &&
                                    text.substr(0, last_part.length()) == last_part) {
                                    is_duplicate = true;
                                }
                            }
                        }

                        if (!is_duplicate) {
                            // Добавляем в аккумулятор
                            {
                                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                                if (!g_accumulated_text.empty()) g_accumulated_text += " ";
                                g_accumulated_text += text;
                            }

                            // Обновляем отображение
                            if (is_first_segment) {
                                current_display_text = text;
                                is_first_segment = false;
                            } else {
                                current_display_text += " " + text;
                            }

                            {
                                std::lock_guard<std::mutex> lock(g_display_mutex);
                                g_display_text = current_display_text;
                            }

                            // Подсчёт токенов через llama_tokenize
                            std::string acc;
                            int tok_count = 0;
                            {
                                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                                acc = g_accumulated_text;
                            }
                            std::vector<llama_token> toks =
                                llama_tokenize(ctx_llama, acc, false);
                            tok_count = static_cast<int>(toks.size());
                            int soft = g_soft_limit_tokens.load();

                            char status_buf[256];
                            snprintf(status_buf, sizeof(status_buf),
                                "[%d/%d tok] \033[36m[SILENCE]\033[0m",
                                tok_count, soft);

                            // [ПАТЧ] ПОЛНЫЙ ТЕКСТ БЕЗ ОБРЕЗКИ
                            redraw_user_message(person_name, current_display_text, status_buf);

                            if (g_verbose_mode.load()) {
                                fprintf(stderr, "[AudioInput] Сегмент: '%s' "
                                    "(%.1f сек, prob=%.3f, %d ток)\n",
                                    text.c_str(), speech_len, prob, tok_count);
                            }

                            g_message_fixed.store(true);
                        }
                        else if (g_verbose_mode.load()) {
                            fprintf(stderr, "[AudioInput] Дубль: '%s'\n", text.c_str());
                        }
                    }
                    else if (g_verbose_mode.load() && !text.empty()) {
                        fprintf(stderr, "[AudioInput] Отфильтровано: '%s' (prob=%.3f)\n",
                            text.c_str(), prob);
                    }
                }
            }
            speech_start_ms = 0;
        }

        // --- ТАЙМАУТ ТИШИНЫ: отправка аккумулятора в LLaMA ---
        if (!speech_active) {
            float silence_ms = (current_time - last_speech_end_ms) * 1000.0f;

            if (silence_ms > silence_timeout_ms) {
                std::string acc;
                {
                    std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                    acc = g_accumulated_text;
                }

                if (!acc.empty() && !g_pending_llm_request.load()) {
                    // Передаём текст в основной цикл
                    {
                        std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
                        g_pending_llm_text = acc;
                    }
                    g_pending_llm_request.store(true);
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        g_accumulated_text.clear();
                    }

                    g_ui_state.store(UiState::UI_HANDOFF);
                    idle_shown = false;

                    // [ПАТЧ] Убираем лишние сообщения — показываем статус в строке
                    {
                        std::lock_guard<std::mutex> lock(g_console_mutex);
                        printf("\r\033[K\033[90m[LLM]\033[0m");
                        fflush(stdout);
                    }

                    last_speech_end_ms = current_time;
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Таймаут тишины: отправка %zu символов\n",
                            acc.size());
                    }

                    // Очищаем display для следующего цикла
                    {
                        std::lock_guard<std::mutex> lock(g_display_mutex);
                        g_display_text.clear();
                    }
                    g_message_fixed.store(false);
                    current_display_text.clear();
                    is_first_segment = true;
                }
            }

            // --- ЛИМИТ ТОКЕНОВ: принудительная отправка ---
            {
                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                if (!g_accumulated_text.empty() && !g_pending_llm_request.load()) {
                    std::vector<llama_token> toks =
                        llama_tokenize(ctx_llama, g_accumulated_text, false);

                    if (static_cast<int>(toks.size()) >= g_soft_limit_tokens.load()) {
                        std::string acc = g_accumulated_text;
                        g_accumulated_text.clear();

                        {
                            std::lock_guard<std::mutex> plock(g_pending_llm_mutex);
                            g_pending_llm_text = acc;
                        }
                        g_pending_llm_request.store(true);

                        g_ui_state.store(UiState::UI_HANDOFF);
                        idle_shown = false;

                        // [ПАТЧ] Показываем лимит токенов как статус
                        {
                            std::lock_guard<std::mutex> clock(g_console_mutex);
                            printf("\r\033[K\033[90m[LLM %d ток]\033[0m",
                                static_cast<int>(toks.size()));
                            fflush(stdout);
                        }

                        last_speech_end_ms = current_time;
                        if (g_verbose_mode.load()) {
                            fprintf(stderr, "[AudioInput] Лимит токенов (%d): отправка\n",
                                static_cast<int>(toks.size()));
                        }

                        {
                            std::lock_guard<std::mutex> lock(g_display_mutex);
                            g_display_text.clear();
                        }
                        g_message_fixed.store(false);
                        current_display_text.clear();
                        is_first_segment = true;
                    }
                }
            }
        }

        // --- ПОКАЗ [IDLE] ---
        // [ПАТЧ] Строгое правило: [IDLE] только когда UI_IDLE, нет речи, аккумулятор пуст.
        if (!speech_active && g_ui_state.load() == UiState::UI_IDLE && g_message_fixed.load()) {
            std::string acc;
            {
                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                acc = g_accumulated_text;
            }
            if (acc.empty() && current_display_text.empty()) {
                if (!idle_shown) {
                    std::lock_guard<std::mutex> lock(g_console_mutex);
                    printf("\r\033[32m%s \xe2\x86\x92\033[0m "
                        "\033[90m[IDLE]\033[0m\033[K",
                        person_name.c_str());
                    fflush(stdout);
                    idle_shown = true;
                }
            }
            else {
                idle_shown = false;
            }
        }
        else {
            idle_shown = false;
        }
    }

    g_audio_thread_running.store(false);
    if (g_verbose_mode.load()) {
        fprintf(stderr, "[AudioInput] Поток остановлен\n");
    }
}

// ========================================================================
// 21. ОСНОВНАЯ ФУНКЦИЯ run()
// ========================================================================

// [21.1] Сердце ассистента: инициализация, запуск потоков, основной цикл
int run(int argc, char** argv) {
    whisper_params params;
    std::vector<std::thread> threads;
    int reply_part = 0;
    bool last_output_has_username = false;
    bool last_output_has_EOT = true;
    int input_tokens_count = 0;
    float llama_time_input = 0.0f;
    float llama_time_output = 0.0f;
    llama_sampler* smpl = nullptr;
    llama_sampler* smpl_high_temp = nullptr;

    // [21.2] Парсинг аргументов
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }
    if (params.language != "auto" &&
        whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n",
            params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // Начальное разрешение TTS
    allow_xtts_file(params.xtts_control_path, 1);

    // [21.3] Инициализация Whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    struct whisper_context* ctx_wsp =
        whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
    if (!ctx_wsp) {
        fprintf(stderr, "No whisper.cpp model specified.  "
            "Please provide using -mw <modelfile>\n");
        return 1;
    }

    // [21.4] Инициализация LLaMA
    llama_backend_init();
    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    }
    else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }
    lmparams.main_gpu = params.main_gpu;
    if (params.split_mode == "layer")
        lmparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
    else
        lmparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    lmparams.tensor_split = params.tensor_split.empty()
        ? nullptr : params.tensor_split.data();
    struct llama_model* model_llama =
        llama_model_load_from_file(params.model_llama.c_str(), lmparams);
    if (!model_llama) {
        fprintf(stderr, "No llama.cpp model specified.  "
            "Please provide using -ml <modelfile>\n");
        return 1;
    }
    params.tensor_split.clear();
    const llama_vocab* vocab_llama = llama_model_get_vocab(model_llama);
    bool add_bos_token = llama_vocab_get_add_bos(vocab_llama);
    const int n_keep = params.n_keep + (add_bos_token ? 1 : 0);

    llama_context_params lcparams = llama_context_default_params();
    lcparams.n_ctx = params.ctx_size;
    if (params.verbose) {
        fprintf(stdout, "n_ctx %d ", lcparams.n_ctx);
    }
    lcparams.n_threads = params.n_threads;
    lcparams.flash_attn_type = params.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_AUTO
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context* ctx_llama =
        llama_init_from_model(model_llama, lcparams);
    if (!ctx_llama) {
        fprintf(stderr, "error: failed to initialize llama context\n");
        return 1;
    }

    // [21.5] Специальные токены для остановки генерации
    llama_token special_token_ids[64] = { 0 };
    int special_token_count = 0;
    if (!params.instruct_preset_data["bot_message_suffix"].empty()) {
        std::vector<llama_token> tokens = ::llama_tokenize(
            ctx_llama, params.instruct_preset_data["bot_message_suffix"], false);
        if (!tokens.empty() && special_token_count < 64) {
            special_token_ids[special_token_count++] = tokens[0];
        }
    }
    if (!params.instruct_preset_data["stop_sequence"].empty()) {
        std::vector<llama_token> tokens = ::llama_tokenize(
            ctx_llama, params.instruct_preset_data["stop_sequence"], false);
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
    const char* special_patterns[] = {
        "<|eot_id|>", "<|im_end|>", "</s>", "<end_of_turn|>",
        "<|endoftext|>", "<|im_start|>", "<|end|>", "<|eo|>",
        "<|start_header_id|>", "<|end_header_id|>",
    };
    for (const char* pattern : special_patterns) {
        std::vector<llama_token> tokens =
            ::llama_tokenize(ctx_llama, pattern, false);
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
    if (!params.stop_words.empty()) {
        std::vector<llama_token> tokens =
            ::llama_tokenize(ctx_llama, params.stop_words, false);
        for (llama_token tok : tokens) {
            if (special_token_count >= 64) break;
            bool already = false;
            for (int j = 0; j < special_token_count; j++) {
                if (special_token_ids[j] == tok) { already = true; break; }
            }
            if (!already) {
                special_token_ids[special_token_count++] = tok;
            }
        }
    }
    if (params.debug && special_token_count > 0) {
        printf("[DEBUG] Special token IDs to filter: ");
        for (int i = 0; i < special_token_count; i++) {
            printf("%d ", special_token_ids[i]);
        }
        printf("\n");
    }

    // [21.6] Информация об обработке
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual,  "
                    "ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s,  "
            "timestamps = %d ...\n",
            __func__, params.n_threads, params.language.c_str(),
            params.translate ? "translate" : "transcribe",
            params.no_timestamps ? 0 : 1);
        fprintf(stderr, "\n");
    }

    // [21.7] Инициализация аудиобуфера
    audio_async audio(15 * 1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: Ошибка инициализации аудиоустройства (ID: %d)\n",
            __func__, params.capture_id);
        fprintf(stderr, "Проверьте доступные аудиоустройства и правильность  "
            "ID захвата\n");
        return 1;
    }
    audio.resume();
    bool is_running = true;
    bool force_speak = false;

    // [21.8] Разделители: двоеточие для модели, стрелка для UI
    const std::string chat_symb = ":";
    const std::string chat_display = " \xe2\x86\x92 ";
    std::vector<float> pcmf32_cur;

    // [21.9] Промпт Whisper (контекстная подсказка)
    if (params.language == "ru") {
        g_whisper_prompt = "Распознавай только речь " + params.person + ". Игнорируй фоновый шум.";
    } else {
        g_whisper_prompt = "Recognize only " + params.person + "'s speech. Ignore background noise.";
    }

    // [21.10] Определение пола пользователя
    bool user_is_female = false;
    {
        std::string user_lower = params.person;
        std::transform(user_lower.begin(), user_lower.end(),
            user_lower.begin(), ::tolower);
        if (!user_lower.empty()) {
            char last_char = user_lower.back();
            if (last_char == 'а' || last_char == 'я' || last_char == 'ь') {
                user_is_female = true;
            }
            else if (user_lower.length() >= 2 &&
                user_lower.substr(user_lower.length() - 2) == "ия") {
                user_is_female = true;
            }
            if (user_is_female) {
                static const std::unordered_set<std::string> male_exceptions = {
                    "никита", "илья", "фома", "лука", "кузьма", "добрыня",
                    "митя", "ваня", "петя", "вася", "лёша", "леша",
                    "саша", "женя", "дима", "коля", "толя",
                    "гриша", "миша", "паша", "стёпа", "степа",
                    "тоша", "гоша", "костя", "владя", "даня",
                    "сеня", "тёма", "тема",
                    "василий", "николай", "алексей", "дмитрий", "сергей",
                    "андрей", "матвей", "арсений", "евгений", "валерий",
                    "юрий", "игорь", "борис", "денис", "роман", "виктор"
                };
                if (male_exceptions.find(user_lower) != male_exceptions.end()) {
                    user_is_female = false;
                }
            }
        }
    }

    // [21.11] Промпт LLaMA
    std::string prompt_llama = params.prompt.empty()
        ? k_prompt_llama : params.prompt;
    if (!params.instruct_preset.empty()) {
        try {
            std::string filename = "instruct_presets/" +
                params.instruct_preset + ".json";
            nlohmann::json jsonData;
            std::ifstream jsonFile(filename);
            if (jsonFile.is_open()) {
                jsonFile >> jsonData;
                jsonFile.close();
                params.instruct_preset_data = jsonData;
            }
            else {
                std::cout << "Warning: preset file '" << filename
                    << "' does not exist. Turning off instruct mode "
                    << std::endl;
                params.instruct_preset = "";
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            return 1;
        }
    }
    else {
        params.instruct_preset = "";
    }
    prompt_llama.insert(0, 1, ' ');

    // [21.12] Подстановка времени и даты в промпт
    std::string time_str, year_str, ymd;
    {
        time_t t = time(0);
        struct tm* now = localtime(&t);
        char buf[128];
        strftime(buf, sizeof(buf), "%H:%M", now);
        time_str = buf;
        strftime(buf, sizeof(buf), "%Y", now);
        year_str = buf;
        strftime(buf, sizeof(buf), "%d %B %Y года", now);
        std::string ymd_str = buf;
        ymd_str = string_replace_all(ymd_str, "January", "января");
        ymd_str = string_replace_all(ymd_str, "February", "февраля");
        ymd_str = string_replace_all(ymd_str, "March", "марта");
        ymd_str = string_replace_all(ymd_str, "April", "апреля");
        ymd_str = string_replace_all(ymd_str, "May", "мая");
        ymd_str = string_replace_all(ymd_str, "June", "июня");
        ymd_str = string_replace_all(ymd_str, "July", "июля");
        ymd_str = string_replace_all(ymd_str, "August", "августа");
        ymd_str = string_replace_all(ymd_str, "September", "сентября");
        ymd_str = string_replace_all(ymd_str, "October", "октября");
        ymd_str = string_replace_all(ymd_str, "November", "ноября");
        ymd_str = string_replace_all(ymd_str, "December", "декабря");
        ymd = ymd_str;
    }
    prompt_llama = string_replace_all(prompt_llama, "{0}", params.person);
    prompt_llama = string_replace_all(prompt_llama, "{1}", params.bot_name);
    prompt_llama = string_replace_all(prompt_llama, "{2}", time_str);
    prompt_llama = string_replace_all(prompt_llama, "{3}", year_str);
    prompt_llama = string_replace_all(prompt_llama, "{4}", chat_symb);
    prompt_llama = string_replace_all(prompt_llama, "{5}", ymd);
    if (params.language == "ru") {
        if (user_is_female) {
            prompt_llama += "\n[" + params.person + " — женщина.]\n";
        }
        else {
            prompt_llama += "\n[" + params.person + " — мужчина.]\n";
        }
    }
    else {
        if (user_is_female) {
            prompt_llama += "\n[" + params.person + " is female.]\n";
        }
        else {
            prompt_llama += "\n[" + params.person + " is male.]\n";
        }
    }

    // [21.13] Подготовка батча и сэмплера
    llama_batch batch = llama_batch_init(params.ctx_size, 0, 1);
    if (params.verbose) {
        fprintf(stdout, "llama_n_ctx %d ", llama_n_ctx(ctx_llama));
    }
    const float top_k = static_cast<float>(params.top_k);
    const float top_p = params.top_p;
    const float min_p = params.min_p;
    float temp = params.temp;
    const float repeat_penalty = params.repeat_penalty;
    const int seed = 0;
    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);
    smpl_high_temp = llama_sampler_chain_init(sparams);
    if (temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        llama_sampler_chain_add(smpl_high_temp,
            llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl_high_temp,
            llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl_high_temp,
            llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl_high_temp,
            llama_sampler_init_temp(2.00f));
        llama_sampler_chain_add(smpl_high_temp,
            llama_sampler_init_dist(seed));
    }
    else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_greedy());
    }

    // [21.14] Поддержка instruct-пресета
    if (!params.instruct_preset.empty()) {
        std::string wrapped_prompt;
        if (!prompt_llama.empty() &&
            !params.instruct_preset_data["system_prompt_prefix"].empty()) {
            wrapped_prompt +=
                params.instruct_preset_data["system_prompt_prefix"];
            wrapped_prompt += prompt_llama;
            wrapped_prompt +=
                params.instruct_preset_data["system_prompt_suffix"];
        }
        if (params.prompt.empty()) {
            if (!params.instruct_preset_data["user_message_prefix"].empty()) {
                wrapped_prompt +=
                    params.instruct_preset_data["user_message_prefix"];
            }
            if (prompt_llama.empty() ||
                wrapped_prompt.find(prompt_llama) == std::string::npos) {
                wrapped_prompt += prompt_llama;
            }
            if (!params.instruct_preset_data["user_message_suffix"].empty()) {
                wrapped_prompt +=
                    params.instruct_preset_data["user_message_suffix"];
            }
        }
        if (!params.instruct_preset_data["bot_message_prefix"].empty()) {
            wrapped_prompt +=
                params.instruct_preset_data["bot_message_prefix"];
        }
        prompt_llama = wrapped_prompt;
    }

    // [21.15] Токенизация промпта и загрузка сессии
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);
    if (static_cast<int>(embd_inp.size()) > params.ctx_size - 512) {
        int keep = std::min(params.n_keep,
            static_cast<int>(embd_inp.size()));
        if (static_cast<int>(embd_inp.size()) > keep + 256) {
            embd_inp.erase(embd_inp.begin() + keep,
                embd_inp.end() - 256);
        }
        std::cerr << "[warn] Context trimmed: " << embd_inp.size()
            << " tokens (ctx limit " << params.ctx_size << ")\n";
    }
    static std::vector<llama_token> recent_tokens;
    if (params.repeat_last_n > 0) {
        if (static_cast<int>(embd_inp.size()) > params.repeat_last_n) {
            recent_tokens.assign(embd_inp.end() - params.repeat_last_n,
                embd_inp.end());
        }
        else {
            recent_tokens = embd_inp;
        }
    }
    std::string path_session = params.path_session;
    std::vector<llama_token> session_tokens;
    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from %s\n",
            __func__, path_session.c_str());
        FILE* fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);
            session_tokens.resize(llama_n_ctx(ctx_llama));
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx_llama, path_session.c_str(),
                session_tokens.data(), session_tokens.size(),
                &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file  "
                    "'%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            embd_inp.assign(session_tokens.begin(), session_tokens.end());
            fprintf(stderr, "%s: loaded a session with prompt size of  "
                "%zu tokens\n", __func__, session_tokens.size());
        }
        else {
            fprintf(stderr, "%s: session file does not exist, will create\n",
                __func__);
        }
    }

    // [21.16] Оценка начального промпта
    if (params.verbose) {
        printf("\n");
    }
    printf("%s : initializing - please wait ...\n", __func__);
    float llama_start_time = get_current_time_ms();
    int n_past = 0;
    batch = llama_batch_init(2048, 0, 1);
    {
        if (embd_inp.size() > 2048) {
            fprintf(stderr, "FATAL: Initial prompt size (%zu tokens)  "
                "exceeds batch limit (2048)\n", embd_inp.size());
            return 1;
        }
        batch.n_tokens = static_cast<int>(embd_inp.size());
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.token[i] = embd_inp[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
        }
    }
    if (llama_decode(ctx_llama, batch)) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        return 1;
    }
    float llama_end_time = get_current_time_ms();
    float llama_time_total = llama_end_time - llama_start_time;
    printf("\nLlama start prompt: %zu/%d tokens in %.3f s at %.0f t/s\n",
        embd_inp.size(), params.ctx_size,
        static_cast<double>(llama_time_total),
        static_cast<double>(embd_inp.size() / llama_time_total));
    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s ", prompt_llama.c_str());
        fflush(stdout);
    }
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() ||
                id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n",
                __func__);
        }
        else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity  "
                "to prompt (%zu / %zu tokens)\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
        else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens  "
                "of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }
    bool need_to_save_session = !path_session.empty() &&
        n_matching_session_tokens < (embd_inp.size() * 3 / 4);
    printf("%s : done! start speaking in the microphone\n", __func__);
    const std::string wake_cmd = params.wake_cmd;
    if (!wake_cmd.empty()) {
        printf("%s : the wake-up command is: '%s'\n",
            __func__, wake_cmd.c_str());
    }

    // [21.17] Показываем горячие клавиши серым цветом
    printf("\n\033[90mVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right),  "
           "Delete(Ctrl+Delete), Reset(Ctrl+R)\033[0m\n");
    if (params.push_to_talk)
        printf("\033[90mType anything or hold 'Alt' to speak:\033[0m\n");
    else
        printf("\033[90mStart speaking or typing:\033[0m\n\n");

    fflush(stdout);

    // [21.18] Инициализация состояния бота
    g_bot_state.store(BotState::IDLE);
    g_interrupt_reason.store(InterruptReason::NONE);
    g_interrupt_processed.store(false);
    g_shutting_down.store(false);
    g_reset_in_progress.store(false);
    audio.clear();
    const int n_ctx = llama_n_ctx(ctx_llama);
    n_past = static_cast<int>(embd_inp.size());
    std::vector<int> past_prev_arr{};
    int n_past_prev = 0;
    const size_t MAX_PAST_PREV_SIZE = 100;
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0
        ? static_cast<int>(session_tokens.size()) : 0;
    std::vector<llama_token> embd;
    std::string text_heard_prev;
    std::string text_heard_trimmed;
    int new_command_allowed = 1;
    std::vector<std::string> tts_intros;
    std::string rand_intro_text = "";
    std::string last_output_buffer = "";
    std::string last_output_needle = "";
    if (params.language == "ru") {
        tts_intros = { "Хм", "Ну", "Нуу", "О", "А", "А?", "Угу", "Ох",
            "Ха", "Ах", "Блин", "Короче", "В общем", "Ой",
            "Слышь", "Ну вообще-то", "Ну а вообще", "Кароче",
            "Вот", "Знаешь", "Как бы", "Прикинь", "Послушай",
            "Типа", "Это", "Так вот", "Погоди", params.person };
    }
    else {
        tts_intros = { "Hm", "Hmm", "Well", "Well well", "Huh", "Ugh",
            "Uh", "Um", "Mmm", "Oh", "Ooh", "Haha", "Ha ha",
            "Ahh", "Whoa", "Really", "I mean", "By the way",
            "Anyway", "So", "Actually", "Uh-huh", "Seriously",
            "Whatever", "Ahh", "Like", "But", "You know",
            "Wait", "Ahem", "Damn", params.person };
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist;
    std::chrono::steady_clock::time_point last_command_time =
        std::chrono::steady_clock::now();
    std::string current_voice = params.xtts_voice.empty()
        ? params.bot_name : params.xtts_voice;

    // [21.19] Антипромпты
    std::vector<std::string> antiprompts;
    if (!params.allow_newline) {
        antiprompts.push_back("\n");
    }
    antiprompts.push_back("\n" + params.person + chat_symb);
    antiprompts.push_back("\n" + params.person + " " + chat_symb);
    // [ПАТЧ] Защита от самодиалога — антипромпты на имя бота
    antiprompts.push_back("\n" + params.bot_name + chat_symb);
    antiprompts.push_back("\n" + params.bot_name + " " + chat_symb);
    // [ПАТЧ] Дополнительные антипромпты для защиты от самодиалога
    antiprompts.push_back("\n" + params.person + ":" + params.person);
    antiprompts.push_back("\n" + params.bot_name + ":" + params.bot_name);
    antiprompts.push_back("\n" + params.person + " сказал" + chat_symb);
    antiprompts.push_back("\n" + params.bot_name + " сказал" + chat_symb);

    if (!params.stop_words.empty()) {
        size_t start = 0, end = params.stop_words.find(';');
        auto add_word = [&](std::string w) {
            if (w.length() >= 2) {
                w = string_replace_all(w, "\r", "\r");
                w = string_replace_all(w, "\n", "\n");
                if (std::find(antiprompts.begin(), antiprompts.end(), w)
                    == antiprompts.end()) {
                    antiprompts.push_back(w);
                }
            }
            };
        if (end == std::string::npos) {
            add_word(params.stop_words);
        }
        else {
            while (start < params.stop_words.size()) {
                std::string word = params.stop_words.substr(start,
                    end - start);
                add_word(word);
                start = end + 1;
                end = params.stop_words.find(';', start);
                if (end == std::string::npos) end = params.stop_words.size();
            }
        }
    }

    // [21.20] Обновление антипромптов при смене имени бота
    auto update_antiprompts = [&](const std::string& new_person,
        const std::string& new_bot_name) {
        size_t user_offset = (!antiprompts.empty() &&
            antiprompts[0] == "\n") ? 1 : 0;
        if (antiprompts.size() >= user_offset + 2) {
            antiprompts[user_offset] = "\n" + new_person + chat_symb;
            antiprompts[user_offset + 1] = "\n" + new_person + " " + chat_symb;
        }
        std::string old_bot_pattern1 = "\n" + params.bot_name + chat_symb;
        std::string old_bot_pattern2 = "\n" + params.bot_name + " " + chat_symb;
        bool found1 = false, found2 = false;
        for (const auto& ap : antiprompts) {
            if (ap == old_bot_pattern1) found1 = true;
            if (ap == old_bot_pattern2) found2 = true;
        }
        if (!found1 && !old_bot_pattern1.empty() &&
            old_bot_pattern1 != "\n" + new_person + chat_symb) {
            antiprompts.push_back(old_bot_pattern1);
        }
        if (!found2 && !old_bot_pattern2.empty() &&
            old_bot_pattern2 != "\n" + new_person + " " + chat_symb) {
            antiprompts.push_back(old_bot_pattern2);
        }
        if (!params.instruct_preset_data["stop_sequence"].empty()) {
            std::string stop_seq =
                params.instruct_preset_data["stop_sequence"];
            bool stop_seq_found = false;
            for (const auto& ap : antiprompts) {
                if (ap == stop_seq) { stop_seq_found = true; break; }
            }
            if (!stop_seq_found && !stop_seq.empty()) {
                antiprompts.push_back(stop_seq);
            }
        }
        if (params.verbose) {
            printf("\n[DEBUG] Antiprompts updated. New bot: '%s'.  "
                "Total antiprompts: %zu\n",
                new_bot_name.c_str(), antiprompts.size());
        }
        };

    if (params.verbose) {
        printf("Llama stop words (%zu total): ", antiprompts.size());
        for (size_t i = 0; i < antiprompts.size(); i++) {
            std::string display = antiprompts[i];
            display = string_replace_all(display, "\r", "\r");
            display = string_replace_all(display, "\n", "\n");
            display = string_replace_all(display, "\t", "\t");
            printf("%s'%s' ", i > 0 ? ", " : " ", display.c_str());
        }
        if (!params.stop_words.empty()) {
            printf(" [+ from --stop-words: %s] ", params.stop_words.c_str());
        }
        printf("\n");
    }

    // [21.21] Вычисление лимита токенов
    g_soft_limit_tokens.store(params.ctx_size / 3);
    if (params.verbose) {
        printf("[Limits] soft_limit_tokens=%d (ctx_size=%d / 3)\n",
            g_soft_limit_tokens.load(), params.ctx_size);
    }

    // [21.22] Запуск потоков
    std::thread input_thread(input_thread_func);
    std::thread shortcut_thread([&]() {
        keyboard_shortcut_func();
        });
    std::thread audio_thread([&]() {
        audio_input_thread_func(ctx_wsp, ctx_llama, params, audio,
            params.person);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (params.verbose && g_audio_thread_running.load()) {
        printf("[AudioInput] Thread started successfully\n");
    }

    // [21.23] Инициализация глобального display
    {
        std::lock_guard<std::mutex> lock(g_display_mutex);
        g_display_text.clear();
    }
    g_message_fixed.store(false);

    // [21.24] Основной цикл работы
    float speech_start_sec = 0;
    float speech_end_sec = 0;
    float speech_len = 0;
    float llama_interrupted_time = 0.0;
    llama_start_time = 0.0;
    float llama_start_generation_time = 0.0;
    llama_end_time = 0.0;
    llama_time_total = 0.0;
    std::string user_typed = "";
    bool user_typed_this = false;
    bool first_dialog_pair = true;
    bool warmup_done = false;

    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            printf("\n[Shutdown requested, cleaning up...]\n");
            break;
        }
        g_interrupt_reason.store(InterruptReason::NONE);
        g_interrupt_processed.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int64_t t_ms = 0;
        std::string text_heard = "";
        user_typed_this = false;
        console::set_display(console::reset);

        // Проверка запроса из audio_input_thread
        if (g_pending_llm_request.load()) {
            std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
            if (!g_pending_llm_text.empty()) {
                user_typed = g_pending_llm_text;
                user_typed_this = true;
                g_pending_llm_text.clear();
                g_ui_state.store(UiState::UI_USER_LINE);
                if (params.verbose) {
                    fprintf(stderr, "[AudioInput] Получен текст: "
                        "%zu символов\n", user_typed.size());
                }
            }
            g_pending_llm_request.store(false);
        }

        // Обработка ввода с клавиатуры
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            if (!input_queue.empty()) {
                std::string buffer;
                while (!input_queue.empty()) {
                    buffer += input_queue.front() + " ";
                    input_queue.pop();
                }
                trim(buffer);
                if (!buffer.empty()) {
                    user_typed = buffer;
                    user_typed_this = true;
                    g_pending_llm_request.store(false);
                    {
                        std::lock_guard<std::mutex> pl(g_pending_llm_mutex);
                        g_pending_llm_text.clear();
                    }
                }
            }
        }

        // Обработка горячих клавиш
        std::string hk_copy;
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            hk_copy = g_hotkey_pressed;
            g_hotkey_pressed = "";
        }
        if (!hk_copy.empty()) {
            if (hk_copy == "Ctrl+Space") {
                user_typed = "Stop";
            }
            else if (hk_copy == "Ctrl+Right") {
                user_typed = "Regenerate";
            }
            else if (hk_copy == "Ctrl+Delete") {
                user_typed = "Delete";
            }
            else if (hk_copy == "Ctrl+R") {
                user_typed = "Reset";
            }
            if (hk_copy != "Alt") {
                user_typed_this = true;
            }
        }
        if (user_typed.empty() && !force_speak) {
            continue;
        }

        // === ОБРАБОТКА ПОЛУЧЕННОГО ТЕКСТА ===
        std::string display_text_for_ui = user_typed;
        trim(display_text_for_ui);
        text_heard = user_typed;
        user_typed = "";
        trim(text_heard);
        if (text_heard.empty() && !force_speak) {
            continue;
        }

        // Wake-command
        if (!params.wake_cmd.empty()) {
            if (text_heard.find(params.wake_cmd) != 0) {
                if (params.verbose) {
                    fprintf(stdout, "[wake] ignored: \"%s\"\n",
                        text_heard.c_str());
                }
                continue;
            }
            text_heard = text_heard.substr(params.wake_cmd.length());
            trim(text_heard);
        }

        // Удаление имени пользователя из начала реплики
        if (!text_heard.empty()) {
            std::string heard_lower = text_heard;
            std::string person_lower = params.person;
            std::transform(heard_lower.begin(), heard_lower.end(),
                heard_lower.begin(), ::tolower);
            std::transform(person_lower.begin(), person_lower.end(),
                person_lower.begin(), ::tolower);
            bool removed = false;
            if (heard_lower.find(person_lower + ":") == 0) {
                text_heard = text_heard.substr(person_lower.length() + 1);
                removed = true;
            }
            else if (heard_lower.find(person_lower + " :") == 0) {
                text_heard = text_heard.substr(person_lower.length() + 2);
                removed = true;
            }
            if (removed) {
                trim(text_heard);
            }
        }

        // Heard-ok
        if (!params.heard_ok.empty()) {
            std::string voice_copy = current_voice;
            std::string lang_copy = params.language;
            std::string url_copy = params.xtts_url;
            std::string heard_ok_copy = params.heard_ok;
            safe_thread_emplace(threads,
                [heard_ok_copy, voice_copy, lang_copy, url_copy]() {
                    send_tts_async(heard_ok_copy, voice_copy,
                        lang_copy, url_copy);
                });
        }

        // Очистка текста от мусора
        try {
            std::regex re(R"(\[[^\[\]]*\])");
            text_heard = std::regex_replace(text_heard, re, "");
        }
        catch (const std::regex_error& e) {
            if (params.verbose) {
                fprintf(stderr, "Regex error while removing [brackets]: %s\n",
                    e.what());
            }
        }
        if (params.language == "en" && !user_typed_this) {
            text_heard = std::regex_replace(text_heard,
                std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");
        }
        text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));
        text_heard = std::regex_replace(text_heard,
            std::regex("^\\s+"), "");
        text_heard = std::regex_replace(text_heard,
            std::regex("\\s+$"), "");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, ",");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, ".");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "\xC2\xBB");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "[");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "]");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "\"");
        if (!text_heard.empty() && text_heard[0] == '.')
            text_heard.erase(0, 1);
        if (!text_heard.empty() && text_heard[0] == '[')
            text_heard.erase(0, 1);
        trim(text_heard);

        // Фильтрация галлюцинаций и мусорного текста
        bool is_garbage = false;
        if (text_heard.empty() ||
            text_heard == "!" || text_heard == "." || text_heard == "?" ||
            text_heard == "..." || text_heard == "!!" || text_heard == "??") {
            is_garbage = true;
        }
        if (!is_garbage && is_hallucination(text_heard)) {
            is_garbage = true;
        }
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
            text_heard.find("ПЕСНЯ") != std::string::npos ||
            text_heard.find("музыка") != std::string::npos ||
            text_heard.find("тишина") != std::string::npos
            )) {
            is_garbage = true;
        }
        if (!is_garbage && !text_heard.empty()) {
            std::string lower = text_heard;
            std::transform(lower.begin(), lower.end(),
                lower.begin(), ::tolower);
            size_t len = lower.length();
            if (len > 20) {
                size_t half = len / 2;
                if (half >= 10 &&
                    lower.substr(0, half) == lower.substr(half, half)) {
                    is_garbage = true;
                }
            }
        }
        if (!is_garbage && (
            text_heard == "Sil" || text_heard == "Bye" ||
            text_heard == "Okay." || text_heard == "Thanks." ||
            text_heard == "Bye."
            )) {
            is_garbage = true;
        }
        if (!is_garbage && text_heard == params.bot_name) {
            is_garbage = true;
        }
        if (is_garbage) {
            text_heard = "";
            if (params.verbose) {
                fprintf(stdout, "\n[Фильтр: удалён мусор]\n");
            }
        }
        text_heard = std::regex_replace(text_heard,
            std::regex("\\s+$"), "");
        text_heard_trimmed = text_heard;
        trim(text_heard_trimmed);
        if (!text_heard_trimmed.empty()) {
            if (text_heard_trimmed[0] == '.')
                text_heard_trimmed.erase(0, 1);
            if (!text_heard_trimmed.empty() && text_heard_trimmed[0] == '!')
                text_heard_trimmed.erase(0, 1);
        }
        if (!text_heard_trimmed.empty()) {
            size_t last_pos = text_heard_trimmed.length() - 1;
            if (text_heard_trimmed[last_pos] == '.' ||
                text_heard_trimmed[last_pos] == '!') {
                text_heard_trimmed.erase(last_pos, 1);
            }
        }
        trim(text_heard);
        if (!text_heard.empty()) {
            if (text_heard[0] == '.' || text_heard[0] == '!') {
                text_heard.erase(0, 1);
                trim(text_heard);
            }
            if (!text_heard.empty()) {
                size_t last_pos = text_heard.length() - 1;
                if (text_heard[last_pos] == '.' ||
                    text_heard[last_pos] == '!') {
                    text_heard.erase(last_pos, 1);
                    trim(text_heard);
                }
            }
        }
        text_heard_trimmed = LowerCase(text_heard);
        trim(text_heard_trimmed);
        fflush(stdout);
        std::string user_command;

        // Вводное предложение XTTS
        if (params.xtts_intro && !text_heard_trimmed.empty()) {
            dist = std::uniform_int_distribution<size_t>(
                0, tts_intros.size() - 1);
            rand_intro_text = tts_intros[dist(gen)];
            if (!rand_intro_text.empty()) {
                for (auto it = threads.begin(); it != threads.end(); ) {
                    if (it->joinable()) {
                        it->detach();
                        it = threads.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                std::string voice_copy = current_voice;
                std::string lang_copy = params.language;
                std::string url_copy = params.xtts_url;
                safe_thread_emplace(threads,
                    [rand_intro_text, voice_copy, lang_copy, url_copy]() {
                        send_tts_async(rand_intro_text, voice_copy,
                            lang_copy, url_copy);
                    });
            }
        }

        // === ОПРЕДЕЛЕНИЕ КОМАНД ===
        // [ПАТЧ] ИСПРАВЛЕН РЕГИСТР — все команды в нижнем регистре
        if (text_heard_trimmed.find("regenerate") != std::string::npos ||
            text_heard_trimmed.find("переделай") != std::string::npos ||
            text_heard_trimmed.find("переделаем") != std::string::npos ||
            text_heard_trimmed.find("егенерируй") != std::string::npos ||
            text_heard_trimmed.find("егенерировать") != std::string::npos) {
            user_command = "regenerate";
        }
        else if (text_heard_trimmed.find("google") != std::string::npos ||
            text_heard_trimmed.find("погугли") != std::string::npos ||
            text_heard_trimmed.find("по гугл") != std::string::npos) {
            user_command = "google";
        }
        else if (text_heard_trimmed.find("reset") != std::string::npos ||
            text_heard_trimmed.find("сброс") != std::string::npos ||
            text_heard_trimmed.find("сбросить") != std::string::npos ||
            text_heard_trimmed.find("удали все") != std::string::npos ||
            text_heard_trimmed.find("удалить все") != std::string::npos) {
            user_command = "reset";
        }
        else if (text_heard_trimmed.find("delete") != std::string::npos ||
            text_heard_trimmed.find("please do it") != std::string::npos ||
            text_heard_trimmed.find("удалить сообщение") != std::string::npos ||
            text_heard_trimmed.find("удали сообщение") != std::string::npos ||
            text_heard_trimmed.find("удали два сообщения") != std::string::npos ||
            text_heard_trimmed.find("удали три сообщения") != std::string::npos) {
            user_command = "delete";
        }
        else if (text_heard_trimmed == "step" ||
            text_heard_trimmed.find("stop") != std::string::npos ||
            text_heard_trimmed.find("стоп") != std::string::npos ||
            text_heard_trimmed.find("тановись") != std::string::npos ||
            text_heard_trimmed.find("хватит") != std::string::npos) {
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
            text_heard_trimmed.find("ты здесь") != std::string::npos ||
            text_heard_trimmed.find("ты меня слышишь") != std::string::npos ||
            text_heard_trimmed.find("ты слышишь меня") != std::string::npos ||
            (text_heard_trimmed.find("то ты думаешь") != std::string::npos &&
                text_heard != "Что ты думаешь?" &&
                text_heard_trimmed.find("то ты об этом думаешь") == std::string::npos) ||
            (text_heard_trimmed.find("то ты об этом думаешь") != std::string::npos &&
                text_heard != "Что ты об этом думаешь?")) {
            user_command = "call";
        }

        if (!user_command.empty() && !new_command_allowed) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - last_command_time).count();
            if (elapsed >= 2) {
                new_command_allowed = 1;
            }
        }

        // === ОБРАБОТКА КОМАНДЫ REGENERATE ===
        if (user_command == "regenerate" ||
            text_heard_trimmed == "please regenerate" ||
            text_heard_trimmed == "regenerate please" ||
            text_heard_trimmed == "regenerate, please" ||
            text_heard_trimmed == "try again please" ||
            text_heard_trimmed == "try again, please" ||
            text_heard_trimmed == "please try again" ||
            text_heard_trimmed == "try again") {
            if (new_command_allowed) {
                std::string dummy;
                allow_xtts_file(dummy, 0);
                g_cancel_tts_requests.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                new_command_allowed = 0;
                last_command_time = std::chrono::steady_clock::now();
                first_dialog_pair = true;
                if (!past_prev_arr.empty()) {
                    n_past_prev = past_prev_arr.back();
                    past_prev_arr.pop_back();
                    int rollback_num = static_cast<int>(embd_inp.size()) -
                        n_past_prev;
                    if (rollback_num > 0 &&
                        rollback_num <= static_cast<int>(embd_inp.size())) {
                        embd_inp.erase(embd_inp.end() - rollback_num,
                            embd_inp.end());
                        printf(" [regenerating %d tokens. Context: %zu]\n",
                            rollback_num, embd_inp.size());
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        llama_memory_seq_rm(llama_get_memory(ctx_llama), 0,
                            static_cast<int>(embd_inp.size()), -1);
                        text_heard = text_heard_prev;
                        text_heard_trimmed = "";
                        std::string text_to_respeak_safe;
                        {
                            std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                            text_to_respeak_safe = g_last_tts_text;
                        }
                        if (!text_to_respeak_safe.empty()) {
                            std::string voice_copy = current_voice;
                            safe_thread_emplace(threads,
                                [text_to_respeak_safe, voice_copy, params]() {
                                    send_tts_async(text_to_respeak_safe,
                                        voice_copy, params.language,
                                        params.xtts_url);
                                });
                        }
                        {
                            std::string dummy2;
                            allow_xtts_file(dummy2, 1);
                        }
                    }
                }
            }
        }
        // === ОБРАБОТКА КОМАНДЫ DELETE ===
        else if (user_command == "delete" ||
            text_heard_trimmed == "please delete" ||
            text_heard_trimmed == "please delete the last message" ||
            text_heard_trimmed == "delete please" ||
            text_heard_trimmed == "delete, please") {
            if (new_command_allowed) {
                if (!past_prev_arr.empty()) {
                    if (text_heard_trimmed == "delete two messages" ||
                        text_heard_trimmed == "удали 2 сообщения" ||
                        text_heard_trimmed == "удали два сообщения" ||
                        text_heard_trimmed == "please donate to the messages") {
                        n_past_prev = past_prev_arr.back();
                        past_prev_arr.pop_back();
                    }
                    else if (text_heard_trimmed == "delete three messages" ||
                        text_heard_trimmed == "удали 3 сообщения" ||
                        text_heard_trimmed == "удали три сообщения") {
                        n_past_prev = past_prev_arr.back();
                        past_prev_arr.pop_back();
                        n_past_prev = past_prev_arr.back();
                        past_prev_arr.pop_back();
                    }
                    n_past_prev = past_prev_arr.back();
                    past_prev_arr.pop_back();
                    int rollback_num = static_cast<int>(embd_inp.size()) -
                        n_past_prev;
                    if (rollback_num) {
                        embd_inp.erase(embd_inp.end() - rollback_num,
                            embd_inp.end());
                        if (params.verbose) {
                            printf(" deleting %d tokens. Tokens in ctx: %zu\n",
                                rollback_num, embd_inp.size());
                        }
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        llama_memory_seq_rm(llama_get_memory(ctx_llama), 0,
                            static_cast<int>(embd_inp.size()), -1);
                        text_heard = "";
                        text_heard_trimmed = "";
                        last_command_time = std::chrono::steady_clock::now();
                        new_command_allowed = 0;
                        std::string text_for_deleted_tts = "Deleted";
                        if (!text_for_deleted_tts.empty()) {
                            std::string voice_copy = current_voice;
                            safe_thread_emplace(threads,
                                [text_for_deleted_tts, voice_copy, params]() {
                                    send_tts_async(text_for_deleted_tts,
                                        voice_copy, params.language,
                                        params.xtts_url);
                                });
                        }
                    }
                }
                else {
                    printf("Nothing to delete more\n");
                    send_tts_async("Nothing to delete more", "ux",
                        params.language);
                }
            }
            audio.clear();
        }
        // === ОБРАБОТКА КОМАНДЫ RESET ===
        else if (user_command == "reset") {
            if (new_command_allowed) {
                // [ПАТЧ] Безопасный reset с блокировкой аудио-потока
                // 1. Блокируем аудио-поток
                g_reset_in_progress.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                // 2. Останавливаем TTS
                std::string dummy;
                allow_xtts_file(dummy, 0);
                g_cancel_tts_requests.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                if (!past_prev_arr.empty()) {
                    n_past_prev = past_prev_arr.front();
                    past_prev_arr.clear();
                    int rollback_num = static_cast<int>(embd_inp.size()) -
                        n_past_prev;
                    if (rollback_num) {
                        printf(" [Resetting context of %zd tokens.]\n",
                            embd_inp.size());
                        {
                            std::lock_guard<std::mutex> lock(g_llama_mutex);
                            llama_batch_free(batch);
                            if (ctx_llama) {
                                llama_free(ctx_llama);
                                ctx_llama = nullptr;
                            }
                            // Пересоздаём контекст
                            ctx_llama = llama_init_from_model(model_llama,
                                lcparams);
                            if (!ctx_llama) {
                                fprintf(stderr, "%s : ERROR: Failed to "
                                    "reinitialize llama context on reset\n",
                                    __func__);
                                g_reset_in_progress.store(false);
                                g_reset_cv.notify_all();
                                return 1;
                            }
                            batch = llama_batch_init(2048, 0, 1);
                            embd_inp = ::llama_tokenize(ctx_llama,
                                prompt_llama, true);
                            if (embd_inp.empty()) {
                                fprintf(stderr, "%s : ERROR: Failed to "
                                    "tokenize prompt after reset\n",
                                    __func__);
                                g_reset_in_progress.store(false);
                                g_reset_cv.notify_all();
                                return 1;
                            }
                            if (embd_inp.size() > 2048) {
                                fprintf(stderr, "%s : FATAL ERROR: Prompt "
                                    "size (%zu tokens) exceeds batch "
                                    "limit (2048)\n",
                                    __func__, embd_inp.size());
                                g_reset_in_progress.store(false);
                                g_reset_cv.notify_all();
                                return 1;
                            }
                            batch.n_tokens = static_cast<int>(embd_inp.size());
                            for (int i = 0; i < batch.n_tokens; i++) {
                                batch.token[i] = embd_inp[i];
                                batch.pos[i] = i;
                                batch.n_seq_id[i] = 1;
                                batch.seq_id[i][0] = 0;
                                batch.logits[i] =
                                    (i == batch.n_tokens - 1) ? 1 : 0;
                            }
                            if (llama_decode(ctx_llama, batch)) {
                                fprintf(stderr, "%s : failed to decode "
                                    "after reset\n", __func__);
                                g_reset_in_progress.store(false);
                                g_reset_cv.notify_all();
                                return 1;
                            }
                        }
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = static_cast<int>(embd_inp.size());
                        printf(" [Context is now %zu/%d tokens. n_past: %d]\n",
                            embd_inp.size(), params.ctx_size, n_past);
                        text_heard = "";
                        text_heard_trimmed = "";
                        send_tts_async("Reset whole context", current_voice,
                            params.language, params.xtts_url);
                        {
                            std::string dummy2;
                            allow_xtts_file(dummy2, 1);
                        }
                        new_command_allowed = 0;
                        last_command_time = std::chrono::steady_clock::now();
                    }
                }
                else {
                    printf(" [Nothing to reset more]\n");
                    send_tts_async("Nothing to reset more", params.xtts_voice,
                        params.language, params.xtts_url);
                }

                // 5. Разблокируем аудио-поток
                g_reset_in_progress.store(false);
                g_reset_cv.notify_all();
            }
            audio.clear();
            continue;
        }
        // === ОБРАБОТКА КОМАНДЫ STOP ===
        if (user_command == "stop") {
            std::string lower_text = LowerCase(text_heard_trimmed);
            bool is_stop_command = (
                lower_text == "стоп" || lower_text == "stop" ||
                lower_text == "хватит" || lower_text == "остановись" ||
                lower_text.find(params.bot_name + " стоп") != std::string::npos ||
                lower_text.find(params.bot_name + " stop") != std::string::npos
                );
            if (!is_stop_command) {
                user_command.clear();
            }
            else {
                std::string dummy;
                allow_xtts_file(dummy, 0);
                g_cancel_tts_requests.store(true);
                printf("\n[Stop]\n");
                fflush(stdout);
                text_heard.clear();
                text_heard_trimmed.clear();
                user_typed.clear();
                user_typed_this = false;
                g_interrupt_reason.store(InterruptReason::MANUAL_STOP);
                g_interrupt_processed.store(true);
                g_bot_state.store(BotState::INTERRUPTED);
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    g_hotkey_pressed = "Ctrl+Space";
                }
                text_heard = "";
                text_heard_trimmed = "";
                continue;
            }
        }
        // === ОБРАБОТКА КОМАНДЫ TIME ===
        else if (text_heard_trimmed.find("время") != std::string::npos ||
            text_heard_trimmed.find("который час") != std::string::npos ||
            text_heard_trimmed.find("what time") != std::string::npos ||
            text_heard_trimmed.find("сколько времени") != std::string::npos ||
            text_heard_trimmed.find("сколько время") != std::string::npos) {
            user_command = "time";
        }
        // === ОБРАБОТКА КОМАНДЫ GOOGLE ===
        else if (user_command == "google") {
            auto speak_direct = [&](const std::string& msg) {
                if (msg.empty()) return;
                std::string msg_copy = msg;
                std::string voice_copy = current_voice;
                try {
                    safe_thread_emplace(threads,
                        [msg_copy, voice_copy, params]() {
                            send_tts_async(msg_copy, voice_copy,
                                params.language, params.xtts_url);
                        });
                }
                catch (const std::exception& e) {
                    if (params.verbose) {
                        fprintf(stderr, "[google] TTS thread spawn failed: "
                            "%s\n", e.what());
                    }
                }
                };
            std::string q = ParseCommandAndGetKeyword(text_heard_trimmed,
                user_command);
            if (q.empty()) {
                fprintf(stdout, "[google] can't get keyword from: %s\n",
                    text_heard_trimmed.c_str());
                speak_direct("Извините, не удалось понять, что именно "
                    "вы хотите найти.");
                user_typed.clear();
                user_typed_this = false;
            }
            else {
                std::string google_search_msg =
                    "Ищу информацию по запросу: " + q;
                std::string voice_copy = current_voice;
                std::string lang_copy = params.language;
                std::string url_copy = params.xtts_url;
                safe_thread_emplace(threads,
                    [google_search_msg, voice_copy, lang_copy, url_copy]() {
                        send_tts_async(google_search_msg, voice_copy,
                            lang_copy, url_copy);
                    });
                const std::string url = params.google_url + "google?q=" +
                    UrlEncode(q);
                std::string resp = send_curl(url);
                if (resp.empty()) {
                    fprintf(stdout, "[google] empty response for (%s)\n",
                        q.c_str());
                    std::string error_msg =
                        "Извините, не удалось найти информацию по запросу: " + q;
                    std::string voice_copy_err = current_voice;
                    safe_thread_emplace(threads,
                        [error_msg, voice_copy_err, params]() {
                            send_tts_async(error_msg, voice_copy_err,
                                params.language, params.xtts_url);
                        });
                }
                else {
                    if (params.verbose) {
                        fprintf(stdout, "[google] resp (%s): %s\n",
                            q.c_str(), resp.c_str());
                    }
                    auto truncate_smart = [](std::string s,
                        size_t hard = 600,
                        size_t prefer = 420) {
                        if (s.size() <= hard) return s;
                        size_t cut = s.find_last_of(".!?");
                        if (cut != std::string::npos &&
                            cut >= std::min(prefer, hard)) {
                            s.erase(cut + 1);
                        }
                        else {
                            s.erase(std::min(hard, s.size()));
                            s += "...";
                        }
                        return s;
                        };
                    resp = truncate_smart(resp);
                    std::string llm_prompt = params.person + ": " +
                        params.bot_name +
                        ", пожалуйста, кратко изложи основное из текста, "
                        "найденного по запросу \"" + q + "\": " + resp;
                    text_heard = llm_prompt;
                    user_typed_this = true;
                }
            }
            audio.clear();
            user_typed.clear();
        }
        // === ОБРАБОТКА КОМАНДЫ TIME (вывод) ===
        else if (user_command == "time") {
            std::time_t t_now = std::time(nullptr);
            std::tm tm_local_now{};
#ifdef _WIN32
            localtime_s(&tm_local_now, &t_now);
#else
            localtime_r(&t_now, &tm_local_now);
#endif
            int hour = tm_local_now.tm_hour;
            int minute = tm_local_now.tm_min;
            char time_buffer[64];
            std::snprintf(time_buffer, sizeof(time_buffer),
                "Сейчас %02d:%02d", hour, minute);
            std::string llm_prompt = params.person +
                ": Который час?\n" + params.bot_name + ":" +
                std::string(time_buffer);
            text_heard = llm_prompt;
            user_typed_this = true;
            audio.clear();
            user_typed.clear();
            text_heard_trimmed = "";
        }
        // === ОБРАБОТКА КОМАНДЫ CALL ===
        else if (user_command == "call") {
            // [ПАТЧ] Блокировка если multi-chars выключен
            if (!params.multi_chars) {
                if (params.verbose) {
                    fprintf(stdout, "[call] Игнорировано: multi-chars режим выключен\n");
                }
                user_command = "";
                text_heard = "";
                text_heard_trimmed = "";
                continue;
            } else {
                std::string q = ParseCommandAndGetKeyword(text_heard, user_command);
                if (!q.empty()) {
                    // [ПАТЧ] Защита от переключения на самого себя
                    if (q == params.bot_name) {
                        fprintf(stdout, "Я уже %s\n", params.bot_name.c_str());
                        send_tts_async("Я уже " + params.bot_name, current_voice, params.language);
                        user_command = "";
                        text_heard = "";
                        text_heard_trimmed = "";
                        continue;
                    }
                    fprintf(stdout, "Переключаюсь на бота: %s", q.c_str());
                    std::string old_bot_name = params.bot_name;
                    params.bot_name = q;
                    update_antiprompts(params.person, params.bot_name);
                    if (params.verbose) {
                        fprintf(stdout, " [antiprompts updated for bot: %s]\n",
                            params.bot_name.c_str());
                    }
                }
                else {
                    fprintf(stdout, "Error: can't find bot name in "
                        "text_heard_trimmed: %s",
                        text_heard_trimmed.c_str());
                }
            }
        }

        // [21.25] Начало генерации LLaMA
        int translation_is_going = 0;
        int n_embd_inp_before_trans = 0;
        int tokens_in_reply = 0;
        std::string current_voice_tmp = "";
        reply_part = 0;
        llama_start_generation_time = 0.0f;
        llama_start_time = get_current_time_ms();

        if (text_heard.empty() || force_speak) {
            audio.clear();
            {
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                g_hotkey_pressed = "";
            }
            force_speak = false;
            continue;
        }
        trim(text_heard);

        text_heard_prev = text_heard;
        n_past_prev = static_cast<int>(embd_inp.size());
        past_prev_arr.push_back(static_cast<int>(embd_inp.size()));
        if (past_prev_arr.size() > MAX_PAST_PREV_SIZE) {
            past_prev_arr.erase(past_prev_arr.begin());
        }

        std::string translation_full = "";
        std::string bot_name_current = params.bot_name;
        std::string bot_name_current_ru = params.bot_name;
        std::string text_heard_with_instruct = text_heard;
        if (params.translate)
            bot_name_current_ru = translit_en_ru(params.bot_name);
        int n_comas = 0;

        // Форматирование реплики пользователя
        if (last_output_has_username && !user_typed_this) {
            text_heard.insert(0, 1, ' ');
            text_heard_with_instruct.insert(0, 1, ' ');
        }
        else {
            if (last_output_has_EOT) {
                text_heard.insert(0, "\n" + params.person + chat_symb + " ");
                text_heard_with_instruct.insert(0,
                    "\n" + params.instruct_preset_data["user_message_prefix"] +
                    "\n" + params.person + chat_symb + " ");
            }
            else {
                text_heard.insert(0, "\n" + params.person + chat_symb + " ");
                text_heard_with_instruct.insert(0,
                    params.instruct_preset_data["bot_message_suffix"] +
                    "\n" + params.instruct_preset_data["user_message_prefix"] +
                    "\n" + params.person + chat_symb + " ");
            }
        }
        text_heard += "\n" + params.bot_name + chat_symb;
        text_heard_with_instruct +=
            params.instruct_preset_data["user_message_suffix"] +
            "\n" + params.instruct_preset_data["bot_message_prefix"] +
            "\n" + params.bot_name + chat_symb;

        // Вывод в консоль
        std::string display_text;
        if (user_typed_this) {
            display_text = user_typed;
        }
        else {
            display_text = text_heard_trimmed;
        }
        std::string clean_prefix = params.person + chat_symb;
        std::string clean_prefix_space = params.person + " " + chat_symb;
        while (display_text.find(clean_prefix) == 0) {
            display_text.erase(0, clean_prefix.length());
        }
        while (display_text.find(clean_prefix_space) == 0) {
            display_text.erase(0, clean_prefix_space.length());
        }
        trim(display_text);
        if (!display_text.empty() && display_text[0] == ':') {
            display_text.erase(0, 1);
            ltrim(display_text);
        }
        if (!first_dialog_pair) {
            printf("\n");
        }
        first_dialog_pair = false;
        {
            std::lock_guard<std::mutex> console_lock(g_console_mutex);
            printf("\r\033[K");
            fflush(stdout);
            if (!display_text_for_ui.empty()) {
                printf("\033[32m%s%s\033[0m %s\n",
                    params.person.c_str(), chat_display.c_str(),
                    display_text_for_ui.c_str());
            }
            fflush(stdout);
            g_ui_state.store(UiState::UI_BOT_LINE);
            printf("\033[1m%s%s\033[0m ", params.bot_name.c_str(), chat_display.c_str());
            fflush(stdout);
        }

        // Восстановление семафора TTS
        {
            std::string dummy;
            allow_xtts_file(dummy, 1);
        }

        // Токенизация
        embd = ::llama_tokenize(ctx_llama, text_heard_with_instruct, false);
        input_tokens_count = static_cast<int>(embd.size());
        if (!path_session.empty()) {
            session_tokens.insert(session_tokens.end(),
                embd.begin(), embd.end());
        }
        float temp_next = params.temp;
        int n_discard = 0;
        int n_left = 0;

        // [21.26] Цикл генерации LLaMA
        bool done = false;
        std::string text_to_speak;
        std::string full_response_text;
        int new_tokens = 0;
        bool first_token_after_bot = true;
        std::string tts_smart_buffer;
        int n_ctx_before_generation = static_cast<int>(embd_inp.size());
        bool was_interrupted = false;
        {
            first_token_after_bot = true;
            tts_smart_buffer.clear();
            g_bot_state.store(BotState::GENERATING);
            g_interrupt_processed.store(false);
        }

        while (true) {
            // Проверка прерывания от audio_input_thread (barge-in)
            InterruptReason reason = g_interrupt_reason.load();
            if (reason != InterruptReason::NONE) {
                std::string dummy;
                allow_xtts_file(dummy, 0);
                // [ПАТЧ] МЯГКОЕ ПРЕРЫВАНИЕ: НЕ отменяем текущие TTS-запросы
                // g_cancel_tts_requests.store(true);
                done = true;
                was_interrupted = true;
                // [ПАТЧ] НЕ ОЧИЩАЕМ буфер TTS
                // text_to_speak = "";
                // tts_smart_buffer.clear();
                {
                    std::lock_guard<std::mutex> lock(g_console_mutex);
                    printf("\r\033[K");
                    fflush(stdout);
                }
                g_ui_state.store(UiState::UI_IDLE);
                g_bot_state.store(BotState::IDLE);
                first_dialog_pair = true;
                if (params.verbose) {
                    printf(" [interrupted by %d]\n",
                        static_cast<int>(reason));
                }
                break;
            }

            // Проверка горячих клавиш
            {
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (!g_hotkey_pressed.empty() &&
                    g_hotkey_pressed != "Alt") {
                    std::string dummy;
                    allow_xtts_file(dummy, 0);
                    g_cancel_tts_requests.store(true);
                    g_interrupt_reason.store(InterruptReason::HOTKEY_STOP);
                    g_interrupt_processed.store(true);
                    g_bot_state.store(BotState::INTERRUPTED);
                    done = true;
                    was_interrupted = true;
                    text_to_speak = "";
                    tts_smart_buffer.clear();
                    g_hotkey_pressed = "";
                    first_dialog_pair = true;
                    break;
                }
            }

            // Проверка команды STOP из очереди ввода
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (!input_queue.empty()) {
                    std::string cmd = input_queue.front();
                    std::string cmd_lower = LowerCase(cmd);
                    if (cmd_lower.find("стоп") != std::string::npos ||
                        cmd_lower.find("stop") != std::string::npos ||
                        cmd_lower.find("хватит") != std::string::npos ||
                        cmd_lower.find("остановись") != std::string::npos) {
                        std::string dummy;
                        allow_xtts_file(dummy, 0);
                        g_cancel_tts_requests.store(true);
                        g_interrupt_reason.store(InterruptReason::MANUAL_STOP);
                        g_interrupt_processed.store(true);
                        g_bot_state.store(BotState::INTERRUPTED);
                        done = true;
                        was_interrupted = true;
                        text_to_speak = "";
                        tts_smart_buffer.clear();
                        first_dialog_pair = true;
                        break;
                    }
                }
            }

            if (new_tokens > params.n_predict) break;
            new_tokens++;

            // Сдвиг контекста (context shift)
            if (embd.size() > 0) {
                if (n_past + static_cast<int>(embd.size()) > n_ctx) {
                    const llama_vocab* vocab_llama_local =
                        llama_model_get_vocab(model_llama);
                    const int n_left_local = std::max(0, n_past - n_keep);
                    int n_discard_local = 0;
                    if (n_left_local > 0) {
                        n_discard_local = std::max(1, n_left_local / 4);
                        n_discard_local = std::min(n_discard_local,
                            n_left_local);
                    }
                    bool context_updated = false;
                    if (n_discard_local > 0 &&
                        n_keep + n_discard_local <= n_past) {
                        if (n_keep >= 0 &&
                            n_keep + n_discard_local <=
                            static_cast<int>(embd_inp.size())) {
                            llama_memory_seq_rm(
                                llama_get_memory(ctx_llama), 0,
                                n_keep, n_keep + n_discard_local);
                            if (n_keep + n_discard_local < n_past) {
                                llama_memory_seq_add(
                                    llama_get_memory(ctx_llama), 0,
                                    n_keep + n_discard_local, n_past,
                                    -n_discard_local);
                            }
                            embd_inp.erase(
                                embd_inp.begin() + n_keep,
                                embd_inp.begin() + n_keep + n_discard_local);
                            if (!path_session.empty()) {
                                if (session_tokens.size() > embd_inp.size()) {
                                    session_tokens.resize(embd_inp.size());
                                }
                                bool need_full_resync = false;
                                int check_limit = std::min(
                                    n_keep,
                                    static_cast<int>(std::min(
                                        embd_inp.size(),
                                        session_tokens.size())));
                                for (int i_check = 0;
                                    i_check < check_limit; i_check++) {
                                    if (embd_inp[i_check] !=
                                        session_tokens[i_check]) {
                                        need_full_resync = true;
                                        break;
                                    }
                                }
                                if (need_full_resync) {
                                    session_tokens = embd_inp;
                                }
                                else if (session_tokens.size() >
                                    embd_inp.size()) {
                                    session_tokens.resize(embd_inp.size());
                                }
                                if (!path_session.empty()) {
                                    llama_state_save_file(
                                        ctx_llama, path_session.c_str(),
                                        session_tokens.data(),
                                        session_tokens.size());
                                }
                            }
                            context_updated = true;
                            if (params.verbose) {
                                printf("\n[Сдвиг: удал. %d ток, ост: %zu]\n",
                                    n_discard_local, embd_inp.size());
                            }
                        }
                    }
                    if (!context_updated) {
                        size_t new_size = std::min(
                            static_cast<size_t>(std::max(0, n_keep)),
                            embd_inp.size());
                        if (new_size < embd_inp.size()) {
                            embd_inp.resize(new_size);
                            if (!path_session.empty() &&
                                !session_tokens.empty()) {
                                session_tokens.resize(std::min(
                                    static_cast<size_t>(std::max(0, n_keep)),
                                    session_tokens.size()));
                            }
                        }
                    }
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    if (vocab_llama_local) {
                        const llama_token bos_token =
                            llama_token_bos(vocab_llama_local);
                        if (!embd_inp.empty() &&
                            embd_inp[0] != bos_token) {
                            embd_inp.insert(embd_inp.begin(), bos_token);
                            if (!session_tokens.empty()) {
                                session_tokens.insert(
                                    session_tokens.begin(), bos_token);
                            }
                            n_past = static_cast<int>(embd_inp.size());
                            n_session_consumed = n_past;
                        }
                    }
                    path_session = "";
                    embd.clear();
                    text_to_speak = "";
                    past_prev_arr.clear();
                    continue;
                }
            }

            // Повторное использование сессии
            if (n_session_consumed <
                static_cast<int>(session_tokens.size())) {
                size_t i = 0;
                int max_check = std::min(
                    static_cast<int>(embd.size()),
                    static_cast<int>(session_tokens.size()) -
                    n_session_consumed);
                for (; i < static_cast<size_t>(max_check); i++) {
                    if (n_session_consumed >=
                        static_cast<int>(session_tokens.size())) {
                        break;
                    }
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }
                    embd_inp.push_back(embd[i]);
                    n_session_consumed++;
                    if (n_session_consumed >=
                        static_cast<int>(session_tokens.size())) {
                        i++;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(),
                        embd.begin() +
                        static_cast<std::ptrdiff_t>(i));
                }
                n_past = static_cast<int>(embd_inp.size());
            }
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(),
                    embd.begin(), embd.end());
                n_session_consumed =
                    static_cast<int>(session_tokens.size());
            }

            // Подготовка батча
            {
                if (embd.empty()) {
                    embd.clear();
                    continue;
                }
                if (embd.size() > 2048) {
                    fprintf(stderr, "ERROR: Input sequence too long "
                        "(%zu tokens). Max batch size is 2048.\n",
                        embd.size());
                    embd.clear();
                    continue;
                }
                batch.n_tokens = static_cast<int>(embd.size());
                for (int i = 0; i < batch.n_tokens; ++i) {
                    batch.logits[i] = 0;
                }
                for (int i = 0; i < batch.n_tokens; ++i) {
                    batch.token[i] = embd[i];
                    batch.pos[i] = n_past + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] =
                        (i == batch.n_tokens - 1) ? 1 : 0;
                }
            }

            // Декодирование
            {
                std::lock_guard<std::mutex> lock(g_llama_mutex);
                if (!ctx_llama) {
                    fprintf(stderr, "\n[Context was reset during "
                        "generation - aborting]\n");
                    done = true;
                    break;
                }
                if (llama_decode(ctx_llama, batch)) {
                    fprintf(stderr, "%s : failed to decode\n", __func__);
                    if (ctx_llama) {
                        llama_memory_seq_rm(
                            llama_get_memory(ctx_llama), 0, n_past, -1);
                    }
                    embd.clear();
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    continue;
                }
            }

            embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
            n_past = static_cast<int>(embd_inp.size());
            embd.clear();
            if (done) break;

            // Сэмплирование токена
            std::string out_token_str = "";
            char out_token_symbol;
            if (llama_start_generation_time == 0.0f)
                llama_start_generation_time = get_current_time_ms();
            {
                if (!path_session.empty() && need_to_save_session) {
                    need_to_save_session = false;
                    llama_state_save_file(ctx_llama, path_session.c_str(),
                        session_tokens.data(), session_tokens.size());
                }
                llama_token id = 0;
                int person_name_is_found = 0;
                int bot_name_is_found = 0;
                if (temp != temp_next) {
                    id = llama_sampler_sample(smpl_high_temp, ctx_llama, -1);
                    temp = temp_next = params.temp;
                }
                else {
                    id = llama_sampler_sample(smpl, ctx_llama, -1);
                }

                // Проверка на стоп-токены
                bool is_stop_token = false;
                for (int i = 0; i < special_token_count; i++) {
                    if (id == special_token_ids[i]) {
                        is_stop_token = true;
                        break;
                    }
                }
                if (id == llama_vocab_eos(vocab_llama)) {
                    is_stop_token = true;
                }
                if (is_stop_token) {
                    done = true;
                    break;
                }

                if (id != llama_vocab_eos(vocab_llama)) {
                    embd.push_back(id);
                    out_token_str = llama_token_to_piece(ctx_llama, id);

                    // Замена плейсхолдеров {0}-{5}
                    size_t pos0 = out_token_str.find("{0}");
                    if (pos0 != std::string::npos)
                        out_token_str.replace(pos0, 3, params.person);
                    size_t pos1 = out_token_str.find("{1}");
                    if (pos1 != std::string::npos)
                        out_token_str.replace(pos1, 3, params.bot_name);
                    size_t pos2 = out_token_str.find("{2}");
                    if (pos2 != std::string::npos)
                        out_token_str.replace(pos2, 3, time_str);
                    size_t pos3 = out_token_str.find("{3}");
                    if (pos3 != std::string::npos)
                        out_token_str.replace(pos3, 3, year_str);
                    size_t pos5 = out_token_str.find("{5}");
                    if (pos5 != std::string::npos)
                        out_token_str.replace(pos5, 3, ymd);
                    if (out_token_str == "{0" || out_token_str == "{0}")
                        out_token_str = params.person;
                    else if (out_token_str == "{1" || out_token_str == "{1}")
                        out_token_str = params.bot_name;
                    else if (out_token_str == "{2" || out_token_str == "{2}")
                        out_token_str = time_str;
                    else if (out_token_str == "{3" || out_token_str == "{3}")
                        out_token_str = year_str;
                    else if (out_token_str == "{5" || out_token_str == "{5}")
                        out_token_str = ymd;
                    else if (out_token_str == "}" &&
                        !text_to_speak.empty()) {
                        if (text_to_speak.size() >= 2) {
                            std::string last2 = text_to_speak.substr(
                                text_to_speak.size() - 2);
                            if (last2 == "{0" || last2 == "{1" ||
                                last2 == "{2" || last2 == "{3" ||
                                last2 == "{5") {
                                text_to_speak.pop_back();
                                out_token_str = "";
                            }
                        }
                    }

                    // Фильтрация спецтокенов
                    bool is_special_id = false;
                    for (int si = 0; si < special_token_count; si++) {
                        if (id == special_token_ids[si]) {
                            is_special_id = true;
                            break;
                        }
                    }
                    if (!is_special_id) {
                        llama_token eos_token =
                            llama_vocab_eos(vocab_llama);
                        if (id == eos_token) {
                            is_special_id = true;
                        }
                    }

                    // pending_fragment
                    std::string pending_fragment = "";
                    int pending_count = 0;
                    static const int MAX_PENDING = 20;
                    bool looks_like_special = false;
                    if (!out_token_str.empty()) {
                        for (size_t ci = 0; ci < out_token_str.size();
                            ci++) {
                            char c = out_token_str[ci];
                            if (c == '<' || c == '|' || c == '>') {
                                looks_like_special = true;
                                break;
                            }
                        }
                        std::string lower_tok = out_token_str;
                        std::transform(lower_tok.begin(), lower_tok.end(), lower_tok.begin(), ::tolower);
                        if (lower_tok.find("eo") != std::string::npos ||
                            lower_tok.find("eot") != std::string::npos ||
                            lower_tok.find("im_") != std::string::npos ||
                            lower_tok.find("end_") != std::string::npos ||
                            lower_tok.find("start_") != std::string::npos ||
                            lower_tok.find("header") != std::string::npos) {
                            looks_like_special = true;
                        }
                    }

                    bool should_print = true;
                    if (is_special_id) {
                        should_print = false;
                        pending_fragment = "";
                        pending_count = 0;
                    }
                    else if (looks_like_special &&
                        !out_token_str.empty()) {
                        pending_fragment += out_token_str;
                        pending_count++;
                        bool has_open =
                            (pending_fragment.find("<|") !=
                                std::string::npos);
                        bool has_close =
                            (pending_fragment.find("|>") !=
                                std::string::npos);
                        if (has_open && has_close) {
                            pending_fragment = "";
                            pending_count = 0;
                            should_print = false;
                        }
                        else if (pending_count >= MAX_PENDING) {
                            printf("%s", pending_fragment.c_str());
                            fflush(stdout);
                            text_to_speak += pending_fragment;
                            tokens_in_reply +=
                                utf8_length(pending_fragment);
                            pending_fragment = "";
                            pending_count = 0;
                            should_print = true;
                        }
                        else {
                            should_print = false;
                        }
                    }
                    else {
                        if (!pending_fragment.empty()) {
                            printf("%s", pending_fragment.c_str());
                            fflush(stdout);
                            text_to_speak += pending_fragment;
                            tokens_in_reply +=
                                utf8_length(pending_fragment);
                            pending_fragment = "";
                            pending_count = 0;
                        }
                    }

                    if (first_token_after_bot && !out_token_str.empty() &&
                        out_token_str[0] == ' ') {
                        out_token_str.erase(0, 1);
                    }
                    first_token_after_bot = false;

                    if (should_print && !out_token_str.empty()) {
                        std::string display_str = out_token_str;
                        try {
                            static const std::regex re_emotion_parens(R"(\([^)]{2,60}\))", std::regex::ECMAScript);
                            static const std::regex re_emotion_stars(R"(\*[^*]{2,60}\*)", std::regex::ECMAScript);
                            display_str = std::regex_replace(display_str, re_emotion_parens, "");
                            display_str = std::regex_replace(display_str, re_emotion_stars, "");
                            display_str = std::regex_replace(display_str,
                                std::regex(R"(<\|[^>]*\|>)"), "");
                            display_str = std::regex_replace(display_str,
                                std::regex(R"(\b(assistant|system|user|end_header_id|eot_id|start_header_id|eo)\b)"), "");
                        }
                        catch (const std::regex_error&) {
                            display_str = string_replace_all(display_str,
                                "<|start_header_id|>", "");
                            display_str = string_replace_all(display_str,
                                "<|end_header_id|>", "");
                            display_str = string_replace_all(display_str,
                                "<|eot_id|>", "");
                            display_str = string_replace_all(display_str,
                                "<|im_start|>", "");
                            display_str = string_replace_all(display_str,
                                "<|im_end|>", "");
                            display_str = string_replace_all(display_str,
                                "<|eo|>", "");
                            display_str = string_replace_all(display_str,
                                "assistant", "");
                            display_str = string_replace_all(display_str,
                                "system", "");
                            display_str = string_replace_all(display_str,
                                "user", "");
                        }
                        if (!display_str.empty()) {
                            printf("%s", display_str.c_str());
                            fflush(stdout);
                        }
                        text_to_speak += out_token_str;
                    }
                    if (should_print) {
                        tokens_in_reply += utf8_length(out_token_str);
                    }

                    // Seqrep
                    if (params.seqrep) {
                        if (utf8_length(last_output_needle) > 25)
                            last_output_needle = utf8_substr(
                                last_output_needle, 5,
                                utf8_length(last_output_needle) - 5);
                        last_output_needle += out_token_str;
                        out_token_symbol =
                            out_token_str[out_token_str.size() - 1];
                        if (out_token_symbol == ' ' ||
                            out_token_symbol == '.' ||
                            out_token_symbol == ',' ||
                            out_token_symbol == '!' ||
                            out_token_symbol == '?') {
                            if (utf8_length(last_output_buffer) > 300 &&
                                utf8_length(last_output_needle) >= 20 &&
                                last_output_buffer.find(last_output_needle)
                                != std::string::npos) {
                                printf(" [LOOP: %s] (length: %d)\n",
                                    last_output_needle.c_str(),
                                    utf8_length(last_output_needle));
                                int symbols_to_delete = static_cast<int>(
                                    utf8_length(last_output_needle) * 1);
                                const std::vector<llama_token> tokens_to_del =
                                    llama_tokenize(ctx_llama,
                                        last_output_needle.c_str(), false);
                                int rollback_num =
                                    static_cast<int>(tokens_to_del.size());
                                if (rollback_num) {
                                    embd_inp.erase(
                                        embd_inp.end() - rollback_num,
                                        embd_inp.end());
                                    n_past =
                                        static_cast<int>(embd_inp.size());
                                    n_session_consumed = n_past;
                                    llama_memory_seq_rm(
                                        llama_get_memory(ctx_llama), 0,
                                        static_cast<int>(embd_inp.size()),
                                        -1);
                                    text_to_speak = utf8_substr(
                                        text_to_speak, 0,
                                        utf8_length(text_to_speak) -
                                        symbols_to_delete);
                                    last_output_needle = utf8_substr(
                                        last_output_needle, 0,
                                        utf8_length(last_output_needle) -
                                        symbols_to_delete);
                                    last_output_buffer = utf8_substr(
                                        last_output_buffer, 0,
                                        utf8_length(last_output_buffer) -
                                        symbols_to_delete);
                                    temp_next = 1.8f;
                                }
                            }
                        }
                        if (utf8_length(last_output_buffer) > 1000)
                            last_output_buffer = utf8_substr(
                                last_output_buffer, 100,
                                last_output_buffer.size() - 100);
                        last_output_buffer += out_token_str;
                    }

                    // [ПАТЧ] Обнаружение имён персонажей (только с multi-chars)
                    if (params.multi_chars) {
                        if (text_to_speak == "\n" + params.person + ":") {
                            person_name_is_found = 1;
                            translation_is_going = 0;
                        }
                        else if (!text_to_speak.empty() && text_to_speak[0] == '\n' &&
                                 text_to_speak[text_to_speak.size() - 1] == ':' &&
                                 text_to_speak.size() < 10) {
                            std::string detected_name = text_to_speak.substr(1, text_to_speak.size() - 2);
                            if (detected_name != params.bot_name) {
                                bot_name_is_found = 1;
                                bot_name_current = detected_name;
                                if (params.translate)
                                    bot_name_current_ru = translit_en_ru(bot_name_current);
                                translation_full = "";
                                text_to_speak = "";
                            } else {
                                text_to_speak = "";
                                bot_name_is_found = 0;
                                done = true;
                                break;
                            }
                        }
                        if (bot_name_is_found) {
                            text_to_speak = "";
                        }
                    } else {
                        // [ПАТЧ] Без multi-chars — игнорируем любые \nИмя:
                        if (!text_to_speak.empty() && text_to_speak[0] == '\n' &&
                            text_to_speak[text_to_speak.size() - 1] == ':' &&
                            text_to_speak.size() < 10) {
                            text_to_speak = "";
                            done = true;
                            break;
                        }
                    }

                    int text_len = static_cast<int>(text_to_speak.size());
                    if (text_len > 0 &&
                        text_to_speak[text_len - 1] == ',')
                        n_comas++;
                    if (text_to_speak.size() >= 3 &&
                        text_to_speak.substr(text_to_speak.size() - 3, 3) ==
                        "Mr.")
                        text_to_speak[text_len - 1] = ' ';
                    if (new_tokens == 20 &&
                        g_interrupt_reason.load() == InterruptReason::NONE) {
                        audio.clear();
                    }

                    // Умное разбиение TTS по предложениям
                    tts_smart_buffer += text_to_speak;
                    text_to_speak.clear();
                    bool should_send = false;
                    size_t buf_len = tts_smart_buffer.size();
                    if (buf_len >= 1) {
                        char last_char = tts_smart_buffer[buf_len - 1];
                        if (last_char == '.' || last_char == '!' ||
                            last_char == '?' || last_char == ':' ||
                            last_char == '\n') {
                            should_send = true;
                        }
                    }
                    if (!should_send && buf_len >= 2) {
                        char c1 = tts_smart_buffer[buf_len - 2];
                        char c2 = tts_smart_buffer[buf_len - 1];
                        if ((c1 == '.' || c1 == '!' || c1 == '?') &&
                            std::isspace(
                                static_cast<unsigned char>(c2))) {
                            should_send = true;
                        }
                    }
                    if (!should_send && buf_len >= 3) {
                        char c1 = tts_smart_buffer[buf_len - 3];
                        char c2 = tts_smart_buffer[buf_len - 2];
                        char c3 = tts_smart_buffer[buf_len - 1];
                        if ((c1 == '.' || c1 == '!' || c1 == '?') &&
                            (c2 == '"' || c2 == '\'' || c2 == ')' ||
                                c2 == ']' || c2 == '\xC2') &&
                            std::isspace(
                                static_cast<unsigned char>(c3))) {
                            should_send = true;
                        }
                    }

                    if (!should_send && buf_len > 500) {
                        size_t cut_pos = tts_smart_buffer.find_last_of(
                            ".!?", buf_len - 100);
                        if (cut_pos != std::string::npos &&
                            cut_pos > 150) {
                            should_send = true;
                            std::string to_send_part =
                                tts_smart_buffer.substr(0, cut_pos + 1);
                            tts_smart_buffer =
                                tts_smart_buffer.substr(cut_pos + 1);
                            to_send_part = string_replace_all(to_send_part,
                                params.bot_name + ":", "");
                            to_send_part = string_replace_all(to_send_part,
                                params.bot_name + " :", "");
                            to_send_part = string_replace_all(to_send_part,
                                "<|eot_id|>", "");
                            to_send_part = string_replace_all(to_send_part,
                                "<|start_header_id|>", "");
                            to_send_part = string_replace_all(to_send_part,
                                "<|end_header_id|>", "");
                            to_send_part = string_replace_all(to_send_part,
                                "<|im_end|>", "");
                            to_send_part = string_replace_all(to_send_part,
                                "<|im_start|>", "");
                            trim(to_send_part);
                            if (!to_send_part.empty() &&
                                person_name_is_found == 0) {
                                full_response_text += to_send_part;
                                if (!params.translate ||
                                    translation_is_going != 0) {
                                    std::string voice_copy = current_voice;
                                    safe_thread_emplace(threads,
                                        [to_send_part, voice_copy,
                                        params]() {
                                            send_tts_async(to_send_part,
                                                voice_copy, params.language,
                                                params.xtts_url);
                                        });
                                    if (params.sleep_before_xtts) {
                                        int sleep_remaining =
                                            params.sleep_before_xtts;
                                        while (sleep_remaining > 0 &&
                                            g_interrupt_reason.load() ==
                                            InterruptReason::NONE) {
                                            std::this_thread::sleep_for(
                                                std::chrono::milliseconds(20));
                                            sleep_remaining -= 20;
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    if (should_send && !tts_smart_buffer.empty()) {
                        std::string to_send = tts_smart_buffer;
                        tts_smart_buffer.clear();
                        to_send = string_replace_all(to_send,
                            params.bot_name + ":", "");
                        to_send = string_replace_all(to_send,
                            params.bot_name + " :", "");
                        trim(to_send);
                        if (!antiprompts.empty() &&
                            !antiprompts[0].empty()) {
                            std::string user_marker = antiprompts[0];
                            if (to_send.size() >= user_marker.size()) {
                                std::string end_part = to_send.substr(
                                    to_send.size() - user_marker.size());
                                if (end_part == user_marker) {
                                    to_send = to_send.substr(
                                        0, to_send.size() -
                                        user_marker.size());
                                    trim(to_send);
                                }
                            }
                        }
                        to_send = string_replace_all(to_send, "<|eot_id|>", "");
                        to_send = string_replace_all(to_send,
                            "<|start_header_id|>", "");
                        to_send = string_replace_all(to_send,
                            "<|end_header_id|>", "");
                        to_send = string_replace_all(to_send, "<|im_end|>", "");
                        to_send = string_replace_all(to_send, "<|im_start|>", "");
                        to_send = string_replace_all(to_send, "</s>", "");
                        to_send = string_replace_all(to_send, "<|endoftext|>", "");
                        to_send = string_replace_all(to_send, "<|", "");
                        to_send = string_replace_all(to_send, "|>", "");
                        trim(to_send);
                        if (!to_send.empty() &&
                            person_name_is_found == 0) {
                            if (params.translate &&
                                translation_is_going == 0) {
                                n_embd_inp_before_trans =
                                    static_cast<int>(embd_inp.size());
                                std::string trans_prompt =
                                    "\nПеревод последнего предложения "
                                    "на русский.\n" +
                                    bot_name_current_ru + ":" +
                                    translation_full + to_send;
                                std::vector<llama_token> trans_prompt_emb =
                                    ::llama_tokenize(ctx_llama,
                                        trans_prompt, false);
                                embd.insert(embd.end(),
                                    trans_prompt_emb.begin(),
                                    trans_prompt_emb.end());
                                translation_is_going = 1;
                                translation_full = to_send;
                                text_to_speak = "";
                                continue;
                            }
                            full_response_text += to_send;
                            std::string voice_copy = current_voice;
                            safe_thread_emplace(threads,
                                [to_send, voice_copy, params]() {
                                    send_tts_async(to_send, voice_copy,
                                        params.language, params.xtts_url);
                                });
                            if (params.sleep_before_xtts) {
                                int sleep_remaining =
                                    params.sleep_before_xtts;
                                while (sleep_remaining > 0 &&
                                    g_interrupt_reason.load() ==
                                    InterruptReason::NONE) {
                                    std::this_thread::sleep_for(
                                        std::chrono::milliseconds(20));
                                    sleep_remaining -= 20;
                                }
                            }
                        }
                    }

                    if (params.translate && translation_is_going == 1) {
                        translation_is_going = 0;
                        if (n_embd_inp_before_trans &&
                            !embd_inp.empty()) {
                            int rollback_num =
                                static_cast<int>(embd_inp.size()) -
                                n_embd_inp_before_trans;
                            if (rollback_num) {
                                embd_inp.erase(
                                    embd_inp.end() - rollback_num,
                                    embd_inp.end());
                                n_past =
                                    static_cast<int>(embd_inp.size());
                                n_session_consumed = n_past;
                                llama_memory_seq_rm(
                                    llama_get_memory(ctx_llama), 0,
                                    static_cast<int>(embd_inp.size()),
                                    -1);
                                printf("\n");
                            }
                        }
                    }
                }
            }
        } // конец while (true) — цикл генерации

        // [21.27] Роллбэк контекста при прерывании
        if (was_interrupted) {
            std::lock_guard<std::mutex> lock(g_llama_mutex);
            int rollback_num = new_tokens;
            if (rollback_num > 0 &&
                rollback_num <= static_cast<int>(embd_inp.size())) {
                embd_inp.erase(embd_inp.end() - rollback_num,
                    embd_inp.end());
                n_past = static_cast<int>(embd_inp.size());
                n_session_consumed = n_past;
                llama_memory_seq_rm(llama_get_memory(ctx_llama), 0,
                    static_cast<int>(embd_inp.size()), -1);
                if (!session_tokens.empty()) {
                    if (session_tokens.size() > embd_inp.size()) {
                        session_tokens.resize(embd_inp.size());
                    }
                    n_session_consumed =
                        static_cast<int>(embd_inp.size());
                }
                if (!past_prev_arr.empty()) {
                    past_prev_arr.pop_back();
                }
                if (params.verbose) {
                    printf("[Rollback] Удалено %d токенов, контекст "
                        "восстановлен до %zu токенов\n",
                        rollback_num, embd_inp.size());
                }
            }
            if (!path_session.empty()) {
                llama_state_save_file(ctx_llama, path_session.c_str(),
                    session_tokens.data(), session_tokens.size());
            }
        }

        // [21.28] Обработка антипромптов
        {
            std::string last_output;
            int total_chars = 0;
            int start_index = static_cast<int>(embd_inp.size()) - 1;
            for (int i = start_index; i >= 0 && total_chars < 100; i--) {
                std::string piece =
                    llama_token_to_piece(ctx_llama, embd_inp[i]);
                total_chars += utf8_length(piece);
                last_output = piece + last_output;
            }
            if (!text_to_speak.empty()) {
                last_output += text_to_speak;
            }
            int i_antiprompt = 0;
            last_output_has_username = false;
            last_output_has_EOT = false;
            bool antiprompt_matched = false;
            for (std::string& antiprompt : antiprompts) {
                if (params.multi_chars && last_output.size() >= 4) {
                    last_output = string_replace_all(last_output, " ???", "");
                    last_output = string_replace_all(last_output, " ??", "");
                    last_output = string_replace_all(last_output, " ?", "");
                    last_output = string_replace_all(last_output, " !!!", "");
                    last_output = string_replace_all(last_output, " !!", "");
                    last_output = string_replace_all(last_output, " !", "");
                    last_output = string_replace_all(last_output, "!!!", "");
                    last_output = string_replace_all(last_output, "!!", "");
                    last_output = string_replace_all(last_output, " ...", "");
                    last_output = string_replace_all(last_output, " .", "");
                    last_output = string_replace_all(last_output, " ,", "");
                    last_output = string_replace_all(last_output, "...", "");
                    last_output = string_replace_all(last_output, "(", "");
                    last_output = string_replace_all(last_output, ")", "");
                    static const std::regex re_name_detect(
                        "\n([^:]*):",
                        std::regex::icase | std::regex::optimize);
                    std::smatch matches;
                    if (std::regex_search(last_output, matches,
                        re_name_detect) &&
                        !matches.empty() && matches.size() >= 2 &&
                        !matches[1].str().empty() &&
                        matches[1].str() != params.person &&
                        matches[1].str() != " \n" + params.person) {
                        std::string current_voice_tmp = matches[1].str();
                        current_voice_tmp =
                            string_replace_all(current_voice_tmp, ":", "");
                        current_voice_tmp =
                            string_replace_all(current_voice_tmp, "\"", "");
                        trim(current_voice_tmp);
                        if (current_voice_tmp.size() > 1 &&
                            current_voice_tmp.size() < 30) {
                            current_voice = current_voice_tmp;
                            thread_local std::string last_voice_for_regex;
                            thread_local std::regex regEx_voice;
                            if (last_voice_for_regex != current_voice) {
                                regEx_voice = std::regex(
                                    "\n" + current_voice + ":");
                                last_voice_for_regex = current_voice;
                            }
                            text_to_speak = std::regex_replace(
                                text_to_speak, regEx_voice, "\n");
                        }
                    }
                }
                if (last_output.length() >= antiprompt.length()) {
                    std::string last_output_trimmed = last_output;
                    rtrim(last_output_trimmed);
                    std::string end_of_output =
                        last_output_trimmed.substr(
                            last_output_trimmed.length() -
                            antiprompt.length());
                    bool is_at_end =
                        (last_output_trimmed.length() >=
                            antiprompt.length() &&
                            last_output_trimmed.substr(
                                last_output_trimmed.length() -
                                antiprompt.length()) == antiprompt);
                    if (is_at_end || end_of_output == antiprompt) {
                        if (params.min_tokens > 0 &&
                            tokens_in_reply < params.min_tokens) {
                            if (params.verbose) {
                                printf(" [ignoring antiprompt '%s', "
                                    "too short (%d < %d)] ",
                                    antiprompt.c_str(),
                                    tokens_in_reply,
                                    params.min_tokens);
                            }
                            i_antiprompt++;
                            continue;
                        }
                        bool is_user_name_antiprompt =
                            (antiprompt == "\n" + params.person + chat_symb ||
                                antiprompt == "\n" + params.person + " " +
                                chat_symb);
                        bool is_eot_antiprompt =
                            (antiprompt ==
                                params.instruct_preset_data["bot_message_suffix"] ||
                                antiprompt == "<|eot_id|>" ||
                                antiprompt == "</s>" ||
                                antiprompt == "<end_of_turn|>" ||
                                antiprompt == "<|im_end|>" ||
                                antiprompt == "</end_of_turn>");
                        if (is_user_name_antiprompt) {
                            if (text_to_speak.empty() ||
                                text_to_speak.length() < 2) {
                                i_antiprompt++;
                                continue;
                            }
                            size_t pos_in_speak =
                                text_to_speak.find(antiprompt);
                            if (pos_in_speak != std::string::npos) {
                                text_to_speak = text_to_speak.substr(
                                    0, pos_in_speak);
                                trim(text_to_speak);
                            }
                            antiprompt_matched = true;
                            last_output_has_username = true;
                            done = true;
                        }
                        else if (is_eot_antiprompt) {
                            size_t pos_in_speak =
                                text_to_speak.find(antiprompt);
                            if (pos_in_speak != std::string::npos) {
                                text_to_speak = text_to_speak.substr(
                                    0, pos_in_speak);
                                trim(text_to_speak);
                            }
                            antiprompt_matched = true;
                            last_output_has_EOT = true;
                            done = true;
                        }
                        else {
                            size_t pos_in_speak =
                                text_to_speak.find(antiprompt);
                            if (pos_in_speak != std::string::npos) {
                                text_to_speak = text_to_speak.substr(
                                    0, pos_in_speak);
                                trim(text_to_speak);
                            }
                            antiprompt_matched = true;
                            done = true;
                        }
                        if (done) {
                            text_to_speak =
                                string_replace_all(text_to_speak, antiprompt, "");
                            fflush(stdout);
                            need_to_save_session = true;
                            if (i_antiprompt == 0) {
                                last_output_has_username = true;
                                printf(" ");
                            }
                            static int short_response_attempts = 0;
                            if (params.min_tokens &&
                                tokens_in_reply < params.min_tokens) {
                                short_response_attempts++;
                                if (short_response_attempts > 5) {
                                    short_response_attempts = 0;
                                    break;
                                }
                                int symbols_to_delete =
                                    static_cast<int>(
                                        utf8_length(antiprompt) * 1) + 1;
                                const std::vector<llama_token>
                                    tokens_to_del = llama_tokenize(
                                        ctx_llama, antiprompt.c_str(),
                                        false);
                                int rollback_num =
                                    static_cast<int>(tokens_to_del.size())
                                    + 1;
                                if (rollback_num) {
                                    embd_inp.erase(
                                        embd_inp.end() - rollback_num,
                                        embd_inp.end());
                                    n_past =
                                        static_cast<int>(embd_inp.size());
                                    n_session_consumed = n_past;
                                    llama_memory_seq_rm(
                                        llama_get_memory(ctx_llama), 0,
                                        static_cast<int>(embd_inp.size()),
                                        -1);
                                    if (symbols_to_delete >
                                        utf8_length(text_to_speak))
                                        text_to_speak = "";
                                    else
                                        text_to_speak = utf8_substr(
                                            text_to_speak, 0,
                                            utf8_length(text_to_speak) -
                                            symbols_to_delete);
                                    temp_next = 1.8f;
                                    fflush(stdout);
                                    printf("\b\b\b\b\b\b\b\b\b\b\b\b");
                                    fflush(stdout);
                                    done = false;
                                }
                            }
                            else {
                                short_response_attempts = 0;
                                break;
                            }
                        }
                    }
                }
                i_antiprompt++;
            }
            if (antiprompt_matched && params.min_tokens > 0 &&
                tokens_in_reply < params.min_tokens) {
                done = false;
            }
        }

        // [21.29] Финальная обработка оставшегося текста
        if (!tts_smart_buffer.empty()) {
            text_to_speak = tts_smart_buffer + text_to_speak;
            tts_smart_buffer.clear();
        }
        InterruptReason local_reason = g_interrupt_reason.load();
        bool was_interrupted_final =
            (local_reason != InterruptReason::NONE ||
                g_interrupt_processed.load());
        if (was_interrupted_final) {
            text_to_speak = "";
            if (local_reason != InterruptReason::VAD_SPEECH) {
                audio.clear();
            }
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_bot_state.store(BotState::IDLE);
        }
        text_to_speak = string_replace_all(text_to_speak, "<|eot_id|>", "");
        text_to_speak = string_replace_all(text_to_speak,
            "<|start_header_id|>", "");
        text_to_speak = string_replace_all(text_to_speak,
            "<|end_header_id|>", "");
        text_to_speak = string_replace_all(text_to_speak, "<|im_end|>", "");
        text_to_speak = string_replace_all(text_to_speak, "<|im_start|>", "");
        text_to_speak = string_replace_all(text_to_speak, "</s>", "");
        text_to_speak = string_replace_all(text_to_speak, "<|endoftext|>", "");
        text_to_speak = string_replace_all(text_to_speak, "<|", "");
        text_to_speak = string_replace_all(text_to_speak, "|>", "");
        trim(text_to_speak);
        if (!text_to_speak.empty()) {
            full_response_text += text_to_speak;
            std::string clean_full = full_response_text;
            try {
                static const std::regex re_emotion_parens_final(R"(\([^)]{2,60}\))", std::regex::ECMAScript);
                static const std::regex re_emotion_stars_final(R"(\*[^*]{2,60}\*)", std::regex::ECMAScript);
                clean_full = std::regex_replace(clean_full, re_emotion_parens_final, "");
                clean_full = std::regex_replace(clean_full, re_emotion_stars_final, "");
                clean_full = std::regex_replace(clean_full, std::regex(R"(<\|[^>]*\|>)"), "");
                clean_full = std::regex_replace(clean_full, std::regex(R"(</?[a-zA-Z_][a-zA-Z0-9_]*>)"), "");
                clean_full = std::regex_replace(clean_full, std::regex(R"(\b(assistant|system|user|end_header_id|eot_id|start_header_id|eo)\b)"), "");
            }
            catch (const std::regex_error&) {
                clean_full = string_replace_all(clean_full, "<|eot_id|>", "");
                clean_full = string_replace_all(clean_full, "<|im_end|>", "");
                clean_full = string_replace_all(clean_full, "<|im_start|>", "");
            }
            trim(clean_full);
            {
                std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                g_last_tts_text = clean_full;
            }
            std::string final_text = text_to_speak;
            std::string voice_copy = current_voice;
            try {
                safe_thread_emplace(threads,
                    [final_text, voice_copy, params]() {
                        send_tts_async(final_text, voice_copy,
                            params.language, params.xtts_url);
                    });
                if (params.sleep_before_xtts) {
                    int sleep_remaining = params.sleep_before_xtts;
                    while (sleep_remaining > 0 &&
                        g_interrupt_reason.load() ==
                        InterruptReason::NONE) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(20));
                        sleep_remaining -= 20;
                    }
                }
            }
            catch (const std::exception& ex) {
                if (params.verbose) {
                    std::cerr << "[Exception] Final TTS flush failed: "
                        << ex.what() << '\n';
                }
            }
        }
        else if (!full_response_text.empty()) {
            std::lock_guard<std::mutex> lock(g_last_tts_mutex);
            if (g_last_tts_text.empty()) {
                g_last_tts_text = full_response_text;
            }
        }

        g_ui_state.store(UiState::UI_IDLE);

        // [21.30] Очистка потоков TTS
        if (was_interrupted_final) {
            std::lock_guard<std::mutex> lock(g_threads_mutex);
            for (auto& t_local : threads) {
                if (t_local.joinable()) t_local.detach();
            }
            threads.clear();
        }
        else {
            std::vector<std::thread> temp_threads;
            temp_threads.swap(threads);
            for (auto& t_local : temp_threads) {
                if (t_local.joinable()) {
                    try { t_local.join(); }
                    catch (...) {}
                }
            }
        }
        if (!was_interrupted_final) {
            audio.clear();
        }
        static int response_counter = 0;
        response_counter++;
        if (response_counter >= THREAD_CLEANUP_INTERVAL) {
            cleanup_finished_threads(threads);
            response_counter = 0;
        }

        llama_end_time = get_current_time_ms();
        if (params.verbose) {
            llama_time_input =
                llama_start_generation_time - llama_start_time;
            llama_time_output =
                llama_end_time - llama_start_generation_time;
            llama_time_total = llama_end_time - llama_start_time;
            printf("\n\n[Context: %d/%d. Tokens: %d in + %d out. "
                "Input %.3f s + output %.3f s = total: %.3f s]",
                n_past, n_ctx, input_tokens_count, new_tokens,
                llama_time_input, llama_time_output,
                llama_time_total);
            float input_speed = (llama_time_input > 0.001f)
                ? static_cast<float>(input_tokens_count) /
                llama_time_input : 0.0f;
            float output_speed = (llama_time_output > 0.001f)
                ? static_cast<float>(new_tokens) /
                llama_time_output : 0.0f;
            float total_speed = (llama_time_total > 0.001f)
                ? static_cast<float>(new_tokens) /
                llama_time_total : 0.0f;
            printf("\n[Speed: input %.2f t/s + output %.2f t/s = "
                "total: %.2f t/s]\n",
                input_speed, output_speed, total_speed);
        }

        g_interrupt_reason.store(InterruptReason::NONE);
        g_interrupt_processed.store(false);
        g_bot_state.store(BotState::IDLE);
        llama_interrupted_time = 0.0;
        llama_start_generation_time = 0.0;
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            g_hotkey_pressed = "";
        }
        printf("\n");
        fflush(stdout);
    } // конец while (is_running) — основной цикл

    // ========================================================================
    // [21.31] ЗАВЕРШЕНИЕ РАБОТЫ
    // ========================================================================

    if (params.verbose) {
        printf("Cleaning up TTS threads...\n");
    }
    g_shutting_down.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (audio_thread.joinable()) {
        for (int i = 0; i < 20 && g_audio_thread_running.load(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        audio_thread.join();
        if (params.verbose) {
            printf("[AudioInput] Thread stopped\n");
        }
    }

    std::vector<std::thread> local_threads;
    {
        std::lock_guard<std::mutex> lock(g_threads_mutex);
        local_threads.swap(threads);
    }
    if (params.verbose) {
        printf("Waiting for %zu TTS threads to finish...\n",
            local_threads.size());
    }
    for (auto& t_local : local_threads) {
        if (t_local.joinable()) {
            try {
                t_local.join();
            }
            catch (const std::exception& e) {
                if (params.verbose) {
                    fprintf(stderr, "Warning: Exception joining thread: "
                        "%s\n", e.what());
                }
                t_local.detach();
            }
            catch (...) {
                if (params.verbose) {
                    fprintf(stderr, "Warning: Unknown exception joining "
                        "thread\n");
                }
                t_local.detach();
            }
        }
    }
    if (params.verbose) {
        printf("Cleanup complete.\n");
    }

    audio.pause();
    audio.clear();
    keyboard_input_running.store(false);
    if (input_thread.joinable()) {
        input_thread.join();
    }
    g_shortcut_thread_running.store(false);
    if (shortcut_thread.joinable()) {
        shortcut_thread.join();
    }
    audio.pause();

    whisper_print_timings(ctx_wsp);
    if (ctx_llama) {
        llama_perf_context_print(ctx_llama);
    }
    whisper_free(ctx_wsp);
    if (smpl) {
        llama_perf_sampler_print(smpl);
    }
    if (smpl_high_temp) {
        llama_sampler_free(smpl_high_temp);
    }
    llama_batch_free(batch);
    llama_free(ctx_llama);
    llama_model_free(model_llama);
    llama_backend_free();
    return 0;
}

// ========================================================================
// 22. ТОЧКИ ВХОДА
// ========================================================================

// [22.1] Точка входа для Windows (wmain)
#if _WIN32
int wmain(int argc, const wchar_t** argv_UTF16LE) {
    console::init(true, true);
    atexit([] {
        console::cleanup();
        });

    std::vector<std::string> buffer(argc);
    std::vector<char*> argv_UTF8(argc);
    for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = &buffer[i][0];
    }
    return run(argc, argv_UTF8.data());
}

// [22.2] Точка входа для Linux/POSIX (main)
#else
int main(int argc, const char** argv_UTF8) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl" << std::endl;
        return 1;
    }
    const char* verbose_env = std::getenv("TALK_LLAMA_VERBOSE");
    if (verbose_env && (std::string(verbose_env) == "1" ||
        std::string(verbose_env) == "true")) {
        g_verbose_mode.store(true);
    }
    console::init(true, true);
    atexit([] {
        console::cleanup();
        curl_global_cleanup();
        });
    return run(argc, argv_UTF8);
}
#endif

// ========================================================================
// КОНЕЦ ФАЙЛА — ВЕРСИЯ 6.1
// ========================================================================