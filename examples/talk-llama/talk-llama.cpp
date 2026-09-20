// ============================================================================
// talk-llama.cpp — голосовой ассистент на базе Whisper, LLaMA и XTTS.
//
// Версия: 34 (super upgrade после v33).
//
// Назначение:
//   Слушает микрофон, распознаёт речь через Whisper, генерирует ответ
//   через LLaMA и озвучивает его через XTTS. Пользователь может
//   перебивать бота голосом, отдавать голосовые команды (стоп, сброс,
//   повтори, время, дата, google, call leo) и работать через
//   клавиатурные горячие клавиши.
//
// Архитектура:
//   - Основной поток (run) — обрабатывает ввод, генерирует ответы LLaMA,
//     управляет состоянием бота.
//   - audio_input_thread_func — читает микрофон, детектирует речь
//     энергетическим VAD, транскрибирует Whisper, отправляет текст
//     в основной поток.
//   - input_thread_func — читает строки с клавиатуры.
//   - keyboard_shortcut_func — опрашивает горячие клавиши (Windows).
//   - tts_worker_func — читает очередь TtsRequest и отправляет текст
//     в XTTS-сервер по HTTP.
//
// Ключевой принцип работы с LLaMA:
//   Промпт собирается пофрагментно. Шаблонные фрагменты
//   (<|start_header_id|>, <|eot_id|> и т.д.) токенизируются с
//   parse_special = true — они превращаются в настоящие control-токены.
//   Пользовательский текст токенизируется с parse_special = false —
//   он не может внедрить служебный токен. Благодаря этому модель
//   видит правильную структуру диалога и генерирует правильный
//   EOG-токен (128009), а не искажённый "<|eo_id|>".
//
// ИСТОРИЯ РЕФАКТОРИНГА (версия 33, super upgrade после v32):
//   - ГЛАВНЫЙ ФИКС: починка K-shift (сдвиг контекста).
//     * Отказ от q8_0 KV-кэша. Работаем с дефолтным f16.
//       На CUDA K-shift на q8_0 ненадёжен (dequant → RoPE → quant),
//       что и приводило к сбою генерации и потере ответа.
//     * n_keep зафиксирован на 512 (было ctx_size/16 → 512, потом
//       уточнялось до prompt_size+128 → 869). Формула n_keep = 512
//       сохраняет личность бота (первые 512 токенов системного
//       промпта) и оставляет ~7680 токенов под диалог при ctx=8192.
//     * n_discard = n_left / 4 (было max(256, n_left/3) в патче D).
//       Меньшие блоки сдвига — меньше нагрузка на KV-кэш.
//     * Добавлен клампинг n_discard в [0, n_left - 1] — защита
//       от underflow и от полного удаления контекста.
//     * Добавлена проверка llama_memory_can_shift() перед сдвигом.
//       Если сдвиг невозможен — fallback на обрезку до n_keep.
//     * После сдвига embd НЕ очищается. В него кладутся последние
//       n_prev = 64 токенов из embd_inp. Это устраняет холостое
//       вращение цикла (раньше embd.clear() + continue → цикл
//       крутился до n_predict, не генерируя токенов).
//     * Убран continue после сдвига — цикл продолжает работу с
//       непустым embd.
//     * Проверка if (embd.empty()) заменена на выход из цикла
//       (защита от пустых итераций).
//     * text_to_speak НЕ очищается при сдвиге — уже сгенерированный
//       ответ пойдёт в TTS.
//   - Патч F (автонастройка) сохранён, но n_keep больше не
//     вычисляется от ctx_size. Остальные параметры (n_predict,
//     batch_size, min_tokens) — как в v32.
//   - Патчи C/D/E/B сохранены без изменений.
//   - Qwen-детекция и reset_kv_cache_with_history() — удалены
//     ещё в v32.
//   - Семафор XTTS возвращён в системную временную директорию
//     (%TEMP% на Windows, TMPDIR на POSIX).
//
// ИСТОРИЯ РЕФАКТОРИНГА (версия 34):
//   - PATCH 3: починен race condition в tts_worker_func.
//     Условие выхода из цикла теперь проверяется под мьютексом,
//     pop_front защищён от гонки с clear_tts_queue.
//   - PATCH 4: подключён safe_remove_fragment для стоп-строк.
//     Он удаляет подстроку только на границах слов, чтобы
//     буквенные стоп-слова (USER, ASSISTANT, END) не вырезались
//     из середины русского текста.
//   - PATCH 5: g_hallucination_count переименован в
//     g_transcription_count — счётчик считает все транскрипции,
//     а не только галлюцинации.
//   - PATCH 7: в call_default добавлены падежи мужских имён.
//     "Позови Иван" теперь распознаётся (Иван → Ивана).
//   - PATCH 8: удаление Gemma-тегов (<start_of_turn>,
//     <end_of_turn> и др.) в strip_special_tokens и
//     sanitize_for_console. Они не подпадают под regex <|...|>.
//   - PATCH 10: разогрев Whisper в момент старта речи.
//     Первый вызов transcribe() в сессии медленнее последующих —
//     прогрев прячет эту задержку.
//   - PATCH 11: антипромпт "</end_of_turn>" для Gemma 2/3.
//     Модель иногда генерирует закрывающий слэш вместо
//     правильного <end_of_turn>.
//   - PATCH 12: антипромпты "Имя:" и "Имя :" — защита от
//     продолжения диалога за пользователя.
//   - PATCH 15: ltrim/rtrim/trim/trim_spaces_only обёрнуты
//     в анонимный namespace. В common.h есть своя trim()
//     с другой сигнатурой — изоляция убирает возможность
//     неоднозначной перегрузки.
//   - PATCH 16: расширен список TTS-интро (27 вариантов).
//
// ПАТЧ A (Whisper endpointing) В ЭТОЙ ВЕРСИИ НЕ ПРИМЕНЁН.
//   Whisper работает как в v32 — пользователь доволен.
// ============================================================================


// ============================================================================
// 1. ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
// ============================================================================

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "llama.h"
#include "llama-chat.h"

// --- 1.1. Системные C++ ---
#include <chrono>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <regex>
#include <sstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <locale>
#include <clocale>
#include <codecvt>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <ctype.h>
#include <map>
#include <iterator>
#include <ctime>
#include <filesystem>
#include <random>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <cstring>

// --- 1.2. Пользовательские модули ---
// console.cpp подключаем как .cpp — так задумано в оригинальном примере
// whisper.cpp. Это даёт доступ к реализации readline() без отдельной
// библиотеки.
#include "console.h"
#include "console.cpp"

// --- 1.3. Сеть (HTTP для XTTS и google-команды) ---
#include <curl/curl.h>
#include "json.hpp"

// --- 1.4. Платформенные заголовки ---
// На Windows нужны WinAPI-функции (CreateFileA, GetTempPath, и т.д.),
// на POSIX — open/write/fsync для файла-семафора XTTS.
#ifdef _WIN32
#include <Windows.h>
#include <fileapi.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif


// ============================================================================
// 2. ЦВЕТОВАЯ СХЕМА КОНСОЛИ
// ============================================================================
// ANSI-коды для цветного вывода. Работают на Windows 10+ с включённым
// VT-processing (см. console::init) и на всех POSIX-терминалах.

#define C_USER          "\033[32m"   // зелёный — реплики пользователя
#define C_BOT           "\033[1;33m" // жёлтый жирный — реплики бота
#define C_RESET         "\033[0m"
#define C_STATUS_TTS    "\033[90m"   // серый — статус озвучки
#define C_STATUS_CMD    "\033[36m"   // голубой — статус команды
#define C_STATUS_STOP   "\033[31m"   // красный — статус остановки
#define C_STATUS_SPEECH "\033[33m"   // жёлтый — статус речи

// Псевдонимы для часто используемых статусов.
#define C_TTS     C_STATUS_TTS
#define C_CMD     C_STATUS_CMD
#define C_STOP    C_STATUS_STOP


// ============================================================================
// 3. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И СИНХРОНИЗАЦИЯ
// ============================================================================
// Здесь собраны все глобалы проекта. Они нужны, потому что фоновые
// потоки (аудио, TTS, клавиатура) должны обмениваться состоянием с
// основным циклом. Для каждого общего ресурса — свой мьютекс или
// атомарный флаг.

// --- 3.1. Состояние бота ---
// IDLE         — бот ждёт ввода, слушает микрофон.
// GENERATING   — бот генерирует ответ LLaMA.
// INTERRUPTED  — генерация прервана пользователем (barge-in/хоткей).
enum class BotState : uint8_t {
    IDLE = 0,
    GENERATING,
    INTERRUPTED
};
std::atomic<BotState> g_bot_state{ BotState::IDLE };

// --- 3.2. Причина прерывания ---
// Нужна, чтобы основной цикл знал, как реагировать: VAD_SPEECH — не
// очищать микрофон (пользователь уже говорит), MANUAL_STOP — очистить.
enum class InterruptReason : uint8_t {
    NONE = 0,
    VAD_SPEECH,      // пользователь заговорил (barge-in)
    HOTKEY_STOP,     // нажата горячая клавиша (Ctrl+Space и т.п.)
    HOTKEY_ALT,      // резерв (не используется)
    MANUAL_STOP      // команда «стоп» вручную
};
std::atomic<InterruptReason> g_interrupt_reason{ InterruptReason::NONE };
std::atomic<bool> g_interrupt_processed{ false };
std::atomic<bool> g_shutting_down{ false };
std::atomic<bool> g_cancel_tts_requests{ false };

// --- 3.3. Атомарный указатель на ctx_llama ---
// ctx_llama освобождается и создаётся заново при команде «сброс».
// Аудио-поток не должен держать «протухший» указатель, поэтому он
// читает его через этот атомарный указатель под g_llama_mutex.
std::atomic<llama_context*> g_ctx_llama_atomic{ nullptr };

// --- 3.4. Файл-семафор XTTS ---
// XTTS-сервер читает файл xtts_play_allowed.txt: "1" — играть,
// "0" — пауза. Это позволяет мгновенно останавливать озвучку без
// разрыва HTTP-соединения.
std::string g_xtts_control_file_path = "";
std::mutex g_xtts_control_mutex;
bool g_last_semaphore_value = true;
bool g_last_semaphore_initialized = false;

// --- 3.5. Ввод с клавиатуры ---
std::queue<std::string> input_queue;
std::mutex input_mutex;
std::atomic<bool> keyboard_input_running{ true };

// --- 3.6. Горячие клавиши ---
// Строка вида "Ctrl+Space", "Ctrl+Right" и т.д. Основной цикл
// читает её и превращает в команду.
std::string g_hotkey_pressed = "";
std::mutex g_hotkey_pressed_mutex;
std::atomic<bool> g_shortcut_thread_running{ true };

// --- 3.7. Мьютекс для доступа к ctx_llama ---
// Любая операция с ctx_llama из фонового потока (tokenize, decode)
// должна брать этот мьютекс. Это защищает от гонок с reset.
std::mutex g_llama_mutex;

// --- 3.8. Последний текст, отправленный в TTS ---
// Нужно для команды «повтори».
std::string g_last_tts_text = "";
std::mutex g_last_tts_mutex;

// --- 3.9. Verbose-режим ---
std::atomic<bool> g_verbose_mode{ false };

// --- 3.10. Мьютекс консольного вывода ---
// Рекурсивный, потому что функции печати могут вызывать друг друга
// (например, print_replica может вызвать begin_new_pair).
std::recursive_mutex g_console_mutex;

// --- 3.11. Буфер накопленного текста от Whisper ---
// Whisper может выдать несколько сегментов подряд. Мы копим их здесь,
// а потом отправляем одной строкой в LLaMA.
std::string g_accumulated_text;
std::mutex g_text_accumulator_mutex;

// --- 3.12. Буфер отображаемого текста ---
// Для UI (пока не используется, но оставлен для будущего).
std::string g_display_text;
std::mutex g_display_mutex;

// --- 3.13. Мягкий лимит токенов ---
// Если накопленный текст превысил этот лимит, принудительно
// отправляем его в LLaMA, не дожидаясь тишины.
std::atomic<int>  g_soft_limit_tokens{ 2560 };
std::atomic<bool> g_audio_thread_running{ false };

// --- 3.14. Ожидающий запрос в LLaMA от аудио-потока ---
std::atomic<bool> g_pending_llm_request{ false };
std::string g_pending_llm_text;
std::mutex g_pending_llm_mutex;

// --- 3.15. Флаг принудительного сброса VAD ---
// Ставится при команде «стоп», чтобы аудио-поток сбросил своё
// локальное состояние (speech_active, accumulated_text).
std::atomic<bool> g_force_vad_reset{ false };

// --- 3.16. Флаг сброса контекста ---
// Пока true, аудио-поток ждёт на g_reset_cv. Основной поток в это
// время пересоздаёт ctx_llama.
std::atomic<bool> g_reset_in_progress{ false };
std::mutex g_reset_mutex;
std::condition_variable g_reset_cv;

// --- 3.17. Whisper-промпт ---
// initial_prompt для транскрибации. Помогает Whisper точнее
// распознавать имена собственные.
std::string g_whisper_prompt = "";

// --- 3.18. Флаг «нужна пустая строка перед следующей парой» ---
static bool g_need_blank_before_next = false;

// --- 3.19. Флаг «стартовое приглашение уже напечатано» ---
// Чтобы не печатать "Друг: " дважды, когда пользователь уже начал
// говорить (приглашение висит в консоли).
static bool g_initial_prompt_printed = false;

// --- 3.20. Сохранённое состояние генерации для auto-continue ---
// Если пользователь перебил бота, но не сказал ничего нового, через
// continue_max_ms бот может продолжить с того же места.
std::string g_saved_generation_text;
int g_saved_n_past = 0;
std::vector<llama_token> g_saved_embd_inp;
float g_saved_timestamp = 0.0f;
bool g_saved_was_interrupted = false;
std::mutex g_saved_generation_mutex;
std::atomic<bool> g_pending_continue_request{ false };

// --- 3.21. Константы ---
const std::string GOOGLE_VOICE = "Google";
const std::string LEO_NAME = "Лео";
const std::string LEO_VOICE = "Лео";
const std::string DEFAULT_CHAT_SYMB = ": ";

// --- 3.22. Структура TTS-запроса ---
// Всё, что нужно TTS-воркеру для отправки запроса в XTTS.
struct TtsRequest {
    std::string text;
    std::string voice;
    std::string language;
    std::string url;
    std::string stop_seq;
    std::string bot_sfx;
    std::string user_sfx;
    std::string bot_pfx;
    std::string user_pfx;
    std::string chat_symb;
    std::string person;
    std::string bot;
};
std::deque<TtsRequest> g_tts_queue;
std::mutex g_tts_queue_mutex;
std::condition_variable g_tts_queue_cv;
std::atomic<bool> g_tts_worker_running{ true };

// --- 3.23. Логирование в файл ---
std::ofstream g_log_file;
std::mutex g_log_mutex;
std::atomic<bool> g_log_enabled{ false };

// --- 3.24. Счётчик транскрипций ---
// Для мониторинга. Простой atomic, не блокирует.
//
// PATCH 5 (v34): переименован из g_hallucination_count.
// Имя вводило в заблуждение — счётчик инкрементируется при
// КАЖДОЙ успешной транскрипции (в note_transcription), а не
// только при галлюцинации. При сбросе контекста обнуляется.
std::atomic<int> g_transcription_count{0};

// --- 3.25. Флаг «идёт обработка команды» ---
// Пока true, auto-continue не срабатывает.
std::atomic<bool> g_command_in_progress{ false };

// --- 3.26. Заголовок консольного окна ---
static std::string g_last_console_title = "";
static std::mutex g_console_title_mutex;


// ============================================================================
// 4. ПРЕДВАРИТЕЛЬНЫЕ ОБЪЯВЛЕНИЯ
// ============================================================================
// Функции, которые используются до своего определения.

std::string getTempDir();
int utf8_length(const std::string& str);
std::string utf8_substr(const std::string& str, unsigned int start, unsigned int leng);
std::string normalize_template_token(const std::string& s);
std::string normalize_json_key(const std::string& key);

static void allow_xtts_file(std::string& path, int xtts_play_allowed);
static void print_replica(const std::string& color, const std::string& name,
                          const std::string& text, const std::string& status = "");
static void print_bot_prefix(const std::string& bot_name);
static void print_status(const std::string& status, const std::string& color);

// --- 4.1. Хелпер: получить актуальный ctx_llama ---
// Возвращает указатель под мьютексом. Если идёт reset — вернёт
// актуальный (возможно, nullptr). Аудио-поток использует этот хелпер,
// чтобы не держать «протухший» указатель.
static llama_context* get_llama_ctx() {
    std::lock_guard<std::mutex> lock(g_llama_mutex);
    return g_ctx_llama_atomic.load();
}


// ============================================================================
// 5. СТРОКОВЫЕ УТИЛИТЫ
// ============================================================================
// Все функции раздела — UTF-8-aware. Не используют std::isspace
// на байтах > 0x7F (иначе кириллица ломается).

// --- 5.1. Обрезка пробелов ---
// ltrim, rtrim, trim — стандартные операции. Учитывают NBSP (0xA0),
// потому что Whisper иногда вставляет неразрывные пробелы.
//
// PATCH 15 (v34): все четыре функции обёрнуты в анонимный namespace.
//
// WHY: в common.h объявлена функция std::string trim(const std::string&).
// Наша trim(std::string&) — это ДРУГАЯ перегрузка (принимает ссылку
// и меняет на месте). Без изоляции обе перегрузки видны во всех
// translation units, и вызов trim(x) с не-const аргументом может
// дать ошибку «ambiguous call to overloaded function», если где-то
// в проекте есть ещё одна trim() с подходящей сигнатурой.
//
// Анонимный namespace даёт внутреннюю линковку — наши функции
// видны ТОЛЬКО в этом .cpp-файле. Конфликт с common.cpp::trim
// становится физически невозможным.
//
// Все вызовы trim() внутри talk-llama.cpp продолжают работать:
// функции в том же translation unit, что и весь остальной код.
namespace {

inline void ltrim(std::string& s) {
    if (s.empty()) return;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch) {
            return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
                && ch != '\f' && ch != '\v' && ch != 0xA0;
        }));
}

inline void rtrim(std::string& s) {
    if (s.empty()) return;
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) {
            return ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r'
                && ch != '\f' && ch != '\v' && ch != 0xA0;
        }).base(), s.end());
}

inline void trim(std::string& s) {
    if (s.empty()) return;
    rtrim(s);
    ltrim(s);
}

// Обрезка только пробелов и табов (без \n). Используется там, где
// переносы строк значимы.
inline void trim_spaces_only(std::string& s) {
    if (s.empty()) return;
    size_t end = s.size();
    while (end > 0) {
        unsigned char c = static_cast<unsigned char>(s[end - 1]);
        if (c == ' ' || c == '\t' || c == 0xA0) --end;
        else break;
    }
    s.resize(end);
    size_t start = 0;
    while (start < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[start]);
        if (c == ' ' || c == '\t' || c == 0xA0) ++start;
        else break;
    }
    if (start > 0) s.erase(0, start);
}

}  // namespace

// --- 5.2. Пунктуация ---
// Используется при парсинге голосовых команд («погугли погода в лондоне!»
// → «погода в лондоне»).
bool IsPunctuationMark(char c) {
    switch (static_cast<unsigned char>(c)) {
        case ',': case '.': case '?': case '!':
        case ';': case ':': case '(': case ')':
        case '[': case ']': case '{': case '}':
        case '"': case '\'': case '`': case '~':
            return true;
        default: return false;
    }
}

std::string StripPunctuationMarks(const std::string& text) {
    std::string cleanText;
    cleanText.reserve(text.size());
    for (const auto& c : text) {
        if (!IsPunctuationMark(c)) cleanText += c;
    }
    return cleanText;
}

// --- 5.3. Приведение к нижнему регистру (UTF-8-aware) ---
// std::tolower на байтах > 0x7F ломает кириллицу. Обрабатываем
// только ASCII и русский диапазон А-Я (0xD0 0x90-0xAF).
std::string LowerCase(const std::string & text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            ++i;
        }
        else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            // А-Я → а-я
            if (c == 0xD0 && c2 >= 0x90 && c2 <= 0xAF) {
                result += static_cast<char>(0xD0);
                result += static_cast<char>(c2 + 0x20);
            }
            // Ё → ё
            else if (c == 0xD0 && c2 == 0x81) {
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

// --- 5.4. Замена всех вхождений подстроки ---
// Защита от бесконечного цикла: если to содержит from — прерываем.
std::string string_replace_all(std::string str, const std::string& from,
    const std::string& to) {
    if (from.empty()) return str;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
        if (to.find(from) != std::string::npos) break;
    }
    return str;
}

// --- 5.5. Удаление завершающих символов (UTF-8) ---
// targetCharacters разбивается на отдельные UTF-8 символы (с кэшем),
// и last_char сравнивается с ними целиком. Это позволяет корректно
// работать с многобайтными target (например, "»" = 2 байта).
std::string RemoveTrailingCharactersUtf8(const std::string& inputString,
    const std::string& targetCharacters) {
    if (inputString.empty() || targetCharacters.empty()) return inputString;

    // Кэш разбиения targetCharacters на UTF-8 символы.
    static std::mutex cache_mutex;
    static std::map<std::string, std::vector<std::string>> cache;

    std::vector<std::string> targets;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(targetCharacters);
        if (it == cache.end()) {
            std::vector<std::string> chars;
            size_t i = 0;
            const size_t n = targetCharacters.size();
            while (i < n) {
                unsigned char c = static_cast<unsigned char>(targetCharacters[i]);
                size_t step = 1;
                if (c < 0x80) step = 1;
                else if ((c & 0xE0) == 0xC0) step = 2;
                else if ((c & 0xF0) == 0xE0) step = 3;
                else if ((c & 0xF8) == 0xF0) step = 4;
                if (i + step > n) step = 1;
                chars.emplace_back(targetCharacters.substr(i, step));
                i += step;
            }
            cache[targetCharacters] = std::move(chars);
            it = cache.find(targetCharacters);
        }
        targets = it->second;
    }

    size_t pos = inputString.length();
    while (pos > 0) {
        size_t char_start = pos - 1;
        while (char_start > 0 &&
            (static_cast<unsigned char>(inputString[char_start]) & 0xC0) == 0x80) {
            char_start--;
        }
        std::string last_char = inputString.substr(char_start, pos - char_start);
        bool should_remove = false;
        for (const auto& t : targets) {
            if (last_char == t) { should_remove = true; break; }
        }
        if (!should_remove) break;
        pos = char_start;
    }
    return inputString.substr(0, pos);
}

// --- 5.6. Длина строки в UTF-8 символах ---
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

// --- 5.7. Подстрока по UTF-8 символам ---
std::string utf8_substr(const std::string & str, unsigned int start,
    unsigned int leng) {
    if (leng == 0) return "";
    const size_t ix = str.size();
    size_t i = 0;
    unsigned int chars = 0;
    size_t min_byte_index = std::string::npos;
    size_t max_byte_index = std::string::npos;
    while (i < ix) {
        if (chars == start) min_byte_index = i;
        if (chars == start + leng) { max_byte_index = i; break; }
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t step = 1;
        if (c <= 0x7F) step = 1;
        else if ((c & 0xE0) == 0xC0) { step = 2; if (i + 1 >= ix) return ""; }
        else if ((c & 0xF0) == 0xE0) { step = 3; if (i + 2 >= ix) return ""; }
        else if ((c & 0xF8) == 0xF0) { step = 4; if (i + 3 >= ix) return ""; }
        else return "";
        i += step;
        ++chars;
    }
    if (max_byte_index == std::string::npos) max_byte_index = ix;
    if (min_byte_index == std::string::npos || max_byte_index > ix) return "";
    return str.substr(min_byte_index, max_byte_index - min_byte_index);
}

// --- 5.8. Разбиение строки на UTF-8 символы ---
static std::vector<std::string> utf8_split_chars(const std::string& s) {
    std::vector<std::string> result;
    result.reserve(s.size() / 2 + 1);
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t step = 1;
        if (c < 0x80) step = 1;
        else if ((c & 0xE0) == 0xC0) step = 2;
        else if ((c & 0xF0) == 0xE0) step = 3;
        else if ((c & 0xF8) == 0xF0) step = 4;
        if (i + step > n) step = 1;
        result.emplace_back(s.substr(i, step));
        i += step;
    }
    return result;
}

// --- 5.9. Первый/последний UTF-8 символ ---
static std::string utf8_last_char(const std::string& s) {
    if (s.empty()) return "";
    size_t pos = s.length();
    size_t char_start = pos - 1;
    while (char_start > 0 &&
        (static_cast<unsigned char>(s[char_start]) & 0xC0) == 0x80) {
        char_start--;
    }
    return s.substr(char_start, pos - char_start);
}

static std::string utf8_first_char(const std::string& s) {
    if (s.empty()) return "";
    unsigned char c = static_cast<unsigned char>(s[0]);
    size_t step = 1;
    if (c < 0x80) step = 1;
    else if ((c & 0xE0) == 0xC0) step = 2;
    else if ((c & 0xF0) == 0xE0) step = 3;
    else if ((c & 0xF8) == 0xF0) step = 4;
    if (step > s.size()) step = s.size();
    return s.substr(0, step);
}

// --- 5.10. Является ли символ «словесным» ---
// Кириллица — true, пробелы и тире — false.
static bool utf8_is_word_char(const std::string& ch) {
    if (ch.empty()) return false;
    unsigned char c = static_cast<unsigned char>(ch[0]);
    if (c < 0x80) {
        return std::isalnum(c) != 0;
    }
    if (ch == "\xC2\xA0") return false;   // NBSP
    if (ch == "\xE2\x80\x94") return false; // —
    if (ch == "\xE2\x80\x93") return false; // –
    return true;
}

// --- 5.11. Нормализация ключа JSON и шаблонного токена ---
std::string normalize_json_key(const std::string& key) {
    std::string result = key;
    trim(result);
    return result;
}

// Убирает пробелы внутри тегов: "<| start_header_id |>" → "<|start_header_id|>".
std::string normalize_template_token(const std::string& s) {
    if (s.empty()) return s;
    std::string result = s;
    size_t start = result.find_first_not_of(" \t");
    size_t end = result.find_last_not_of(" \t");
    if (start == std::string::npos) return "";
    result = result.substr(start, end - start + 1);
    try {
        static const std::regex token_space_before_close(R"(<\|([^|>]+)\| >)");
        result = std::regex_replace(result, token_space_before_close, "<|$1|>");
        static const std::regex token_space_after_open(R"(<\| ([^|>]+)\|>)");
        result = std::regex_replace(result, token_space_after_open, "<|$1|>");
    }
    catch (const std::regex_error&) {}
    return result;
}

// --- 5.12. Парсинг списка float через запятую ---
// Для --tensor-split.
std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> result;
    if (s.empty()) return result;
    std::stringstream ss(s);
    std::string item;
    try {
        while (std::getline(ss, item, ',')) {
            trim(item);
            if (!item.empty()) result.push_back(std::stof(item));
        }
    }
    catch (...) { result.clear(); }
    return result;
}

// --- 5.12.1. Разбор типа KV-кэша ---
// Используется в run() для установки lcparams.type_k / type_v.
// В v33 мы НЕ используем квантованный KV-кэш, поэтому функция
// оставлена для совместимости с CLI-флагами --cache-type-k/v,
// но по умолчанию возвращает f16.
//
// ВАЖНО: q8_0 несовместим с K-shift на CUDA, поэтому мы всегда
// возвращаем f16, если только пользователь явно не задал другой тип
// через CLI (на свой риск).
static ggml_type parse_cache_type_str(const std::string& s, bool /*flash_attn*/) {
    std::string low = s;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (low == "f16"  || low == "fp16") return GGML_TYPE_F16;
    if (low == "f32"  || low == "fp32") return GGML_TYPE_F32;
    if (low == "q8_0")                  return GGML_TYPE_Q8_0;
    if (low == "q4_0")                  return GGML_TYPE_Q4_0;
    if (low == "q4_1")                  return GGML_TYPE_Q4_1;
    if (low == "q5_0")                  return GGML_TYPE_Q5_0;
    if (low == "q5_1")                  return GGML_TYPE_Q5_1;
    return GGML_TYPE_F16;
}

// --- 5.13. Текущее время в секундах (монотонное) ---
float get_current_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    ) / 1000.0f;
}

// --- 5.14. Директория для временных файлов ---
std::string getTempDir() {
    try {
        auto temp_path = std::filesystem::temp_directory_path();
        if (!temp_path.empty()) return temp_path.string();
    }
    catch (...) {}
#ifdef _WIN32
    TCHAR path_buf[MAX_PATH] = { 0 };
    DWORD ret_val = GetTempPath(MAX_PATH, path_buf);
    if (ret_val == 0 || ret_val > MAX_PATH) return "";
    if (path_buf[0] == 0) return "";
#if defined(UNICODE) || defined(_UNICODE)
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::string result = converter.to_bytes(path_buf);
        if (!result.empty() && (result.back() == '\\' || result.back() == '/'))
            result.pop_back();
        return result;
    }
    catch (...) { return ""; }
#else
    std::string result(path_buf);
    if (!result.empty() && (result.back() == '\\' || result.back() == '/'))
        result.pop_back();
    return result;
#endif
#else
    const char* tmpdir = std::getenv("TMPDIR");
    if (tmpdir && tmpdir[0] != '\0') return std::string(tmpdir);
    return "/tmp";
#endif
}

// --- 5.15. Файл-семафор XTTS ---
// Записываем "1" или "0" в файл с fsync/FlushFileBuffers, чтобы
// XTTS-сервер увидел изменение мгновенно.

// Низкоуровневая запись значения семафора с принудительным сбросом
// на диск. На Windows используется CreateFileA с FILE_FLAG_WRITE_THROUGH,
// на POSIX — write + fsync.
static void write_semaphore_instant(const std::string & filepath, bool allowed) {
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
#else
    int fd = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        write(fd, value, 1);
        fsync(fd);
        close(fd);
    }
#endif
}

// Управление семафором XTTS.
//
// ВАЖНО: путь к семафору ВСЕГДА берётся из системной временной
// директории. Параметр path игнорируется намеренно — XTTS-сервер
// (Python) ждёт файл именно в %TEMP% / TMPDIR.
static void allow_xtts_file(std::string & path, int xtts_play_allowed) {
    // Однократная инициализация глобального пути.
    if (g_xtts_control_file_path.empty()) {
        // ВСЕГДА используем временную директорию — так было раньше,
        // и XTTS-сервер ждёт файл именно там.
        // Параметр path игнорируется намеренно.
        std::string temp_path = getTempDir();
        if (temp_path.empty()) return;
#if __cplusplus >= 201703L
        std::filesystem::path p(temp_path);
        g_xtts_control_file_path = (p / "xtts_play_allowed.txt").string();
#else
        g_xtts_control_file_path = temp_path;
        if (g_xtts_control_file_path.back() != '/' &&
            g_xtts_control_file_path.back() != '\\') {
            g_xtts_control_file_path += '/';
        }
        g_xtts_control_file_path += "xtts_play_allowed.txt";
#endif
    }

    // Отдаём вызывающему актуальный путь (некоторые места его используют).
    path = g_xtts_control_file_path;

    std::lock_guard<std::mutex> lock(g_xtts_control_mutex);

    // Не пишем, если значение не изменилось.
    bool new_value = (xtts_play_allowed == 1);
    if (g_last_semaphore_initialized && g_last_semaphore_value == new_value) {
        return;
    }
    g_last_semaphore_value = new_value;
    g_last_semaphore_initialized = true;

    write_semaphore_instant(g_xtts_control_file_path, new_value);

    if (g_verbose_mode.load()) {
        fprintf(stderr, "[Semaphore] Записано %d в %s\n",
            xtts_play_allowed, g_xtts_control_file_path.c_str());
    }
}

// --- 5.16. Токенизация и детокенизация ---

// Базовая токенизация. parse_special = false — пользовательский
// текст не должен интерпретировать <|...|> как control-токены.
// Для шаблонных фрагментов используйте llama_tokenize_ex().
static std::vector<llama_token> llama_tokenize(
    struct llama_context* ctx,
    const std::string& text,
    bool add_bos) {
    if (!ctx) return {};
    const llama_model* model = llama_get_model(ctx);
    if (!model) return {};
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};
    int n_tokens = static_cast<int>(text.length()) + (add_bos ? 1 : 0);
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(),
        result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        // Буфер был мал — перевыделяем и пробуем снова.
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

// --- 5.16.1. Токенизация с явным управлением parse_special ---
// В отличие от llama_tokenize(), эта версия позволяет указать,
// интерпретировать ли <|...|> как control-токены.
// Для шаблонных фрагментов — true, для пользовательского текста — false.
static std::vector<llama_token> llama_tokenize_ex(
    struct llama_context* ctx,
    const std::string& text,
    bool add_bos,
    bool parse_special) {
    if (!ctx) return {};
    const llama_model* model = llama_get_model(ctx);
    if (!model) return {};
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};
    int n_tokens = static_cast<int>(text.length()) + (add_bos ? 1 : 0);
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(),
        result.data(), result.size(), add_bos, parse_special);
    if (n_tokens < 0) {
        result.resize(static_cast<size_t>(-n_tokens));
        int check = llama_tokenize(vocab, text.data(), text.length(),
            result.data(), result.size(), add_bos, parse_special);
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

// Детокенизация: токен → строковое представление.
static std::string llama_token_to_piece(
    const struct llama_context* ctx,
    llama_token token) {
    if (!ctx) return "";
    const llama_model* model = llama_get_model(ctx);
    if (!model) return "";
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return "";
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

// --- 5.17. Санитизация пользовательского текста ---
// Удаляет из пользовательского текста любые конструкции вида <|...|>,
// чтобы пользователь не мог внедрить служебный токен.
static std::string sanitize_user_text(const std::string& text) {
    std::string result = text;
    try {
        static const std::regex re_special_token(R"(<\|[^|]*\|>)", std::regex::ECMAScript);
        result = std::regex_replace(result, re_special_token, "");
    } catch (const std::regex_error&) {
        // Fallback: удаляем известные токены.
        result = string_replace_all(result, "<|start_header_id|>", "");
        result = string_replace_all(result, "<|end_header_id|>", "");
        result = string_replace_all(result, "<|eot_id|>", "");
        result = string_replace_all(result, "<|eo_id|>", "");
    }
    return result;
}


// ============================================================================
// 6. ФИЛЬТР ЛОГОВ WHISPER
// ============================================================================
// Whisper.cpp очень болтлив. Пропускаем только ошибки, чтобы не
// засорять stderr.
static void whisper_log_filtered(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (text == nullptr) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}


// ============================================================================
// 7. ЛОГИРОВАНИЕ В ФАЙЛ
// ============================================================================
// Простой логгер с временной меткой [HH:MM:SS].
static void log_line(const std::string& msg) {
    if (!g_log_enabled.load()) return;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_log_file.is_open()) return;
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm_buf);
    g_log_file << "[" << buf << "] " << msg << "\n";
    g_log_file.flush();
}

// ============================================================================
// 8. СТРУКТУРА ПАРАМЕТРОВ
// ============================================================================
// Все параметры командной строки и производные от них значения.
struct whisper_params {
    // --- 8.1. Производительность ---
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_gpu_layers = 999;
    bool use_gpu = true;
    bool flash_attn = false;
    int main_gpu = 0;
    std::string split_mode = "none";
    std::vector<float> tensor_split;

    // --- 8.2. Аудиозахват ---
    int32_t voice_ms = 10000;   // макс. длина одного сегмента речи
    int32_t capture_id = -1;    // ID микрофона (-1 = дефолтный)

    // --- 8.3. Whisper ---
    int32_t max_tokens = 64;
    int32_t audio_ctx = 0;

    // Скорость Whisper: greedy-декодирование. beam_size=5 (дефолт
    // whisper.cpp) в 3-5 раз медленнее.
    int32_t beam_size = 1;
    int32_t best_of = 1;
    float   temperature_inc = 0.0f;

    // --- 8.4. Энергетический VAD ---
    // Silero VAD удалён: на RTX 3060 с параллельным XTTS и LLaMA
    // он добавлял 200-500 мс на каждую транскрибацию.
    float vad_thold = 0.6f;
    float vad_start_thold = 0.000270f;
    float vad_last_ms = 800.0f;
    float freq_thold = 90.0f;

    // --- 8.5. Прерывание ---
    int32_t interrupt_check_ms = 50;
    int32_t interrupt_threshold_ms = 100;

    // --- 8.6. LLaMA ---
    int32_t ctx_size = 2048;
    int32_t batch_size = 64;
    int32_t n_predict = 512;
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
    bool seqrep = true;

    // --- 8.6.1. Автонастройка (PATCH F, v33) ---
    // Если пользователь явно задал --n_predict / --batch-size /
    // --min-tokens — соответствующий auto_* сбрасывается в false.
    // Иначе значение вычисляется от ctx_size в run() до инициализации LLaMA.
    //
    // ВАЖНО (v33): n_keep больше НЕ участвует в автонастройке. Он
    // зафиксирован на 512 (см. run() 22.1.5). Это ключевой параметр
    // K-shift, его нельзя вычислять от ctx_size — иначе буферная
    // зона сдвига сжимается, и генерация сбоит.
    bool auto_n_predict   = true;
    bool auto_batch_size  = true;
    bool auto_min_tokens  = true;
    // Типы KV-кэша. Пустая строка = дефолт llama.cpp (f16).
    // Явное задание --cache-type-k/v позволяет переопределить,
    // но использовать q8_0 вместе с K-shift на CUDA нельзя.
    std::string cache_type_k = "";
    std::string cache_type_v = "";

    // --- 8.7. TTS ---
    std::string xtts_voice = "Эмма";
    std::string xtts_url = "http://localhost:8020/";
    std::string xtts_control_path = "xtts_play_allowed.txt";
    bool xtts_intro = false;
    int sleep_before_xtts = 0;
    int split_after = 0;

    // --- 8.8. Имена ---
    std::string person = "Друг";
    std::string bot_name = "Эмма";
    std::string wake_cmd = "";
    std::string heard_ok = "";

    // --- 8.9. Язык ---
    std::string language = "ru";
    bool translate = false;

    // --- 8.10. Модели ---
    std::string model_wsp = "whisper-ggml-medium-q4_0.bin";
    std::string model_llama = "saiga_yandexgpt_8b_Q5_K.gguf";

    // --- 8.11. Флаги ---
    bool speed_up = false;
    bool print_special = false;
    bool print_energy = false;
    bool no_timestamps = true;
    bool verbose_prompt = false;
    bool verbose = false;
    bool push_to_talk = false;
    std::string google_url = "http://localhost:8003/";
    std::string prompt = "";
    std::string alt_prompt_file = "";
    std::string instruct_preset = "";
    std::string path_session = "";
    std::string stop_words = "";

    // --- 8.12. Автопродолжение ---
    bool auto_continue = true;
    int32_t continue_max_ms = 10000;

    // --- 8.13. Instruct-пресет ---
    // 7 универсальных полей. Работают для Llama3, Qwen, Mistral,
    // Gemma, Yandex, Alpaca, Vicuna, none.
    std::map<std::string, std::string> instruct_preset_data = {
        {"system_prompt_prefix", ""},
        {"system_prompt_suffix", ""},
        {"user_message_prefix", ""},
        {"user_message_suffix", ""},
        {"bot_message_prefix", ""},
        {"bot_message_suffix", ""},
        {"stop_sequence", ""}
    };

    // --- 8.14. ChatML-шаблон ---
    // Всегда LLM_CHAT_TEMPLATE_UNKNOWN — автопоиск отключён.
    llm_chat_template chat_template = LLM_CHAT_TEMPLATE_UNKNOWN;
};


// ============================================================================
// 9. ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
// ============================================================================
// Разбор argv. Все параметры имеют значения по умолчанию (см. раздел 8),
// парсер только переопределяет их. При ошибке печатает usage и
// возвращает false — run() завершается.

// --- 9.1. Печать справки ---
// Выводит все поддерживаемые опции с текущими значениями по умолчанию.
void whisper_print_usage(int argc, char** argv, const whisper_params & params) {
    (void)argc;
    fprintf(stderr, "\nusage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n", params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum tokens per audio chunk\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N    [%-7d] whisper beam size (1=greedy, fastest)\n", params.beam_size);
    fprintf(stderr, "  -bo N,    --best-of N      [%-7d] whisper best_of (1=off)\n", params.best_of);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n", params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] energy vad threshold\n", params.vad_thold);
    fprintf(stderr, "  -vths N,  --vad-start-thold[%-7.6f] energy vad min level to start\n", params.vad_start_thold);
    fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7.2f] energy vad min silence after speech\n", params.vad_last_ms);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name\n", params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name\n", params.bot_name.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -tr,      --translate                translate from source language to english\n");
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n", params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n", params.model_llama.c_str());
    fprintf(stderr, "  --ctx_size N               [%-7d] size of the prompt context\n", params.ctx_size);
    fprintf(stderr, "  -b N,     --batch-size N   [auto: ctx/8, 128..2048] logical batch size for prompt ingestion\n");
    fprintf(stderr, "  -n N,     --n_predict N    [auto: ctx/16, 256..1024] max number of tokens to predict\n");
    fprintf(stderr, "  --temp N                   [%-7.2f] temperature\n", params.temp);
    fprintf(stderr, "  --top_k N                  [%-7d] top_k\n", params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p\n", params.top_p);
    fprintf(stderr, "  --min_p N                  [%-7.2f] min_p\n", params.min_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repeat_penalty\n", params.repeat_penalty);
    fprintf(stderr, "  --repeat_last_n N          [%-7d] repeat_last_n\n", params.repeat_last_n);
    fprintf(stderr, "  --n_keep N                 [%-7d] keep first n_tokens after context_shift (v33: fixed 512)\n", params.n_keep);
    fprintf(stderr, "  --min-tokens N             [auto: max(20, ctx/256)] min new tokens to output\n");
    fprintf(stderr, "  --cache-type-k NAME        [default: f16] K cache type (q8_0 incompatible with K-shift on CUDA)\n");
    fprintf(stderr, "  --cache-type-v NAME        [default: f16] V cache type (q8_0 incompatible with K-shift on CUDA)\n");
    fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n", params.xtts_voice.c_str());
    fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts server URL\n", params.xtts_url.c_str());
    fprintf(stderr, "  --xtts-control-path NAME   [%-7s] IGNORED (семафор всегда в %%TEMP%%)\n", params.xtts_control_path.c_str());
    fprintf(stderr, "  --xtts-intro                        play short intro before response\n");
    fprintf(stderr, "  --sleep-before-xtts N      [%-7d] sleep llama inference before xtts, ms\n", params.sleep_before_xtts);
    fprintf(stderr, "  --split-after N            [%-7d] chunk size for xtts, 0=off\n", params.split_after);
    fprintf(stderr, "  --google-url TEXT          [%-7s] google-serper server URL\n", params.google_url.c_str());
    fprintf(stderr, "  --stop-words TEXT          [%-7s] llama stop words separated by ;\n", params.stop_words.c_str());
    fprintf(stderr, "  --prompt-file FNAME                 file with custom prompt (REQUIRED)\n");
    fprintf(stderr, "  --alt-prompt-file FNAME             file with prompt for Leo\n");
    fprintf(stderr, "  --instruct-preset TEXT              instruct preset without .json\n");
    fprintf(stderr, "  --session FNAME                     file to cache model state\n");
    fprintf(stderr, "  --auto-continue                     enable auto-continue (default: on)\n");
    fprintf(stderr, "  --no-auto-continue                  disable auto-continue\n");
    fprintf(stderr, "  --continue-max-ms N                 max ms after barge-in\n");
    fprintf(stderr, "  --allow-newline                     allow newlines in llama output\n");
    fprintf(stderr, "  --seqrep                            enable sequence repetition penalty\n");
    fprintf(stderr, "  --verbose                           print speed, debug info and write log file\n");
    fprintf(stderr, "  --verbose-prompt                    print the full prompt at start\n");
    fprintf(stderr, "  --print-energy                      print energy levels for vad debug\n");
    fprintf(stderr, "  --push-to-talk                      hold Alt to speak (default: off)\n");
    fprintf(stderr, "\n");
}

// --- 9.2. Основной парсер ---
// Итерируется по argv. Для каждой опции проверяет, что следующий
// аргумент существует (иначе — ошибка). Использует stoi/stof,
// ловит исключения.
//
// PATCH F (v33): при явном задании --n_predict / --batch-size /
// --min-tokens сбрасывается соответствующий auto_* флаг.
// n_keep больше не имеет auto_* — он фиксирован на 512.
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
                if (tensor_split_str.empty()) { std::cerr << "Error: empty tensor-split list" << std::endl; return false; }
                params.tensor_split = parse_float_list(tensor_split_str);
                if (params.tensor_split.empty()) { std::cerr << "Error: failed to parse tensor-split list" << std::endl; return false; }
                float sum = 0.0f;
                for (float val : params.tensor_split) {
                    if (val < 0.0f || val > 1.0f) { std::cerr << "Error: tensor-split values must be between 0.0 and 1.0" << std::endl; return false; }
                    sum += val;
                }
                if (fabs(sum - 1.0f) > 0.001f) {
                    std::cerr << "Warning: tensor-split values sum to " << sum << " (expected ~1.0)" << std::endl;
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
            else if (arg == "-bs" || arg == "--beam-size") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.beam_size = std::stoi(argv[++i]);
            }
            else if (arg == "-bo" || arg == "--best-of") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.best_of = std::stoi(argv[++i]);
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
                params.auto_batch_size = false;   // PATCH F (v33)
            }
            else if (arg == "-n" || arg == "--n_predict") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_predict = std::stoi(argv[++i]);
                params.auto_n_predict = false;    // PATCH F (v33)
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
                // n_keep больше не имеет auto_* — пользователь может
                // переопределить его вручную, если хочет.
            }
            else if (arg == "--min-tokens") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.min_tokens = std::stoi(argv[++i]);
                params.auto_min_tokens = false;   // PATCH F (v33)
            }
            // --- PATCH F (v33): типы KV-кэша ---
            // Явное переопределение. По умолчанию — f16 (дефолт llama.cpp).
            // q8_0 использовать нельзя вместе с K-shift на CUDA.
            else if (arg == "--cache-type-k") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.cache_type_k = argv[++i];
            }
            else if (arg == "--cache-type-v") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.cache_type_v = argv[++i];
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
            else if (arg == "-vp" || arg == "--verbose-prompt") { params.verbose_prompt = true; }
            else if (arg == "--verbose") { params.verbose = true; g_verbose_mode.store(true); }
            else if (arg == "--push-to-talk") { params.push_to_talk = true; }
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
            else if (arg == "--auto-continue") { params.auto_continue = true; }
            else if (arg == "--no-auto-continue") { params.auto_continue = false; }
            else if (arg == "--continue-max-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.continue_max_ms = std::stoi(argv[++i]);
            }
            else if (arg == "--prompt-file") {
                if (i + 1 >= argc) { whisper_print_usage(argc, argv, params); return false; }
                std::ifstream file(argv[++i]);
                if (!file.is_open()) { std::cerr << "Failed to open prompt file: " << argv[i] << std::endl; return false; }
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(params.prompt));
                if (!params.prompt.empty() && params.prompt.back() == '\n') params.prompt.pop_back();
            }
            else if (arg == "--alt-prompt-file") {
                if (i + 1 >= argc) { whisper_print_usage(argc, argv, params); return false; }
                params.alt_prompt_file = argv[++i];
            }
            else if (arg == "--session") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.path_session = argv[++i];
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

// ============================================================================
// 10. ОПРЕДЕЛЕНИЕ ГОЛОСОВЫХ КОМАНД
// ============================================================================

// --- 10.1. Команда «стоп» ---
// Распознаёт короткие фразы: «стоп», «хватит», «тихо» и т.п.
// Ограничение в 4 слова — чтобы не путать с обычной речью.
static bool is_stop_command(const std::string& text_lower) {
    if (text_lower.empty()) return false;
    std::string clean = text_lower;
    clean = string_replace_all(clean, ".", " ");
    clean = string_replace_all(clean, "!", " ");
    clean = string_replace_all(clean, "?", " ");
    clean = string_replace_all(clean, ",", " ");
    trim(clean);
    if (clean.empty()) return false;
    int word_count = 1;
    for (size_t i = 1; i < clean.size(); ++i) {
        if (clean[i] == ' ' && clean[i - 1] != ' ') word_count++;
    }
    if (word_count > 4) return false;
    if (clean == "стоп" || clean == "stop" || clean == "хватит" ||
        clean == "тихо" || clean == "стой" || clean == "остановись" ||
        clean == "прекрати" || clean == "замолчи") return true;
    if (clean.find("стоп") == 0 || clean.find("stop") == 0) return true;
    return false;
}

// --- 10.2. Извлечение ключевого слова из команды ---
// Например: «погугли погода в лондоне» → «погода в лондоне».
// Используется для команды google и call.
std::string ParseCommandAndGetKeyword(std::string textHeardTrimmed,
    const std::string& command = "google") {
    textHeardTrimmed = StripPunctuationMarks(textHeardTrimmed);
    std::string sanitizedInput = textHeardTrimmed;
    size_t pos = 0;
    bool startsWithPrefix = false;

    // Убираем вежливые вставки, которые Whisper иногда добавляет.
    static const std::unordered_set<std::string> please_needles = {
        "can you hear me", "Can you hear me", "Are you here", "are you here",
        "Do you hear me", "do you hear me", "Пожалуйста", "пожалуйста",
        "Позови", "позови", "ты тут", "Ты тут", "ты здесь", "Ты здесь",
        "ты меня слышишь", "Ты меня слышишь", "ты слышишь меня",
        "Ты слышишь меня", "Hey", "hey", "please", "Please",
        "can you", "Can you", "let's", "Let's",
        "What do you think", "Что ты думаешь", "что ты думаешь",
        "Что ты об этом думаешь", "что ты об этом думаешь"
    };
    for (const auto& prefix : please_needles) {
        sanitizedInput = string_replace_all(sanitizedInput, prefix, "");
    }
    trim(sanitizedInput);

    // Для команды google ищем префикс «погугли» и т.п.
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

    // Если префикс не нашли — ищем саму команду в строке.
    if (!startsWithPrefix) {
        size_t found = sanitizedInput.find(command);
        if (found != std::string::npos) {
            size_t base = found + command.size();
            while (base < sanitizedInput.size() &&
                (std::isspace(static_cast<unsigned char>(sanitizedInput[base])) ||
                 sanitizedInput[base] == ':')) { ++base; }
            pos = base;
        } else {
            size_t foundCall = sanitizedInput.find("Call");
            if (foundCall != std::string::npos) {
                size_t base = foundCall + 4;
                while (base < sanitizedInput.size() &&
                    (std::isspace(static_cast<unsigned char>(sanitizedInput[base])) ||
                     sanitizedInput[base] == ':')) { ++base; }
                pos = base;
            } else { pos = 0; }
        }
    }

    // Для команды call — нормализуем падеж имени.
    if (command == "call") {
        trim(sanitizedInput);
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
            // у → а, ю → я (винительный падеж русских имён).
            utf8_rule_applied = safeReplace(len - 2, "\xD1\x83", "\xD0\xB0") || utf8_rule_applied;
            utf8_rule_applied = safeReplace(len - 2, "\xD1\x8E", "\xD1\x8F") || utf8_rule_applied;
            if (utf8_rule_applied) trim(sanitizedInput);
        }
        if (sanitizedInput.size() >= 2) {
            thread_local const std::regex re_male_genitive_ogo_ego(R"((.+)([оe]го)$)", std::regex_constants::icase);
            thread_local const std::regex re_male_u(R"((.+)у$)", std::regex_constants::icase);
            thread_local const std::regex re_male_a(R"((.+)а$)", std::regex_constants::icase);
            thread_local const std::regex re_male_om(R"((.+)ом$)", std::regex_constants::icase);
            thread_local const std::regex re_male_em(R"((.+)ем$)", std::regex_constants::icase);
            thread_local const std::regex re_male_yu(R"((.+)ю$)", std::regex_constants::icase);
            thread_local const std::regex re_male_yem(R"((.+)еем$)", std::regex_constants::icase);
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_genitive_ogo_ego, "$1");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_om, "$1");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_em, "$1й");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_yem, "$1й");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_yu, "$1й");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_u, "$1");
            sanitizedInput = std::regex_replace(sanitizedInput, re_male_a, "$1");
        }
        trim(sanitizedInput);
        textHeardTrimmed = sanitizedInput;
    }

    return textHeardTrimmed.substr(pos);
}


// ============================================================================
// 11. URL-КОДИРОВАНИЕ И HTTP-ЗАПРОСЫ
// ============================================================================

// --- 11.1. URL-кодирование ---
// Используется для google-команды, чтобы кириллица в запросе
// не ломала URL.
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

// --- 11.2. Callback записи HTTP-ответа ---
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    if (userp) {
        ((std::string*)userp)->append(static_cast<char*>(contents), realsize);
    }
    return realsize;
}

// --- 11.3. Callback прогресса ---
// Возвращает 1, если запрос нужно отменить. Используется для
// прерывания долгих TTS-запросов.
static int progress_callback(void* /*clientp*/,
    curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
    curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    if (g_cancel_tts_requests.load()) return 1;
    return 0;
}

// --- 11.4. Простой GET-запрос ---
std::string send_curl(std::string url) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        res = curl_easy_perform(curl);
        if (res != CURLE_OK && g_verbose_mode.load()) {
            fprintf(stderr, "[cURL] Ошибка GET-запроса: %s\n", curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

// --- 11.5. URL → произносимый текст ---
// Для google-команды: «http://example.com/path» → «example dot com slash path».
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
    if (!clean_url.empty() && clean_url.back() == '/') clean_url.pop_back();
    std::vector<std::string> parts;
    std::string current;
    for (char c : clean_url) {
        if (c == '.' || c == '/' || c == '-' || c == ' ') {
            if (!current.empty()) { parts.push_back(current); current.clear(); }
            if (c == '.') parts.push_back("dot");
            else if (c == '/') parts.push_back("slash");
            else if (c == '-') parts.push_back("dash");
            else if (c == ' ') parts.push_back("underscore");
        } else { current += c; }
    }
    if (!current.empty()) parts.push_back(current);
    for (size_t i = 0; i < parts.size(); ++i) {
        if (!result.empty()) result += " ";
        result += parts[i];
    }
    return result;
}


// ============================================================================
// 12. СБОРКА ПРОИЗВОДНЫХ ИЗ INSTRUCT-ПРЕСЕТА
// ============================================================================
// Все стоп-строки, антипромпты и префиксы выводятся из 7 полей
// JSON-пресета. Это делает код универсальным: новый пресет
// добавляется без перекомпиляции.

// --- 12.1. Структура производных ---
struct PresetDerived {
    // Префиксы/суффиксы из JSON.
    std::string sys_prefix;
    std::string sys_suffix;
    std::string user_prefix;
    std::string user_suffix;
    std::string bot_prefix;
    std::string bot_suffix;
    std::string stop_sequence;

    // Стоп-строки: stop_sequence + суффиксы сторон.
    // При появлении в token_accumulator генерация останавливается.
    std::vector<std::string> stop_strings;

    // Антипромпты: те же стоп-строки + префиксы сторон.
    // Нужны, если модель начнёт новый ход без явного суффикса.
    std::vector<std::string> antiprompts;

    // Известные статусы []. Это внутренний протокол проекта,
    // не часть пресета.
    std::unordered_set<std::string> statuses;
};

// --- 12.2. Сборка PresetDerived из JSON-данных ---
static PresetDerived build_preset_derived(
    const std::map<std::string, std::string>& data)
{
    PresetDerived d;

    // Универсальный геттер: если поля нет — пустая строка.
    auto get = [&](const std::string& k) -> std::string {
        auto it = data.find(k);
        return (it == data.end()) ? "" : it->second;
    };

    d.sys_prefix    = get("system_prompt_prefix");
    d.sys_suffix    = get("system_prompt_suffix");
    d.user_prefix   = get("user_message_prefix");
    d.user_suffix   = get("user_message_suffix");
    d.bot_prefix    = get("bot_message_prefix");
    d.bot_suffix    = get("bot_message_suffix");
    d.stop_sequence = get("stop_sequence");

    // Стоп-строки = stop_sequence + суффиксы сторон.
    auto add_stop = [&](const std::string& s) {
        if (!s.empty() &&
            std::find(d.stop_strings.begin(), d.stop_strings.end(), s)
                == d.stop_strings.end()) {
            d.stop_strings.push_back(s);
        }
    };
    add_stop(d.stop_sequence);
    add_stop(d.bot_suffix);
    add_stop(d.user_suffix);

    // Антипромпты = стоп-строки + префиксы сторон.
    d.antiprompts = d.stop_strings;
    auto add_ap = [&](const std::string& s) {
        if (!s.empty() &&
            std::find(d.antiprompts.begin(), d.antiprompts.end(), s)
                == d.antiprompts.end()) {
            d.antiprompts.push_back(s);
        }
    };
    add_ap(d.bot_prefix);
    add_ap(d.user_prefix);

    // Опциональное поле stop_words (разделитель ';').
    {
        auto it = data.find("stop_words");
        if (it != data.end() && !it->second.empty()) {
            const std::string& s = it->second;
            size_t start = 0;
            while (start <= s.size()) {
                size_t end = s.find(';', start);
                if (end == std::string::npos) end = s.size();
                std::string word = s.substr(start, end - start);
                trim(word);
                if (!word.empty()) add_ap(word);
                if (end == s.size()) break;
                start = end + 1;
            }
        }
    }

    // Известные статусы — внутренний протокол проекта.
    d.statuses = {
        "google", "time", "date", "reset", "del", "regen", "repeat",
        "tts", "int", "stop", "call leo", "call emma"
    };

    return d;
}


// ============================================================================
// 13. ТРАНСКРИБАЦИЯ WHISPER
// ============================================================================
// Транскрибация аудио-сегмента через Whisper. Скорость:
// beam_size=1, best_of=1, temperature_inc=0.0f, vad=false.

// --- 13.1. Транскрибация ---
static std::string transcribe(
    whisper_context* ctx,
    const whisper_params& params,
    const std::vector<float>& pcmf32,
    const std::string& prompt_text,
    float& prob,
    int64_t& t_ms,
    bool translate = false) {
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

    whisper_full_params wparams = whisper_full_default_params(
        WHISPER_SAMPLING_GREEDY);

    // Явно отключаем Silero VAD — детекцию речи делает
    // энергетический VAD в аудио-потоке.
    wparams.vad = false;

    // Скорость: greedy-декодирование.
    wparams.beam_search.beam_size = params.beam_size;
    wparams.greedy.best_of = params.best_of;
    wparams.temperature_inc = params.temperature_inc;

    // Обработка prompt (подсказка для Whisper).
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
        } else {
            wparams.prompt_tokens = nullptr;
            wparams.prompt_n_tokens = 0;
        }
    } else {
        wparams.prompt_tokens = nullptr;
        wparams.prompt_n_tokens = 0;
    }

    // Отключаем лишний вывод.
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.no_timestamps = true;
    wparams.translate = translate;
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.token_timestamps = false;
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;

    // Параметры качества (не скорости).
    wparams.temperature = 0.2f;
    wparams.entropy_thold = 2.4f;
    wparams.logprob_thold = -1.0f;
    wparams.no_speech_thold = 0.6f;
    wparams.length_penalty = 0.5f;
    wparams.max_len = 96;

    // Ограничение max_tokens по размеру модели.
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

    // Ограничение audio_ctx по размеру модели.
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
            std::cerr << "Ошибка: не удалось выполнить транскрипцию аудио"
                      << std::endl;
        }
        const auto t_end = std::chrono::high_resolution_clock::now();
        t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();
        return "";
    }

    // Собираем текст и среднюю вероятность.
    int prob_n = 0;
    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text != nullptr) result += text;
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);
            prob += token.p;
            ++prob_n;
        }
    }
    if (prob_n > 0) prob /= static_cast<float>(prob_n);
    else prob = 0.0f;

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
// ============================================================================
// 14. ДЕТЕКТОР ГАЛЛЮЦИНАЦИЙ WHISPER
// ============================================================================
// Whisper склонен галлюцинировать на шумной/непонятной речи:
// субтитры, музыка, повторяющиеся фразы. Этот детектор отсеивает
// такие случаи.
static bool is_hallucination(const std::string& text) {
    // Точные совпадения (короткие мусорные фразы).
    static const std::unordered_set<std::string> exact_matches = {
        " ", "!", ".", "Sil", "У-у-у!",
        "Редактор субтитров", "Продолжение следует", "Спасибо за просмотр",
        "Субтитры", "Перевод", "Translated by", "Thanks for watching",
        "Thank you for listening.", "Thank you guys.", "Я передам вам.",
        "молочи", "сочивающие", "направленный", "офигенно",
        "трендишь", "тарантиш", "отвыка", "отвык", "пеной",
        "поднявший", "вкалывала", "минерал", "купила",
        "звук", "музыка", "шум", "тишина", "пауза", "смех",
        "аплодисменты", "кашель", "вздох", "скрип", "стук",
        "ПЕСНЯ", "СМЕХ", "СТУК",
    };
    if (exact_matches.count(text)) return true;

    // Подстроки (шаблоны субтитров, рекламы).
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

    // Повторяющийся паттерн (aaaa, abababab и т.п.).
    {
        std::string clean = text;
        clean = string_replace_all(clean, ",", "");
        clean = string_replace_all(clean, ".", "");
        clean = string_replace_all(clean, "!", "");
        clean = string_replace_all(clean, "?", "");
        clean = string_replace_all(clean, " ", "");
        if (clean.length() > 10) {
            for (size_t len = 3; len <= 20 && len <= clean.length() / 2; ++len) {
                std::string pattern = clean.substr(0, len);
                size_t count = 0;
                size_t pos = 0;
                while ((pos = clean.find(pattern, pos)) != std::string::npos) {
                    count++;
                    pos += len;
                }
                if (count >= 4) return true;
            }
        }
    }

    // Слишком много одинаковых слов.
    {
        std::string clean = text;
        clean = string_replace_all(clean, ",", " ");
        clean = string_replace_all(clean, ".", " ");
        clean = string_replace_all(clean, "!", " ");
        clean = string_replace_all(clean, "?", " ");
        std::istringstream iss(clean);
        std::string word;
        std::unordered_map<std::string, int> word_counts;
        int total_words = 0;
        while (iss >> word) {
            if (word.length() >= 2) {
                word_counts[word]++;
                total_words++;
            }
        }
        if (total_words >= 3) {
            for (const auto& [w, c] : word_counts) {
                if (c >= total_words * 0.5) return true;
            }
        }
    }

    // Слишком короткое (и не «да/нет/ок»).
    if (utf8_length(text) < 3 &&
        text.find("да") == std::string::npos &&
        text.find("нет") == std::string::npos &&
        text.find("ок") == std::string::npos &&
        text.find("ага") == std::string::npos &&
        text.find("угу") == std::string::npos &&
        text.find("ясно") == std::string::npos) {
        return true;
    }

    // Всё одинаковые символы (3-20).
    if (text.length() > 3 && text.length() < 20) {
        bool all_same = true;
        for (size_t i = 1; i < text.length(); ++i) {
            if (text[i] != text[0]) { all_same = false; break; }
        }
        if (all_same && text.length() >= 3) return true;
    }

    return false;
}

// --- 14.1. Учёт транскрипции ---
// PATCH 5 (v34): переименован g_hallucination_count →
// g_transcription_count. Функция вызывается при каждой успешной
// транскрипции (после is_hallucination == false), счётчик
// инкрементируется. Не путать с самим детектором галлюцинаций.
static void note_transcription(const std::string& text) {
    if (text.empty()) return;
    (void)text;
    g_transcription_count.fetch_add(1);
}


// ============================================================================
// 15. ОЧИСТКА ТЕКСТА ДЛЯ TTS
// ============================================================================
// Задача — превратить вывод модели в чистый текст для XTTS.
// Правила:
//   (круглые)     → ". " (содержимое удаляется, ставим границу предложения)
//   [квадратные]  → ", содержимое" (содержимое сохраняется)
//   {фигурные}    → "" (удаляются полностью)
//   <|...|>       → "" (любые спецтокены, включая <|eo_id|>)
//   markdown      → очищается
//   HTML-теги     → пробел
//
// PATCH C (v33): сохранены все правки для чистоты TTS:
//   - круглые скобки → ". " вместо ", ";
//   - лимит длины круглых скобок 300;
//   - содержимое *...* удаляется (это описания действий);
//   - P.S./P.P.S. → словарная замена;
//   - маркеры нумерованных списков внутри строки удаляются.

// --- 15.1. Базовая очистка ---
static std::string clean_text_for_tts(const std::string& text) {
    std::string result = text;

    // Ранний выход, если текст — чистый маркер списка ("1.", "-").
    {
        std::string stripped = text;
        trim_spaces_only(stripped);
        while (!stripped.empty() && (stripped.front() == '\n' || stripped.front() == '\r')) {
            stripped.erase(0, 1);
        }
        trim_spaces_only(stripped);
        while (!stripped.empty() &&
               (stripped.back() == '.' || stripped.back() == ')' ||
                stripped.back() == ':' || stripped.back() == '-' ||
                stripped.back() == '*' || stripped.back() == '+' ||
                stripped.back() == '\n' || stripped.back() == '\r')) {
            stripped.pop_back();
        }
        trim_spaces_only(stripped);
        if (!stripped.empty()) {
            bool all_digits = true;
            size_t n_digits = 0;
            for (char c : stripped) {
                if (std::isdigit(static_cast<unsigned char>(c))) n_digits++;
                else { all_digits = false; break; }
            }
            bool is_marker = false;
            if (all_digits && n_digits > 0 && n_digits <= 2) is_marker = true;
            else if (utf8_length(stripped) == 1) is_marker = true;
            if (is_marker) {
                if (g_verbose_mode.load()) {
                    fprintf(stderr, "[TTS] Пропущен маркер: '%s'\n", text.c_str());
                }
                return "";
            }
        }
    }

    // Regex — static const, компилируются один раз.
    // Порядок важен: фигурные (полное удаление), квадратные (с
    // содержимым), круглые (без содержимого).
    try {
        static const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
        result = std::regex_replace(result, re_curly, "");

        static const std::regex re_square(R"(\[([^\[\]]{1,80})\])", std::regex::ECMAScript);
        result = std::regex_replace(result, re_square, ", $1");

        // PATCH C (v33): круглые скобки → ". " (граница предложения),
        // лимит 300 символов — чтобы ловить длинные
        // ремарки вида "(Плюс я всегда готова помочь...)".
        static const std::regex re_round(R"(\([^()]{1,300}\))", std::regex::ECMAScript);
        result = std::regex_replace(result, re_round, ". ");

        // Markdown: bold/italic/strike — снимаем формат.
        static const std::regex re_bold_double(R"(\*\*([^*]+)\*\*)", std::regex::ECMAScript);
        static const std::regex re_bold_under(R"(__([^_]+)__)", std::regex::ECMAScript);
        static const std::regex re_italic_star(R"(\*([^*]+)\*)", std::regex::ECMAScript);
        static const std::regex re_italic_under(R"(_([^_]+)_)", std::regex::ECMAScript);
        static const std::regex re_strike(R"(~~([^~]+)~~)", std::regex::ECMAScript);
        static const std::regex re_multidot(R"(\.{2,})", std::regex::ECMAScript);
        static const std::regex re_md_noise(R"([#*_~`>])", std::regex::ECMAScript);
        static const std::regex re_spaces_tabs(R"([ \t]+)", std::regex::ECMAScript);

        result = std::regex_replace(result, re_bold_double, "$1 ");
        result = std::regex_replace(result, re_bold_under, "$1 ");
        // PATCH C (v33): содержимое *...* удаляется полностью —
        // это описания действий (улыбается, вздыхает), которые TTS
        // не должен озвучивать.
        result = std::regex_replace(result, re_italic_star, " ");
        result = std::regex_replace(result, re_italic_under, "$1 ");
        result = std::regex_replace(result, re_strike, "$1 ");
        result = std::regex_replace(result, re_multidot, ". ");
        result = std::regex_replace(result, re_md_noise, " ");
        result = std::regex_replace(result, re_spaces_tabs, " ");

        // PATCH C (v33): P.S. / P.P.S. → литературная замена.
        static const std::regex re_pps(R"(\bP\.P\.S\.?)", std::regex::ECMAScript | std::regex::icase);
        static const std::regex re_ps (R"(\bP\.S\.?)",    std::regex::ECMAScript | std::regex::icase);
        result = std::regex_replace(result, re_pps, "Пост-постскриптум");
        result = std::regex_replace(result, re_ps,  "Постскриптум");

        // PATCH C (v33): маркеры нумерованных списков внутри строки.
        // Пример: "сразу:1. Продолжай" → "сразу: Продолжай".
        static const std::regex re_numbered_inline(
            R"((\s|:|^)\d{1,2}[.):](\s|$))", std::regex::ECMAScript);
        result = std::regex_replace(result, re_numbered_inline, "$1$2");

        trim(result);
    } catch (const std::regex_error& e) {
        if (g_verbose_mode.load()) {
            fprintf(stderr, "[clean_text_for_tts] regex error: %s\n", e.what());
        }
        // Fallback: ручные замены.
        result = string_replace_all(result, "**", " ");
        result = string_replace_all(result, "__", " ");
        result = string_replace_all(result, "*", " ");
        result = string_replace_all(result, "_", " ");
        result = string_replace_all(result, "~~", " ");
        result = string_replace_all(result, "[", " ");
        result = string_replace_all(result, "]", " ");
        result = string_replace_all(result, "#", " ");
        result = string_replace_all(result, "~", " ");
        result = string_replace_all(result, "`", " ");
        result = string_replace_all(result, ">", " ");
        while (result.find("  ") != std::string::npos) {
            result = string_replace_all(result, "  ", " ");
        }
        trim(result);
    }

    // Удаляем "P.S.", "P.P.S.", "PS." в начале/конце.
    {
        for (const auto& ps : {"P.P.S.", "P.S.", "PS."}) {
            std::string ps_lower = LowerCase(ps);
            std::string result_lower = LowerCase(result);
            size_t rlen = result.size();
            size_t plen = ps_lower.size();
            if (rlen >= plen) {
                std::string tail = result_lower.substr(rlen - plen);
                if (tail == ps_lower) {
                    result = result.substr(0, rlen - plen);
                    trim_spaces_only(result);
                }
            }
        }
        for (const auto& ps : {"P.P.S. ", "P.S. ", "PS. "}) {
            std::string ps_str = std::string(ps);
            std::string ps_lower = LowerCase(ps_str);
            std::string result_lower = LowerCase(result);
            if (result_lower.size() >= ps_lower.size() &&
                result_lower.substr(0, ps_lower.size()) == ps_lower) {
                result = result.substr(ps_str.size());
                trim_spaces_only(result);
            }
        }
    }

    return result;
}

// --- 15.2. Безопасное удаление фрагмента спецтокена ---
// std::isalnum(0xD0) возвращает false для кириллицы, поэтому
// фрагменты вида "id", "end" удалялись из середины русских слов.
// Проверяем границы по UTF-8.
//
// PATCH 4 (v34): функция теперь реально используется —
// см. strip_special_tokens (15.3) и send_tts_async (16.2).
// Раньше она была объявлена, но не вызывалась.
static std::string safe_remove_fragment(const std::string& text,
                                         const std::string& fragment) {
    if (fragment.empty()) return text;

    // Если во фрагменте есть не-буквенные символы (<, |, _, #),
    // удаление безопасно — он не встречается в обычном тексте.
    // Пример: <|eot_id|>, </s>, ### — сюда.
    bool has_non_alpha = false;
    for (unsigned char c : fragment) {
        if (c < 0x80 && !std::isalpha(c)) {
            has_non_alpha = true;
            break;
        }
    }
    if (has_non_alpha) {
        return string_replace_all(text, fragment, "");
    }

    // Фрагмент — чисто буквенный (USER, ASSISTANT, END, STOP).
    // Удаляем ТОЛЬКО на границах слов, чтобы не вырезать
    // подстроку из середины другого слова.
    std::string result = text;
    size_t pos = 0;
    while ((pos = result.find(fragment, pos)) != std::string::npos) {
        bool left_ok = true;
        if (pos > 0) {
            size_t char_start = pos - 1;
            while (char_start > 0 &&
                (static_cast<unsigned char>(result[char_start]) & 0xC0) == 0x80) {
                char_start--;
            }
            std::string left_char = result.substr(char_start, pos - char_start);
            left_ok = !utf8_is_word_char(left_char);
        }
        bool right_ok = true;
        if (pos + fragment.size() < result.size()) {
            std::string right_char = utf8_first_char(result.substr(pos + fragment.size()));
            right_ok = !utf8_is_word_char(right_char);
        }
        if (left_ok && right_ok) {
            result.erase(pos, fragment.size());
        } else {
            pos += fragment.size();
        }
    }
    return result;
}

// --- 15.3. Очистка от служебных токенов ---
// Все правила берутся из пресета. Плюс универсальное удаление
// любых <|...|> конструкций через regex.
//
// PATCH 4 (v34): лямбда remove_all теперь вызывает
// safe_remove_fragment вместо string_replace_all. Для стоп-строк
// с не-буквенными символами (<|eot_id|>, </s>) поведение
// не меняется, для буквенных (USER:, ASSISTANT:) — защищает
// от вырезания из середины русского текста.
//
// PATCH 8 (v34): добавлено удаление Gemma-тегов
// (<start_of_turn>, <end_of_turn> и др.). Они не подпадают
// под regex <|...|>, потому что не содержат |.
static std::string strip_special_tokens(
    const std::string& text,
    const PresetDerived& preset,
    const std::vector<std::string>& extra_stops = {}) {
    std::string result = text;

    // Убираем все непустые префиксы/суффиксы из пресета.
    // PATCH 4 (v34): safe_remove_fragment вместо string_replace_all.
    auto remove_all = [&](const std::string& s) {
        if (!s.empty()) {
            result = safe_remove_fragment(result, s);
        }
    };
    remove_all(preset.sys_prefix);
    remove_all(preset.sys_suffix);
    remove_all(preset.user_prefix);
    remove_all(preset.user_suffix);
    remove_all(preset.bot_prefix);
    remove_all(preset.bot_suffix);
    remove_all(preset.stop_sequence);

    // Дополнительные стопы (например, из --stop-words).
    // PATCH 4 (v34): тоже через safe_remove_fragment.
    for (const auto& s : extra_stops) {
        if (!s.empty()) {
            result = safe_remove_fragment(result, s);
        }
    }

    // Универсальное удаление ЛЮБЫХ спецтокенов вида <|...|>.
    try {
        static const std::regex re_special_token(
            R"(<\|[^|]*\|>)",
            std::regex::ECMAScript);
        result = std::regex_replace(result, re_special_token, "");
    } catch (const std::regex_error&) {
        result = string_replace_all(result, "<|eo_id|>", "");
        result = string_replace_all(result, "<|eot_id|>", "");
        result = string_replace_all(result, "<|start_header_id|>", "");
        result = string_replace_all(result, "<|end_header_id|>", "");
    }

    // PATCH 8 (v34): удаление Gemma-тегов. Они не подпадают
    // под regex <|...|>, потому что не содержат |.
    // <start_of_turn> и <end_of_turn> — обёртки ходов Gemma.
    // <start_of_image>, <end_of_image>, <image_soft_token> —
    // мультимодальные теги (для текстовых моделей не нужны,
    // но если модель их сгенерирует, они попадут в TTS).
    result = string_replace_all(result, "<start_of_turn>", "");
    result = string_replace_all(result, "<end_of_turn>", "");
    result = string_replace_all(result, "<start_of_image>", "");
    result = string_replace_all(result, "<end_of_image>", "");
    result = string_replace_all(result, "<image_soft_token>", "");

    // Фигурные скобки — placeholders, удаляем полностью.
    try {
        static const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
        result = std::regex_replace(result, re_curly, "");
    } catch (const std::regex_error&) {}

    // Удаляем "[описание]" в конце реплики, если это не известный статус.
    if (!result.empty() && result.back() == ']') {
        size_t open = result.rfind('[');
        if (open != std::string::npos && open > 0) {
            std::string inside = result.substr(open + 1, result.size() - open - 2);
            std::string inside_lower = LowerCase(inside);
            trim(inside_lower);
            if (preset.statuses.find(inside_lower) == preset.statuses.end()) {
                result = result.substr(0, open);
                trim_spaces_only(result);
            }
        }
    }

    // Схлопываем двойные пробелы.
    while (result.find("  ") != std::string::npos) {
        result = string_replace_all(result, "  ", " ");
    }
    while (result.find("\t\t") != std::string::npos) {
        result = string_replace_all(result, "\t\t", "\t");
    }

    return result;
}

// --- 15.4. Схлопка знаков препинания ---
// "???" → "?", "!!" → "!", "..." → ".".
static std::string collapse_punctuation_marks(const std::string& text) {
    std::string result = text;
    try {
        static const std::regex re_punct_group(
            R"(([.,!?])(?:\s*[.,!?])+)",
            std::regex::ECMAScript);

        std::string out;
        out.reserve(result.size());

        auto it_begin = std::sregex_iterator(result.begin(), result.end(), re_punct_group);
        auto it_end = std::sregex_iterator();
        size_t last_pos = 0;

        for (auto i = it_begin; i != it_end; ++i) {
            std::smatch match = *i;
            out += result.substr(last_pos, match.position() - last_pos);

            std::string group = match.str();
            bool has_q = group.find('?') != std::string::npos;
            bool has_e = group.find('!') != std::string::npos;
            bool has_d = group.find('.') != std::string::npos;

            char winner = '.';
            if (has_q)      winner = '?';
            else if (has_e) winner = '!';
            else if (has_d) winner = '.';
            else            winner = ',';

            out += winner;
            last_pos = match.position() + match.length();
        }
        out += result.substr(last_pos);
        result = std::move(out);
    } catch (const std::regex_error&) {
        while (result.find("!!") != std::string::npos)
            result = string_replace_all(result, "!!", "!");
        while (result.find("??") != std::string::npos)
            result = string_replace_all(result, "??", "?");
        while (result.find("..") != std::string::npos)
            result = string_replace_all(result, "..", ".");
    }
    return result;
}

// --- 15.5. Удаление префиксов диалога ---
// Имя пользователя и бота удаляется ТОЛЬКО в начале строки/текста.
static std::string strip_dialog_prefixes(const std::string& text,
                                          const std::string& person,
                                          const std::string& bot,
                                          const std::string& symb) {
    std::string r = text;

    // Убираем только "\nИмя: " → на перенос строки.
    for (const auto& name : {person, bot}) {
        r = string_replace_all(r, "\n" + name + symb, "\n");
        r = string_replace_all(r, "\n" + name + ":", "\n");
        r = string_replace_all(r, "\r\n" + name + symb, "\r\n");
        r = string_replace_all(r, "\r\n" + name + ":", "\r\n");
    }

    // Убираем только в самом начале текста.
    for (const auto& name : {person, bot}) {
        std::string with_symb = name + symb;
        std::string with_colon = name + ":";
        if (r.compare(0, with_symb.size(), with_symb) == 0) {
            r = r.substr(with_symb.size());
        } else if (r.compare(0, with_colon.size(), with_colon) == 0) {
            r = r.substr(with_colon.size());
        }
    }

    trim_spaces_only(r);
    return r;
}

// --- 15.6. Санитизация токена для КОНСОЛИ (PATCH B, v33) ---
// В консоль выводим честный текст, как сгенерировала модель.
// Убираем ТОЛЬКО:
//   - любые <|...|> (служебные токены);
//   - Gemma-теги (PATCH 8, v34);
//   - префикс "Имя: " в самом начале.
static std::string sanitize_for_console(const std::string& text,
                                         const std::string& person,
                                         const std::string& bot,
                                         const std::string& chat_symb) {
    std::string result = text;
    try {
        static const std::regex re_special(R"(<\|[^|]*\|>)", std::regex::ECMAScript);
        result = std::regex_replace(result, re_special, "");
    } catch (const std::regex_error&) {}

    // PATCH 8 (v34): Gemma-теги. Они не подпадают под regex
    // <|...|>, потому что не содержат |. Добавлены явно.
    result = string_replace_all(result, "<start_of_turn>", "");
    result = string_replace_all(result, "<end_of_turn>", "");
    result = string_replace_all(result, "<start_of_image>", "");
    result = string_replace_all(result, "<end_of_image>", "");
    result = string_replace_all(result, "<image_soft_token>", "");

    // Убираем name-prefix ТОЛЬКО в самом начале.
    for (const auto& name : {person, bot}) {
        std::string with_symb  = name + chat_symb;
        std::string with_colon = name + ":";
        if (result.compare(0, with_symb.size(), with_symb) == 0) {
            result = result.substr(with_symb.size());
        } else if (result.compare(0, with_colon.size(), with_colon) == 0) {
            result = result.substr(with_colon.size());
        }
    }
    return result;
}

// ============================================================================
// 16. АСИНХРОННАЯ ОТПРАВКА ТЕКСТА В XTTS
// ============================================================================
// TTS-воркер берёт запрос из очереди и отправляет его на XTTS-сервер
// по HTTP. Перед отправкой текст проходит несколько стадий очистки.

// --- 16.1. Глобальный пресет для TTS-воркера ---
static PresetDerived g_tts_preset;
static std::mutex g_tts_preset_mutex;

static void set_tts_preset(const PresetDerived& preset) {
    std::lock_guard<std::mutex> lock(g_tts_preset_mutex);
    g_tts_preset = preset;
}

// --- 16.2. Отправка текста в XTTS ---
void send_tts_async(std::string text, std::string speaker_wav,
    std::string language, std::string tts_url,
    const std::string& stop_sequence,
    const std::string& bot_suffix,
    const std::string& user_suffix,
    const std::string& bot_prefix,
    const std::string& user_prefix,
    const std::string& chat_symb,
    const std::string& person_name,
    const std::string& bot_name) {
    if (text.empty()) return;

    // Базовая очистка: скобки, markdown, HTML, эмодзи.
    text = clean_text_for_tts(text);
    if (text.empty()) return;

    // Универсальное удаление любых <|...|> из текста перед TTS.
    try {
        static const std::regex re_special_token(R"(<\|[^|]*\|>)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_special_token, "");
    } catch (const std::regex_error&) {}

    if (text.empty()) return;

    // Защита паттернов: email, IP, время, дата, проценты.
    std::vector<std::pair<std::string, std::string>> protected_patterns;

    thread_local const std::regex re_email(
        R"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
        std::regex::ECMAScript);
    thread_local const std::regex re_ip(
        R"(\b(?:\d{1,3}\.){3}\d{1,3}\b)",
        std::regex::ECMAScript);
    thread_local const std::regex re_phone(
        R"(\+?\d{1,3}[\s\-]?\(?\d{3,5}\)?[\s\-]?\d{2,3}[\s\-]?\d{2,3})",
        std::regex::ECMAScript);
    thread_local const std::regex re_time(
        R"(\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?\b)",
        std::regex::ECMAScript);
    thread_local const std::regex re_date_dots(
        R"(\b(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\.(\d{4})\b)",
        std::regex::ECMAScript);
    thread_local const std::regex re_decimal(
        R"(\b\d+[.,]\d+\b(?![\w-]))",
        std::regex::ECMAScript);
    thread_local const std::regex re_percent(
        R"(\b\d+(?:[.,]\d+)?\s*%)",
        std::regex::ECMAScript);
    thread_local const std::regex re_currency(
        R"(\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*[$€£¥₽]|\b[$€£¥₽]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)",
        std::regex::ECMAScript);
    thread_local const std::regex re_abbrev(
        R"(\b(?:т\.\s?д|т\.\s?е|т\.\s?п|т\.\s?к|др|г|гг|в)\b\.)",
        std::regex::ECMAScript);

    auto protect_pattern = [&](const std::regex& re, const std::string& prefix) {
        try {
            std::string processed;
            auto it_begin = std::sregex_iterator(text.begin(), text.end(), re);
            auto it_end = std::sregex_iterator();
            size_t last_pos = 0;
            for (auto i = it_begin; i != it_end; ++i) {
                std::smatch match = *i;
                processed += text.substr(last_pos, match.position() - last_pos);
                std::string marker = "@@" + prefix +
                    std::to_string(protected_patterns.size()) + "@@";
                protected_patterns.emplace_back(marker, match.str());
                processed += marker;
                last_pos = match.position() + match.length();
            }
            processed += text.substr(last_pos);
            text = processed;
        } catch (const std::regex_error&) {}
    };
    protect_pattern(re_email, "EMAIL");
    protect_pattern(re_ip, "IP");
    protect_pattern(re_time, "TIME");
    protect_pattern(re_date_dots, "DATE");
    protect_pattern(re_abbrev, "ABBR");
    protect_pattern(re_decimal, "DEC");
    protect_pattern(re_percent, "PCT");
    protect_pattern(re_currency, "CUR");
    protect_pattern(re_phone, "PHONE");

    // Пробел после [.!?] перед буквой.
    try {
        thread_local const std::regex re_space_after_punct(
            R"(([.!?])([A-Za-zА-Яа-яЁё]))", std::regex::ECMAScript);
        text = std::regex_replace(text, re_space_after_punct, "$1 $2");
    } catch (const std::regex_error&) {}

    // HTML-теги → пробел.
    try {
        thread_local const std::regex re_html_tag(R"(<[^>]*>)", std::regex::ECMAScript);
        text = std::regex_replace(text, re_html_tag, " ");
    } catch (const std::regex_error&) {
        text = string_replace_all(text, "<", " ");
        text = string_replace_all(text, ">", " ");
    }
    text = string_replace_all(text, "&nbsp;", " ");
    text = string_replace_all(text, "&amp;", "&");
    text = string_replace_all(text, "&lt;", "<");
    text = string_replace_all(text, "&gt;", ">");
    text = string_replace_all(text, "&quot;", "\"");
    text = string_replace_all(text, "&#39;", "'");
    text = string_replace_all(text, "&apos;", "'");
    text = string_replace_all(text, "&mdash;", "-");
    text = string_replace_all(text, "&ndash;", "-");
    text = string_replace_all(text, "&hellip;", "...");

    // Типографские Unicode → ASCII.
    text = string_replace_all(text, "\xE2\x80\x9C", "\"");
    text = string_replace_all(text, "\xE2\x80\x9D", "\"");
    text = string_replace_all(text, "\xE2\x80\x98", "'");
    text = string_replace_all(text, "\xE2\x80\x99", "'");
    text = string_replace_all(text, "\xE2\x80\x93", "-");
    text = string_replace_all(text, "\xE2\x80\x94", "-");
    text = string_replace_all(text, "\xC2\xA0", " ");
    text = string_replace_all(text, "\xE2\x80\xA6", "...");

    // Эмодзи → удалить.
    {
        std::string clean;
        clean.reserve(text.size());
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            size_t step = 1;
            uint32_t cp = c;
            if ((c & 0xE0) == 0xC0) {
                step = 2;
                if (i + 1 < text.size()) {
                    cp = ((c & 0x1F) << 6) |
                         (static_cast<unsigned char>(text[i + 1]) & 0x3F);
                }
            } else if ((c & 0xF0) == 0xE0) {
                step = 3;
                if (i + 2 < text.size()) {
                    cp = ((c & 0x0F) << 12) |
                         ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                         (static_cast<unsigned char>(text[i + 2]) & 0x3F);
                }
            } else if ((c & 0xF8) == 0xF0) {
                step = 4;
                if (i + 3 < text.size()) {
                    cp = ((c & 0x07) << 18) |
                         ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                         ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                         (static_cast<unsigned char>(text[i + 3]) & 0x3F);
                }
            }
            bool is_emoji =
                (cp >= 0x1F300 && cp <= 0x1FAFF) ||
                (cp >= 0x2600 && cp <= 0x27BF) ||
                (cp >= 0x1F000 && cp <= 0x1F2FF) ||
                (cp >= 0x1F1E6 && cp <= 0x1F1FF) ||
                (cp == 0x200D) ||
                (cp >= 0xFE00 && cp <= 0xFE0F);
            if (!is_emoji) {
                clean.append(text, i, step);
            }
            i += step;
        }
        text = std::move(clean);
    }
    text = string_replace_all(text, ":)", " ");
    text = string_replace_all(text, ":-)", " ");
    text = string_replace_all(text, ":(", " ");
    text = string_replace_all(text, ":-(", " ");
    text = string_replace_all(text, ";)", " ");
    text = string_replace_all(text, ";-)", " ");
    text = string_replace_all(text, ":D", " ");
    text = string_replace_all(text, ":P", " ");
    text = string_replace_all(text, "=)", " ");
    text = string_replace_all(text, "=(", " ");

    // Удаление кавычек с сохранением сокращений.
    try {
        std::vector<std::pair<std::string, std::string>> saved_contractions;
        thread_local const std::regex re_contractions("\\b\\w+'\\w+\\b", std::regex::ECMAScript);
        {
            std::string protected_text;
            auto it_begin = std::sregex_iterator(text.begin(), text.end(), re_contractions);
            auto it_end = std::sregex_iterator();
            size_t last_pos = 0;
            for (auto i = it_begin; i != it_end; ++i) {
                std::smatch match = *i;
                protected_text += text.substr(last_pos, match.position() - last_pos);
                std::string marker = "@@CONTR" +
                    std::to_string(saved_contractions.size()) + "@@";
                saved_contractions.emplace_back(marker, match.str());
                protected_text += marker;
                last_pos = match.position() + match.length();
            }
            protected_text += text.substr(last_pos);
            text = protected_text;
        }
        thread_local const std::regex re_quotes_double("\"([^\"]*)\"", std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_double, "$1");
        thread_local const std::regex re_quotes_single("'([^']*)'", std::regex::ECMAScript);
        text = std::regex_replace(text, re_quotes_single, "$1");
        for (const auto& p : saved_contractions) {
            text = string_replace_all(text, p.first, p.second);
        }
    } catch (const std::regex_error&) {
        text = string_replace_all(text, "\"", "");
        text = string_replace_all(text, "'", "");
    }

    // Ссылки → произносимый текст.
    try {
        thread_local const std::regex re_link_md(
            R"(\[([^\]]*)\]\(([^)\s]+)\))", std::regex::ECMAScript);
        {
            std::string result;
            auto it_begin = std::sregex_iterator(text.begin(), text.end(), re_link_md);
            auto it_end = std::sregex_iterator();
            size_t last_pos = 0;
            for (auto i = it_begin; i != it_end; ++i) {
                std::smatch match = *i;
                result += text.substr(last_pos, match.position() - last_pos);
                std::string link_text = match[1].str();
                std::string url = match[2].str();
                if (link_text.length() > 2 &&
                    link_text != "ссылка" && link_text != "link") {
                    result += link_text + ". ";
                } else {
                    result += url_to_speech(url) + ". ";
                }
                last_pos = match.position() + match.length();
            }
            result += text.substr(last_pos);
            text = result;
        }
        thread_local const std::regex re_bare_url(
            R"(https?://[^\s<>]+|www\.[^\s<>]+)", std::regex::ECMAScript);
        {
            std::string result;
            auto it_begin = std::sregex_iterator(text.begin(), text.end(), re_bare_url);
            auto it_end = std::sregex_iterator();
            size_t last_pos = 0;
            for (auto i = it_begin; i != it_end; ++i) {
                std::smatch match = *i;
                result += text.substr(last_pos, match.position() - last_pos);
                result += url_to_speech(match.str()) + ". ";
                last_pos = match.position() + match.length();
            }
            result += text.substr(last_pos);
            text = result;
        }
    } catch (const std::regex_error&) {
        text = string_replace_all(text, "<", " ");
        text = string_replace_all(text, ">", " ");
    }

    // Фигурные скобки → удалить.
    try {
        thread_local const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
        bool changed = true;
        int iteration = 0;
        while (changed && iteration < 10) {
            changed = false;
            iteration++;
            std::string t1 = std::regex_replace(text, re_curly, " ");
            if (t1 != text) { text.swap(t1); changed = true; }
        }
    } catch (const std::regex_error&) {
        text = string_replace_all(text, "{", " ");
        text = string_replace_all(text, "}", " ");
    }

    // Маркеры списков в начале строки → удалить.
    {
        std::string result;
        result.reserve(text.size());
        std::istringstream iss(text);
        std::string line;
        bool first_line = true;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();

            size_t p = 0;
            while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;

            size_t marker_end = p;
            bool is_marker = false;

            if (p < line.size()) {
                unsigned char c0 = static_cast<unsigned char>(line[p]);

                if ((line[p] == '-' || line[p] == '*' || line[p] == '+') &&
                    p + 1 < line.size() &&
                    (line[p + 1] == ' ' || line[p + 1] == '\t')) {
                    marker_end = p + 2;
                    while (marker_end < line.size() &&
                           (line[marker_end] == ' ' || line[marker_end] == '\t'))
                        marker_end++;
                    is_marker = true;
                }
                else if (std::isdigit(c0)) {
                    size_t q = p;
                    while (q < line.size() &&
                           std::isdigit(static_cast<unsigned char>(line[q]))) q++;
                    if (q < line.size() &&
                        (line[q] == '.' || line[q] == ')' || line[q] == ':')) {
                        size_t r = q + 1;
                        while (r < line.size() &&
                               (line[r] == ' ' || line[r] == '\t')) r++;
                        if (r > q + 1) {
                            marker_end = r;
                            is_marker = true;
                        }
                    }
                }
                else if (c0 >= 0x80) {
                    size_t char_len = 1;
                    if ((c0 & 0xE0) == 0xC0) char_len = 2;
                    else if ((c0 & 0xF0) == 0xE0) char_len = 3;
                    else if ((c0 & 0xF8) == 0xF0) char_len = 4;

                    if (p + char_len < line.size() &&
                        (line[p + char_len] == '.' ||
                         line[p + char_len] == ')' ||
                         line[p + char_len] == ':')) {
                        size_t r = p + char_len + 1;
                        size_t spaces = 0;
                        while (r < line.size() &&
                               (line[r] == ' ' || line[r] == '\t')) {
                            r++;
                            spaces++;
                        }
                        if (spaces > 0 || r >= line.size()) {
                            bool single_letter = true;
                            if (p + char_len < line.size()) {
                                if (std::isalpha(static_cast<unsigned char>(line[p + char_len]))) {
                                    single_letter = false;
                                }
                            }
                            if (single_letter) {
                                marker_end = r;
                                is_marker = true;
                            }
                        }
                    }
                }
            }

            if (is_marker) {
                line = line.substr(marker_end);
            }

            if (!first_line) result += "\n";
            result += line;
            first_line = false;
        }
        text = result;
    }

    // Маркеры списков, прилипшие к предыдущему предложению.
    try {
        std::string result;
        result.reserve(text.size());
        size_t i = 0;
        const size_t n = text.size();
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            if (std::isdigit(c)) {
                size_t digit_start = i;
                size_t digit_end = i;
                while (digit_end < n && std::isdigit(static_cast<unsigned char>(text[digit_end]))) {
                    digit_end++;
                }
                size_t n_digits = digit_end - digit_start;

                bool left_ok = (digit_start == 0);
                if (!left_ok) {
                    size_t prev = digit_start;
                    while (prev > 0 &&
                           (text[prev - 1] == ' ' || text[prev - 1] == '\t')) {
                        prev--;
                    }
                    if (prev == 0) {
                        left_ok = true;
                    } else {
                        unsigned char p = static_cast<unsigned char>(text[prev - 1]);
                        if (p == '?' || p == '!' || p == '.' || p == '\n' ||
                            p == '\r' || p == ':') {
                            left_ok = true;
                        }
                    }
                }

                bool middle_ok = false;
                if (digit_end < n) {
                    unsigned char m = static_cast<unsigned char>(text[digit_end]);
                    if (m == '.' || m == ')' || m == ':') {
                        middle_ok = true;
                    }
                }

                bool right_ok = false;
                size_t after_punct = digit_end + 1;
                if (middle_ok && after_punct < n) {
                    size_t spaces = 0;
                    size_t j = after_punct;
                    while (j < n && (text[j] == ' ' || text[j] == '\t')) {
                        j++;
                        spaces++;
                    }
                    if (spaces > 0 && j < n) {
                        unsigned char next = static_cast<unsigned char>(text[j]);
                        if (std::isalpha(next)) {
                            right_ok = true;
                        } else if (next >= 0xC0 && next <= 0xF7) {
                            right_ok = true;
                        }
                    }
                }

                if (n_digits >= 1 && n_digits <= 2 && left_ok && middle_ok && right_ok) {
                    result += ' ';
                    i = after_punct;
                    while (i < n && (text[i] == ' ' || text[i] == '\t')) {
                        i++;
                    }
                    continue;
                }

                for (size_t k = digit_start; k < digit_end; ++k) {
                    result += text[k];
                }
                i = digit_end;
                continue;
            }

            result += text[i];
            i++;
        }
        text = std::move(result);
    } catch (const std::exception&) {
        if (g_verbose_mode.load()) {
            fprintf(stderr, "[13.5.9b] ошибка обработки маркеров\n");
        }
    }

    // Финальная очистка спецсимволов.
    text = string_replace_all(text, "|", " ");
    text = string_replace_all(text, "\\", " ");

    // Пробелы перед знаками препинания.
    try {
        thread_local const std::regex re_space_before_excl(R"([ \t]+(!))", std::regex::ECMAScript);
        thread_local const std::regex re_space_before_ques(R"([ \t]+(\?))", std::regex::ECMAScript);
        thread_local const std::regex re_space_before_dot(R"([ \t]+(\.))", std::regex::ECMAScript);
        text = std::regex_replace(text, re_space_before_excl, "$1");
        text = std::regex_replace(text, re_space_before_ques, "$1");
        text = std::regex_replace(text, re_space_before_dot, "$1");
    } catch (const std::regex_error&) {
        text = string_replace_all(text, " !", "!");
        text = string_replace_all(text, " ?", "?");
        text = string_replace_all(text, " .", ".");
    }

    // Схлопка знаков по приоритету.
    text = collapse_punctuation_marks(text);

    // Финальная нормализация запятых и пробелов.
    while (text.find(",,") != std::string::npos) {
        text = string_replace_all(text, ",,", ",");
    }
    text = string_replace_all(text, ". ,", ". ");
    text = string_replace_all(text, "! ,", "! ");
    text = string_replace_all(text, "? ,", "? ");
    while (text.find(", ,") != std::string::npos) {
        text = string_replace_all(text, ", ,", ", ");
    }
    while (text.find("  ") != std::string::npos) {
        text = string_replace_all(text, "  ", " ");
    }
    trim_spaces_only(text);
    if (text.empty()) return;

    // Восстановление защищённых паттернов.
    for (const auto& p : protected_patterns) {
        text = string_replace_all(text, p.first, p.second);
    }

    // Удаление префикса имени бота и пользователя.
    text = strip_dialog_prefixes(text, person_name, bot_name, chat_symb);
    if (text.empty()) return;

    // Удаление стоп-последовательностей.
    //
    // PATCH 4 (v34): safe_remove_fragment вместо string_replace_all.
    // Для стоп-строк с не-буквенными символами (<|eot_id|>, </s>)
    // поведение не меняется, для буквенных (USER:, ASSISTANT:) —
    // защищает от вырезания из середины русского текста.
    if (!stop_sequence.empty()) text = safe_remove_fragment(text, stop_sequence);
    if (!bot_suffix.empty())    text = safe_remove_fragment(text, bot_suffix);
    if (!user_suffix.empty())   text = safe_remove_fragment(text, user_suffix);
    if (!bot_prefix.empty())    text = safe_remove_fragment(text, bot_prefix);
    if (!user_prefix.empty())   text = safe_remove_fragment(text, user_prefix);
    trim_spaces_only(text);
    if (text.empty()) return;

    // Удаление висящей запятой в конце.
    if (!text.empty() && text.back() == ',') text.pop_back();
    text = string_replace_all(text, ",.", ".");
    text = string_replace_all(text, ",!", "!");
    text = string_replace_all(text, ",?", "?");
    trim_spaces_only(text);
    if (text.empty()) return;

    // Фильтрация бессмысленных остатков.
    if (text == "eo" || text == "Eo" || text == "t_id" || text == "id" ||
        text == "eot" || text == speaker_wav) {
        return;
    }

    // Проверка баланса скобок.
    // PATCH C (v33): вместо отбрасывания фрагмента с незакрытой
    // скобкой — дополняем её закрывающей.
    {
        int open_paren = 0, open_bracket = 0, open_curly = 0;
        for (char c : text) {
            if      (c == '(') open_paren++;
            else if (c == ')') open_paren--;
            else if (c == '[') open_bracket++;
            else if (c == ']') open_bracket--;
            else if (c == '{') open_curly++;
            else if (c == '}') open_curly--;
        }
        if (open_paren > 0 || open_bracket > 0 || open_curly > 0) {
            if (g_verbose_mode.load()) {
                fprintf(stderr,
                    "[TTS] Дополняю незакрытые скобки: (x%d [x%d {x%d — '%s'\n",
                    open_paren, open_bracket, open_curly, text.c_str());
            }
            for (int i = 0; i < open_paren;   ++i) text += ')';
            for (int i = 0; i < open_bracket; ++i) text += ']';
            for (int i = 0; i < open_curly;   ++i) text += '}';
        }
    }

    // Формирование JSON и отправка в XTTS.
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
                    if (c >= 32 && c != 127) result += static_cast<char>(c);
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
        curl_easy_setopt(http_handle, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(http_handle, CURLOPT_CONNECTTIMEOUT, 15L);
        curl_easy_setopt(http_handle, CURLOPT_XFERINFOFUNCTION, progress_callback);
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
        if (res != CURLE_OK && g_verbose_mode.load()) {
            fprintf(stderr, "[TTS] cURL ошибка: %s\n", curl_easy_strerror(res));
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(http_handle);
    }
}


// ============================================================================
// 17. ОЧЕРЕДЬ TTS С ВОРКЕРОМ
// ============================================================================
// Воркер читает очередь и отправляет запросы в XTTS. Очередь
// нужна, чтобы не блокировать основной поток генерации.

static void enqueue_tts(TtsRequest req) {
    if (g_cancel_tts_requests.load()) return;
    {
        std::lock_guard<std::mutex> lock(g_tts_queue_mutex);
        g_tts_queue.push_back(std::move(req));
    }
    g_tts_queue_cv.notify_one();
}

static void clear_tts_queue() {
    std::lock_guard<std::mutex> lock(g_tts_queue_mutex);
    g_tts_queue.clear();
}

// PATCH 3 (v34): race condition починен.
//
// Раньше условие выхода из цикла было:
//     while (g_tts_worker_running.load() || !g_tts_queue.empty())
// Второе слагаемое (!g_tts_queue.empty()) читало очередь БЕЗ
// мьютекса — это data race с enqueue_tts/clear_tts_queue.
// На практике «работало», но формально UB.
//
// Теперь условие проверяется под мьютексом:
//     - внутри wait_for(lock, ...) лямбда возвращает
//       «есть работа или пора выходить»;
//     - если очередь пуста и running == false → break;
//     - если очередь пуста, но running == true → continue
//       (ждём следующего пробуждения);
//     - pop_front защищён тем же мьютексом, что и empty().
static void tts_worker_func() {
    while (true) {
        TtsRequest req;
        {
            std::unique_lock<std::mutex> lock(g_tts_queue_mutex);
            g_tts_queue_cv.wait_for(lock, std::chrono::milliseconds(100), [] {
                return !g_tts_queue.empty() || !g_tts_worker_running.load();
            });
            if (g_tts_queue.empty()) {
                if (!g_tts_worker_running.load()) break;
                continue;
            }
            req = std::move(g_tts_queue.front());
            g_tts_queue.pop_front();
        }
        if (g_cancel_tts_requests.load()) continue;
        send_tts_async(req.text, req.voice, req.language, req.url,
            req.stop_seq, req.bot_sfx, req.user_sfx, req.bot_pfx, req.user_pfx,
            req.chat_symb, req.person, req.bot);
    }
}


// ============================================================================
// 18. ФУНКЦИИ ОТРИСОВКИ КОНСОЛИ
// ============================================================================

// --- 18.1. Печать статуса с цветом ---
static void print_status(const std::string& status, const std::string& color) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    printf("%s%s%s", color.c_str(), status.c_str(), C_RESET);
    fflush(stdout);
}

// --- 18.2. Паддинг имени до 4 символов ---
static std::string make_padded_name(const std::string& name) {
    std::string padded = name;
    while (utf8_length(padded) < 4) padded += " ";
    return padded;
}

// --- 18.3. Пустая строка перед следующей парой реплик ---
static void begin_new_pair() {
    if (g_need_blank_before_next) {
        std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
        printf("\n");
        fflush(stdout);
    }
    g_need_blank_before_next = true;
}

// --- 18.4. Печать реплики с именем и статусом ---
static void print_replica(const std::string& color, const std::string& name,
                          const std::string& text, const std::string& status) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    std::string padded = make_padded_name(name);
    printf("%s%s%s: %s", color.c_str(), padded.c_str(), C_RESET, text.c_str());
    if (!status.empty()) {
        printf(" ");
        printf("%s[%s]%s", C_CMD, status.c_str(), C_RESET);
    }
    printf("\n");
    fflush(stdout);
}

// --- 18.5. Печать префикса бота без перевода строки ---
// Используется при потоковой генерации: печатаем "Эмма: " один
// раз, потом токены подряд.
static void print_bot_prefix(const std::string& bot_name) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    std::string padded = make_padded_name(bot_name);
    printf("%s%s%s: ", C_BOT, padded.c_str(), C_RESET);
    fflush(stdout);
}

// --- 18.6. Обновление заголовка окна ---
// Кэширует последнее значение, чтобы не дёргать WinAPI каждый раз.
static void update_console_title(const std::string& bot_name,
                                   const std::string& status) {
    std::string title = "Talk-LLaMA | " + bot_name + " | " + status;
    {
        std::lock_guard<std::mutex> lock(g_console_title_mutex);
        if (title == g_last_console_title) return;
        g_last_console_title = title;
    }
#ifdef _WIN32
    std::wstring wtitle = console::UTF8toUTF16(title);
    SetConsoleTitleW(wtitle.c_str());
#else
    printf("\033]0;%s\007", title.c_str());
    fflush(stdout);
#endif
}

// --- 18.7. Добавить статус в конец текущей строки ---
// Используется для [tts], [stop], [int].
static void append_status_to_current_line(const std::string& status,
                                            const std::string& color) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    printf(" %s[%s]%s\n", color.c_str(), status.c_str(), C_RESET);
    fflush(stdout);
    g_need_blank_before_next = true;
}


// ============================================================================
// 19. АУДИО-ПОТОК
// ============================================================================
// Читает микрофон, детектирует речь энергетическим VAD, транскрибирует
// Whisper, отправляет текст в основной поток через g_pending_llm_request.
//
// Важная особенность: семафор XTTS (xtts_play_allowed.txt) падает в 0
// СРАЗУ при первом VAD-хите, а не через 2 кадра. Это даёт мгновенную
// остановку TTS при barge-in.

void audio_input_thread_func(whisper_context* ctx_wsp,
                             const whisper_params& params, audio_async& audio_ref,
                             const std::string& person_name,
                             const std::string& chat_symb) {
    (void)chat_symb;
    (void)person_name;
    g_audio_thread_running.store(true);

    // --- 19.1. Локальное состояние VAD ---
    bool speech_active = false;
    float speech_start_ms = 0.0f;
    float last_speech_end_ms = get_current_time_ms();
    int consecutive_barge_in = 0;

    float silence_timeout_ms = params.vad_last_ms;
    if (silence_timeout_ms < 1500.0f) silence_timeout_ms = 1500.0f;

    const float max_segment_ms = 30000.0f;

    bool tts_allowed_prev = true;
    {
        std::string dummy;
        allow_xtts_file(dummy, 1);
    }

    if (g_verbose_mode.load()) {
        fprintf(stderr, "[AudioInput] Поток запущен: silence=%.0f мс, "
                "max_segment=%.0f мс, soft_limit=%d токенов, VAD=энергетический, "
                "beam_size=%d\n",
                silence_timeout_ms, max_segment_ms,
                g_soft_limit_tokens.load(), params.beam_size);
    }

    std::vector<float> pcmf32_cur;
    pcmf32_cur.reserve(2000 * WHISPER_SAMPLE_RATE / 1000);

    while (!g_shutting_down.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (g_shutting_down.load()) break;

        // --- 19.2. Принудительный сброс VAD (после команды «стоп») ---
        if (g_force_vad_reset.load()) {
            speech_active = false;
            speech_start_ms = 0.0f;
            last_speech_end_ms = get_current_time_ms();
            {
                std::lock_guard<std::mutex> lock(g_display_mutex);
                g_display_text.clear();
            }
            {
                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                g_accumulated_text.clear();
            }
            g_force_vad_reset.store(false);
            if (g_verbose_mode.load()) {
                fprintf(stderr, "[AudioInput] VAD сброшен по stop\n");
            }
            continue;
        }

        // --- 19.3. Если идёт сброс контекста — ждём ---
        if (g_reset_in_progress.load()) {
            std::unique_lock<std::mutex> lock(g_reset_mutex);
            g_reset_cv.wait(lock, [] { return !g_reset_in_progress.load(); });
            continue;
        }

        // --- 19.4. Push-to-talk ---
        if (params.push_to_talk) {
            std::string hk;
            {
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                hk = g_hotkey_pressed;
            }
            if (hk != "Alt") {
                audio_ref.clear();
                continue;
            }
        }

        float current_time = get_current_time_ms();

        pcmf32_cur.clear();
        audio_ref.get(2000, pcmf32_cur);
        if (pcmf32_cur.empty()) continue;

        // --- 19.5. Управление семафором XTTS ---
        bool should_allow_tts = !speech_active &&
                                g_interrupt_reason.load() == InterruptReason::NONE &&
                                !g_reset_in_progress.load();
        if (should_allow_tts != tts_allowed_prev) {
            std::string dummy;
            allow_xtts_file(dummy, should_allow_tts ? 1 : 0);
            tts_allowed_prev = should_allow_tts;
        }

        BotState state = g_bot_state.load();

        // --- 19.6. Энергетический VAD ---
        int vad_result = vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE,
            static_cast<int>(params.vad_last_ms), params.vad_thold,
            params.freq_thold, params.print_energy, params.vad_start_thold);

        // --- 19.7. Мгновенная остановка TTS при VAD-хите ---
        if (vad_result == 1) {
            std::string dummy;
            allow_xtts_file(dummy, 0);
        }

        // --- 19.8. Barge-in во время генерации ---
        if (state == BotState::GENERATING) {
            if (vad_result == 1) {
                consecutive_barge_in++;
                if (consecutive_barge_in >= 2) {
                    g_cancel_tts_requests.store(true);
                    g_interrupt_reason.store(InterruptReason::VAD_SPEECH);
                    g_interrupt_processed.store(true);
                    g_bot_state.store(BotState::INTERRUPTED);
                    clear_tts_queue();
                    consecutive_barge_in = 0;
                    log_line("Barge-in: 2 VAD hits");
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Barge-in: речь обнаружена!\n");
                    }
                }
            } else {
                consecutive_barge_in = 0;
            }
            continue;
        }

        if (state != BotState::IDLE) continue;

        // --- 19.9. Начало речи ---
        if (vad_result == 1 && !speech_active) {
            speech_active = true;
            speech_start_ms = current_time;

            std::string dummy;
            allow_xtts_file(dummy, 0);

            // PATCH 10 (v34): разогрев Whisper в момент старта речи.
            //
            // WHY: первый вызов transcribe() в сессии всегда медленнее
            // последующих — Whisper инициализирует буферы, читает
            // prompt_tokens, прогревает модель. Если делать это в
            // момент окончания речи, пользователь слышит задержку
            // перед ответом бота. Разогрев в момент НАЧАЛА речи
            // прячет эту задержку — к тому моменту, когда пользователь
            // закончит говорить, Whisper уже прогрет.
            //
            // Результат разогрева ИГНОРИРУЕТСЯ — мы не добавляем его
            // в g_accumulated_text. Это просто «прогрев» модели.
            if (!pcmf32_cur.empty()) {
                float warmup_prob = 0.0f;
                int64_t warmup_t_ms = 0;
                (void)transcribe(ctx_wsp, params, pcmf32_cur,
                                 g_whisper_prompt, warmup_prob, warmup_t_ms,
                                 params.translate);
                if (g_verbose_mode.load()) {
                    fprintf(stderr, "[AudioInput] Whisper разогрет (%.0f мс)\n",
                            static_cast<double>(warmup_t_ms));
                }
            }

            log_line("VAD: speech started");
            if (g_verbose_mode.load()) {
                fprintf(stderr, "[AudioInput] Начало речи (%.3f)\n", current_time);
            }
        }

        // --- 19.10. Речь идёт слишком долго — аварийное сегментирование ---
        if (speech_active && (vad_result == 0 || vad_result == 1)) {
            float speech_duration = current_time - speech_start_ms;

            if (speech_duration > max_segment_ms) {
                std::vector<float> pcmf32_seg;
                audio_ref.get(params.voice_ms, pcmf32_seg);
                if (!pcmf32_seg.empty()) {
                    float prob = 0.0f;
                    int64_t t_ms = 0;
                    std::string seg_text = transcribe(ctx_wsp, params,
                        pcmf32_seg, g_whisper_prompt, prob, t_ms,
                        params.translate);
                    trim(seg_text);
                    if (!seg_text.empty() && !is_hallucination(seg_text) &&
                        prob > 0.5f) {
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
                            note_transcription(seg_text);
                        }
                    }
                }
                speech_start_ms = current_time;
            }
        }

        // --- 19.11. Конец речи ---
        if (vad_result == 2 && speech_active) {
            speech_active = false;
            float speech_len = current_time - speech_start_ms;
            last_speech_end_ms = current_time;

            if (speech_len > 0.5f) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                std::vector<float> pcmf32_full;
                audio_ref.get(params.voice_ms, pcmf32_full);

                // Добавляем 300 мс тишины в конец — Whisper лучше
                // распознаёт конец фразы.
                const int padding = (WHISPER_SAMPLE_RATE * 300) / 1000;
                pcmf32_full.insert(pcmf32_full.end(), padding, 0.0f);

                if (!pcmf32_full.empty()) {
                    float prob = 0.0f;
                    int64_t t_ms = 0;
                    std::string text = transcribe(ctx_wsp, params, pcmf32_full,
                        g_whisper_prompt, prob, t_ms, params.translate);
                    trim(text);
                    if (!text.empty() && !is_hallucination(text) && prob > 0.5f) {
                        if (speech_len > 2.0f && utf8_length(text) < 5) {
                            if (g_verbose_mode.load()) {
                                fprintf(stderr, "[Whisper] Отброшено: %.1f сек речи, "
                                                "транскрипция '%s' (%d символов)\n",
                                        speech_len, text.c_str(), utf8_length(text));
                            }
                            text = "";
                        }
                    }
                    if (!text.empty()) {
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
                            {
                                std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                                if (!g_accumulated_text.empty()) g_accumulated_text += " ";
                                g_accumulated_text += text;
                                note_transcription(text);
                            }
                            {
                                std::lock_guard<std::mutex> lock(g_display_mutex);
                                g_display_text = text;
                            }
                            if (g_verbose_mode.load()) {
                                fprintf(stderr, "[AudioInput] Сегмент: '%s'\n",
                                        text.c_str());
                            }
                        }
                    }
                }
            }
            speech_start_ms = 0;
        }

        // --- 19.12. Тишина — проверяем таймаут ---
        if (!speech_active) {
            float silence_ms = (current_time - last_speech_end_ms) * 1000.0f;
            if (silence_ms > silence_timeout_ms) {
                std::string acc;
                {
                    std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                    acc = g_accumulated_text;
                }

                bool should_continue = false;
                if (params.auto_continue && !g_command_in_progress.load()) {
                    std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                    float time_since_interrupt =
                        (current_time - g_saved_timestamp) * 1000.0f;
                    if (g_saved_was_interrupted &&
                        acc.empty() &&
                        time_since_interrupt < params.continue_max_ms) {
                        should_continue = true;
                    }
                }

                if (should_continue) {
                    g_pending_continue_request.store(true);
                    last_speech_end_ms = current_time;
                    {
                        std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                        g_saved_was_interrupted = false;
                    }
                    log_line("Auto-continue: triggered");
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Автопродолжение\n");
                    }
                }
                else if (!acc.empty() && !g_pending_llm_request.load()) {
                    {
                        std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
                        g_pending_llm_text = acc;
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        g_accumulated_text.clear();
                    }
                    g_pending_llm_request.store(true);
                    last_speech_end_ms = current_time;
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Отправка %zu символов в LLaMA\n",
                                acc.size());
                    }
                }
            }

            // --- 19.13. Мягкий лимит токенов ---
            {
                std::string snapshot;
                {
                    std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                    if (g_accumulated_text.empty() ||
                        g_pending_llm_request.load()) {
                        snapshot.clear();
                    } else {
                        snapshot = g_accumulated_text;
                    }
                }
                llama_context* local_ctx = nullptr;
                if (!snapshot.empty() && !g_reset_in_progress.load()) {
                    local_ctx = get_llama_ctx();
                }
                if (local_ctx) {
                    std::vector<llama_token> toks =
                        llama_tokenize(local_ctx, snapshot, false);
                    if (static_cast<int>(toks.size()) >= g_soft_limit_tokens.load()) {
                        std::string acc_to_send;
                        {
                            std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                            acc_to_send = g_accumulated_text;
                            g_accumulated_text.clear();
                        }
                        {
                            std::lock_guard<std::mutex> plock(g_pending_llm_mutex);
                            g_pending_llm_text = acc_to_send;
                        }
                        g_pending_llm_request.store(true);
                        last_speech_end_ms = current_time;
                    }
                }
            }
        }
    }

    g_audio_thread_running.store(false);
    if (g_verbose_mode.load()) {
        fprintf(stderr, "[AudioInput] Поток остановлен\n");
    }
}
// ============================================================================
// 20. ПОТОКИ ВВОДА И ГОРЯЧИХ КЛАВИШ
// ============================================================================

// --- 20.1. Чтение клавиатуры ---
// Читает строки через console::readline. При EOF (Ctrl+D/Ctrl+Z)
// ставит g_shutting_down и завершается.
void input_thread_func() {
    std::string line;
    std::string buffer;
    bool found_another_line = true;
    while (keyboard_input_running.load()) {
        buffer.clear();
        do {
            if (!keyboard_input_running.load()) break;
            found_another_line = console::readline(line, true);
            if (!found_another_line) break;
            buffer += line;
            if (!line.empty() && line.back() == '\n') break;
        } while (found_another_line);

        if (console::is_eof()) {
            g_shutting_down.store(true);
            keyboard_input_running.store(false);
            break;
        }

        trim(buffer);
        if (!buffer.empty()) {
            std::lock_guard<std::mutex> lock(input_mutex);
            input_queue.push(buffer);
        }
    }
}

// --- 20.2. Проверка фокуса консольного окна (Windows) ---
bool IsConsoleWindowFocused() {
#ifdef _WIN32
    HWND console_window = GetConsoleWindow();
    if (console_window == NULL) return false;
    HWND foreground_window = GetForegroundWindow();
    if (foreground_window == NULL) return false;
    return (console_window == foreground_window);
#else
    return true;
#endif
}

// --- 20.3. Горячие клавиши ---
// Опрашивает состояние клавиш каждые 100 мс. Обрабатывает:
//   Ctrl+Space  — Stop
//   Ctrl+Right  — Regenerate
//   Ctrl+Delete — Delete
//   Ctrl+R      — Reset
//   Alt         — Push-to-talk (если включён)
void keyboard_shortcut_func(const whisper_params& params) {
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

            if (b_alt && params.push_to_talk) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    if (g_hotkey_pressed.empty() || g_hotkey_pressed == "Alt")
                        g_hotkey_pressed = "Alt";
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (b_ctr_space && !b_ctr_space_prev && !b_ctr_space_processed) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    if (g_hotkey_pressed.empty()) g_hotkey_pressed = "Ctrl+Space";
                }
                b_ctr_space_processed = true;
            }
            else if (!b_ctr_space && b_ctr_space_prev && b_ctr_space_processed) {
                b_ctr_space_processed = false;
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (g_hotkey_pressed == "Ctrl+Space") g_hotkey_pressed = "";
            }

            if (b_ctr_right && !b_ctr_right_prev && !b_ctr_right_processed) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    if (g_hotkey_pressed.empty()) g_hotkey_pressed = "Ctrl+Right";
                }
                b_ctr_right_processed = true;
            }
            else if (!b_ctr_right && b_ctr_right_prev && b_ctr_right_processed) {
                b_ctr_right_processed = false;
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (g_hotkey_pressed == "Ctrl+Right") g_hotkey_pressed = "";
            }

            if (b_ctr_delete && !b_ctr_delete_prev && !b_ctr_delete_processed) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    if (g_hotkey_pressed.empty()) g_hotkey_pressed = "Ctrl+Delete";
                }
                b_ctr_delete_processed = true;
            }
            else if (!b_ctr_delete && b_ctr_delete_prev && b_ctr_delete_processed) {
                b_ctr_delete_processed = false;
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (g_hotkey_pressed == "Ctrl+Delete") g_hotkey_pressed = "";
            }

            if (b_ctr_r && !b_ctr_r_prev && !b_ctr_r_processed) {
                {
                    std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                    if (g_hotkey_pressed.empty()) g_hotkey_pressed = "Ctrl+R";
                }
                b_ctr_r_processed = true;
            }
            else if (!b_ctr_r && b_ctr_r_prev && b_ctr_r_processed) {
                b_ctr_r_processed = false;
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (g_hotkey_pressed == "Ctrl+R") g_hotkey_pressed = "";
            }

            b_ctr_space_prev = b_ctr_space;
            b_ctr_right_prev = b_ctr_right;
            b_ctr_delete_prev = b_ctr_delete;
            b_ctr_r_prev = b_ctr_r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    while (g_shortcut_thread_running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#endif
}

// ============================================================================
// 21. ОСНОВНАЯ ФУНКЦИЯ RUN()
// ============================================================================
// Точка входа приложения. Настраивает Whisper, LLaMA, аудио,
// запускает фоновые потоки, входит в основной цикл обработки реплик.

int run(int argc, char** argv) {
    whisper_params params;
    int input_tokens_count = 0;
    float llama_time_input = 0.0f;
    float llama_time_output = 0.0f;
    llama_sampler* smpl = nullptr;
    llama_sampler* smpl_high_temp = nullptr;

    // --- 21.1. Парсинг аргументов ---
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }
    if (params.language != "auto" &&
        whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // --- 21.1.5. Автонастройка параметров от ctx_size (PATCH F, v33) ---
    // n_predict, batch_size, min_tokens вычисляются от ctx_size.
    // n_keep — ИСКЛЮЧЕНИЕ: он фиксирован на 512 (см. ниже).
    // Типы KV-кэша — дефолт llama.cpp (f16), если не заданы явно.
    {
        if (params.ctx_size < 2048) params.ctx_size = 2048;

        // ВАЖНО (v33): n_keep больше не вычисляется от ctx_size.
        // Он фиксирован на 512 — это значение сохраняет личность
        // бота (первые 512 токенов системного промпта) и оставляет
        // ~7680 токенов под диалог при ctx=8192. Формула ctx/16
        // давала 512, но потом «уточнение по промпту» раздувало
        // его до 869, что сжимало буферную зону K-shift и ломало
        // генерацию. Возвращаем жёсткое 512.
        params.n_keep = 512;

        if (params.auto_n_predict) {
            params.n_predict = params.ctx_size / 16;
            if (params.n_predict < 256) params.n_predict = 256;
            if (params.n_predict > 1024) params.n_predict = 1024;
        }
        if (params.auto_batch_size) {
            params.batch_size = params.ctx_size / 8;
            if (params.batch_size < 128) params.batch_size = 128;
            if (params.batch_size > 2048) params.batch_size = 2048;
        }
        if (params.auto_min_tokens) {
            params.min_tokens = std::max(20, params.ctx_size / 256);
        }
        // Типы KV-кэша: по умолчанию — f16 (дефолт llama.cpp).
        // Если пользователь задал --cache-type-k/v явно — оставляем
        // как есть (на свой риск: q8_0 несовместим с K-shift на CUDA).
        if (params.verbose) {
            fprintf(stderr, "[Auto] n_keep=%d (fixed), n_predict=%d, batch_size=%d, "
                    "min_tokens=%d, cache_k='%s', cache_v='%s'\n",
                    params.n_keep, params.n_predict, params.batch_size,
                    params.min_tokens,
                    params.cache_type_k.empty() ? "f16(default)" : params.cache_type_k.c_str(),
                    params.cache_type_v.empty() ? "f16(default)" : params.cache_type_v.c_str());
        }
    }

    // --- 21.2. Проверка обязательного --prompt-file ---
    if (params.prompt.empty()) {
        fprintf(stderr, "Error: --prompt-file is required.\n");
        fprintf(stderr, "       Bot's personality must be defined explicitly.\n");
        return 1;
    }

    // --- 21.3. Начальное значение семафора XTTS ---
    {
        std::string control_path = params.xtts_control_path;
        allow_xtts_file(control_path, 1);
    }

    // --- 21.4. Подмена логгера Whisper ---
    whisper_log_set(whisper_log_filtered, nullptr);

    // --- 21.5. Логирование в файл ---
    if (params.verbose) {
        g_log_file.open("talk-llama.log", std::ios::app);
        if (g_log_file.is_open()) {
            g_log_enabled.store(true);
            log_line("=== session started ===");
        }
    }

    // --- 21.6. Инициализация Whisper ---
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    struct whisper_context* ctx_wsp =
        whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
    if (!ctx_wsp) {
        fprintf(stderr, "Failed to load whisper model: %s\n",
                params.model_wsp.c_str());
        return 1;
    }

    // --- 21.7. Инициализация LLaMA ---
    llama_backend_init();
    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
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
        fprintf(stderr, "Failed to load llama model: %s\n",
                params.model_llama.c_str());
        whisper_free(ctx_wsp);
        llama_backend_free();
        return 1;
    }
    params.tensor_split.clear();

    const llama_vocab* vocab_llama = llama_model_get_vocab(model_llama);
    bool add_bos_token = llama_vocab_get_add_bos(vocab_llama);
    // n_keep фиксирован в 21.1.5 (512). Здесь только добавляем BOS.
    int n_keep = params.n_keep + (add_bos_token ? 1 : 0);

    llama_context_params lcparams = llama_context_default_params();
    lcparams.n_ctx = params.ctx_size;
    lcparams.n_batch = params.batch_size;
    lcparams.n_ubatch = std::min(params.batch_size, 512);
    if (params.verbose) {
        fprintf(stdout, "n_ctx %d, n_batch %d, n_ubatch %d\n",
                lcparams.n_ctx, lcparams.n_batch, lcparams.n_ubatch);
    }
    lcparams.n_threads = params.n_threads;
    lcparams.flash_attn_type = params.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // --- PATCH F (v33): типы KV-кэша ---
    // По умолчанию — f16 (дефолт llama.cpp). Мы НЕ ставим q8_0
    // автоматически при --flash-attn, потому что q8_0 несовместим
    // с K-shift на CUDA (это и приводило к сбою генерации и потере
    // ответа). Если пользователь явно задал --cache-type-k/v —
    // используем его значение (на свой риск).
    if (!params.cache_type_k.empty()) {
        lcparams.type_k = parse_cache_type_str(params.cache_type_k, params.flash_attn);
    }
    if (!params.cache_type_v.empty()) {
        lcparams.type_v = parse_cache_type_str(params.cache_type_v, params.flash_attn);
    }
    // Если пусто — оставляем дефолт llama.cpp (f16).

    struct llama_context* ctx_llama = llama_init_from_model(model_llama, lcparams);
    if (!ctx_llama) {
        fprintf(stderr, "error: failed to initialize llama context\n");
        llama_model_free(model_llama);
        whisper_free(ctx_wsp);
        llama_backend_free();
        return 1;
    }

    g_ctx_llama_atomic.store(ctx_llama);

    // --- 21.9. Проверка мультиязычности Whisper ---
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en") {
                params.language = "en";
                fprintf(stderr, "%s: WARNING: model is not multilingual, using English\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s\n",
                __func__, params.n_threads, params.language.c_str());
        fprintf(stderr, "\n");
    }

    // --- 21.10. Инициализация аудио-захвата ---
    audio_async audio(15 * 1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: Ошибка инициализации аудиоустройства (ID: %d)\n",
                __func__, params.capture_id);
        llama_free(ctx_llama);
        llama_model_free(model_llama);
        whisper_free(ctx_wsp);
        llama_backend_free();
        return 1;
    }
    audio.resume();
    bool is_running = true;
    bool force_speak = false;
    const std::string chat_symb = DEFAULT_CHAT_SYMB;

    // --- 21.11. Промпт LLaMA ---
    std::string prompt_llama = params.prompt;

    // --- 21.12. Whisper-промпт ---
    g_whisper_prompt = params.person + ", " + params.bot_name + ".";
    if (params.verbose) {
        fprintf(stderr, "[Whisper] Prompt: '%s'\n", g_whisper_prompt.c_str());
    }

    // --- 21.13. Загрузка instruct-пресета ---
    if (!params.instruct_preset.empty()) {
        try {
            std::string filename = "instruct_presets/" + params.instruct_preset + ".json";
            nlohmann::json jsonData;
            std::ifstream jsonFile(filename);
            if (jsonFile.is_open()) {
                jsonFile >> jsonData;
                jsonFile.close();
                for (auto& [key, value] : jsonData.items()) {
                    std::string clean_key = normalize_json_key(key);
                    std::string val;
                    try {
                        val = value.get<std::string>();
                    }
                    catch (const std::exception&) {
                        continue;
                    }
                    val = normalize_template_token(val);
                    params.instruct_preset_data[clean_key] = val;
                }
                if (params.verbose) {
                    fprintf(stderr, "\n[Instruct] Загружен пресет из %s:\n", filename.c_str());
                    for (const auto& [k, v] : params.instruct_preset_data) {
                        if (!v.empty()) {
                            std::string printable = v;
                            size_t pos = 0;
                            while ((pos = printable.find('\n', pos)) != std::string::npos) {
                                printable.replace(pos, 1, "\\n");
                                pos += 2;
                            }
                            fprintf(stderr, "  %s = '%s'\n", k.c_str(), printable.c_str());
                        }
                    }
                }
            }
            else {
                std::cout << "Warning: preset file '" << filename
                    << "' does not exist. Turning off instruct mode\n";
                params.instruct_preset = "";
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            llama_free(ctx_llama);
            llama_model_free(model_llama);
            whisper_free(ctx_wsp);
            llama_backend_free();
            return 1;
        }
    }
    else {
        params.instruct_preset = "";
    }

    // --- 21.14. Сборка производных из пресета ---
    PresetDerived preset_d = build_preset_derived(params.instruct_preset_data);

    // Публикуем пресет для TTS-воркера.
    set_tts_preset(preset_d);

    // --- 21.15. Отключение автопоиска шаблона ---
    // ВСЕГДА используем ручную сборку из JSON. Автопоиск в
    // llama-chat.cpp может найти встроенный шаблон, несовместимый
    // с моделью (например, "yandex" описывает формат
    // "Пользователь:/Ассистент:[SEP]", несовместимый с Saiga).
    params.chat_template = LLM_CHAT_TEMPLATE_UNKNOWN;
    if (params.verbose) {
        fprintf(stderr, "[ChatML] Ручная сборка из пресета '%s'.\n",
                params.instruct_preset.c_str());
    }

    // --- 21.16. Строки времени и даты ---
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

    // --- 21.17. Текущие значения ---
    std::string default_bot_name = params.bot_name;
    std::string default_voice = params.xtts_voice;
    std::string default_prompt = prompt_llama;
    std::string current_bot_name = default_bot_name;
    std::string current_voice = default_voice;
    std::string current_prompt = default_prompt;

    // --- 21.18. Альтернативный промпт (Лео) ---
    std::string alt_prompt = default_prompt;
    if (!params.alt_prompt_file.empty()) {
        std::ifstream file(params.alt_prompt_file);
        if (file.is_open()) {
            std::copy(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>(),
                      std::back_inserter(alt_prompt));
            alt_prompt = string_replace_all(alt_prompt, "{0}", params.person);
            alt_prompt = string_replace_all(alt_prompt, "{1}", LEO_NAME);
            alt_prompt = string_replace_all(alt_prompt, "{3}", year_str);
            alt_prompt = string_replace_all(alt_prompt, "{4}", chat_symb);
            alt_prompt = string_replace_all(alt_prompt, "{5}", ymd);
            if (params.verbose) {
                fprintf(stderr, "[Prompts] Загружен альт-промпт для Лео\n");
            }
        }
        else {
            fprintf(stderr, "Warning: alt-prompt-file '%s' not found\n",
                    params.alt_prompt_file.c_str());
        }
    }

    // --- 21.19. Подстановка плейсхолдеров в основной промпт ---
    // {2} НЕ подставляем — оставляем плейсхолдер, чтобы динамически
    // подставлять время перед каждой генерацией.
    prompt_llama = string_replace_all(prompt_llama, "{0}", params.person);
    prompt_llama = string_replace_all(prompt_llama, "{1}", default_bot_name);
    prompt_llama = string_replace_all(prompt_llama, "{3}", year_str);
    prompt_llama = string_replace_all(prompt_llama, "{4}", chat_symb);
    prompt_llama = string_replace_all(prompt_llama, "{5}", ymd);
    default_prompt = prompt_llama;
    current_prompt = default_prompt;

    // --- 21.20. Батч LLaMA ---
    llama_batch batch = llama_batch_init(params.batch_size, 0, 1);
    if (params.verbose) fprintf(stdout, "llama_n_ctx %d\n", llama_n_ctx(ctx_llama));

    // --- 21.21. Сэмплеры ---
    const float top_k = static_cast<float>(params.top_k);
    const float top_p = params.top_p;
    const float min_p = params.min_p;
    float temp = params.temp;
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
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_temp(2.00f));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_dist(seed));
    }
    else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_greedy());
    }

    // --- 21.22. Токенизация промпта (правильная, пофрагментная) ---
    // Шаблонные фрагменты <|start_header_id|>, <|eot_id|> и т.д.
    // токенизируются с parse_special = true — превращаются в
    // настоящие control-токены. Пользовательский текст токенизируется
    // с parse_special = false — он не может внедрить спецтокен.
    //
    // ВАЖНО (v33): блок «уточнения n_keep по размеру промпта»
    // УДАЛЁН. n_keep фиксирован на 512, и это значение используется
    // для K-shift. Раздувание n_keep до prompt_size+128 (869)
    // сжимало буферную зону сдвига и ломало генерацию.
    std::vector<llama_token> embd_inp;
    {
        std::string updated_system_prompt = prompt_llama;

        std::string safe_user_text = "";

        bool add_bos_here = true;

        auto append_part = [&](const std::string& part, bool parse_special) {
            if (part.empty()) return;
            std::vector<llama_token> part_tokens =
                llama_tokenize_ex(ctx_llama, part, add_bos_here, parse_special);
            add_bos_here = false;
            embd_inp.insert(embd_inp.end(), part_tokens.begin(), part_tokens.end());
        };

        append_part(preset_d.sys_prefix, true);
        append_part(updated_system_prompt, false);
        append_part(preset_d.sys_suffix, true);
        append_part(preset_d.bot_prefix, true);  // начальное состояние — ждём user

        (void)safe_user_text;
    }

    // PATCH F (v33): блок уточнения n_keep удалён.
    // n_keep = 512, и это значение идёт в K-shift напрямую.
    // Если системный промпт больше 512 токенов, при K-shift
    // его хвост будет обрезан — но личность бота (первые 512 токенов)
    // сохранится, а места для диалога станет больше.

    if (static_cast<int>(embd_inp.size()) > params.ctx_size - 512) {
        int keep = std::min(params.n_keep, static_cast<int>(embd_inp.size()));
        if (static_cast<int>(embd_inp.size()) > keep + 256) {
            embd_inp.erase(embd_inp.begin() + keep, embd_inp.end() - 256);
        }
        std::cerr << "[warn] Context trimmed: " << embd_inp.size()
                  << " tokens (ctx limit " << params.ctx_size << ")\n";
    }

    // --- 21.23. Загрузка сессии ---
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
                session_tokens.data(), session_tokens.size(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file\n", __func__);
                session_tokens.clear();
            } else {
                session_tokens.resize(n_token_count_out);
                embd_inp.assign(session_tokens.begin(), session_tokens.end());
            }
        }
    }

    // --- 21.24. Прогрев контекста ---
    if (params.verbose) printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);
    float llama_start_time = get_current_time_ms();
    int n_past = 0;
    {
        if (embd_inp.size() > static_cast<size_t>(params.batch_size)) {
            for (size_t offset = 0; offset < embd_inp.size();
                 offset += params.batch_size) {
                size_t chunk = std::min(
                    static_cast<size_t>(params.batch_size),
                    embd_inp.size() - offset);
                batch.n_tokens = static_cast<int>(chunk);
                for (size_t i = 0; i < chunk; ++i) {
                    batch.token[i] = embd_inp[offset + i];
                    batch.pos[i] = static_cast<int>(offset + i);
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] =
                        ((offset + i) == embd_inp.size() - 1) ? 1 : 0;
                }
                if (llama_decode(ctx_llama, batch)) {
                    fprintf(stderr, "%s : failed to decode chunk at %zu\n",
                            __func__, offset);
                    llama_batch_free(batch);
                    llama_free(ctx_llama);
                    llama_model_free(model_llama);
                    whisper_free(ctx_wsp);
                    llama_backend_free();
                    return 1;
                }
            }
            n_past = static_cast<int>(embd_inp.size());
        }
        else {
            batch.n_tokens = static_cast<int>(embd_inp.size());
            for (int i = 0; i < batch.n_tokens; i++) {
                batch.token[i] = embd_inp[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
            }
            if (llama_decode(ctx_llama, batch)) {
                fprintf(stderr, "%s : failed to decode\n", __func__);
                llama_batch_free(batch);
                llama_free(ctx_llama);
                llama_model_free(model_llama);
                whisper_free(ctx_wsp);
                llama_backend_free();
                return 1;
            }
            n_past = static_cast<int>(embd_inp.size());
        }
    }
    float llama_end_time = get_current_time_ms();
    float llama_time_total = llama_end_time - llama_start_time;
    if (llama_time_total > 0.001f) {
        printf("\nLlama start prompt: %zu/%d tokens in %.3f s at %.0f t/s\n",
               embd_inp.size(), params.ctx_size,
               static_cast<double>(llama_time_total),
               static_cast<double>(embd_inp.size() / llama_time_total));
    } else {
        printf("\nLlama start prompt: %zu/%d tokens\n",
               embd_inp.size(), params.ctx_size);
    }
    if (params.verbose_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s\n", prompt_llama.c_str());
        fflush(stdout);
    }

    // --- 21.25. Сравнение сессии ---
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() ||
                id != embd_inp[n_matching_session_tokens]) break;
            n_matching_session_tokens++;
        }
    }
    bool need_to_save_session = !path_session.empty() &&
        n_matching_session_tokens < (embd_inp.size() * 3 / 4);

    // --- 21.26. Приветствие ---
    printf("%s : done! start speaking in the microphone\n", __func__);
    const std::string wake_cmd = params.wake_cmd;
    if (!wake_cmd.empty())
        printf("%s : the wake-up command is: '%s'\n", __func__, wake_cmd.c_str());
    printf("\n\033[90mVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right), "
           "Delete(Ctrl+Delete), Reset(Ctrl+R)\033[0m\n\n");
    if (params.push_to_talk)
        printf("\033[90mPush-to-talk: hold 'Alt' to speak\033[0m\n\n");
    fflush(stdout);

    // --- 21.27. Флаги состояния ---
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

    // --- 21.28. TTS-интро ---
    // PATCH 16 (v34): расширенный список TTS-интро.
    //
    // WHY: короткие междометия перед ответом делают речь бота живее.
    // Взято из форка Mozer — 27 вариантов вместо 10.
    // Выбирается случайно через std::mt19937 (см. ниже).
    //
    // ВАЖНО: intro проигрывается только если --xtts-intro задан в CLI.
    std::vector<std::string> tts_intros;
    std::string rand_intro_text = "";
    std::string last_output_buffer = "";
    std::string last_output_needle = "";
    std::string token_accumulator = "";
    if (params.language == "ru") {
        tts_intros = {
            "Хм", "Ну", "Нуу", "О", "А", "А?", "Угу", "Ох", "Ха", "Ах",
            "Блин", "Короче", "В общем", "Ой", "Слышь", "Ну вообще-то",
            "Ну а вообще", "Кароче", "Вот", "Знаешь", "Как бы", "Прикинь",
            "Послушай", "Типа", "Это", "Так вот", "Погоди"
        };
    }
    else {
        tts_intros = {
            "Hm", "Hmm", "Well", "Well well", "Huh", "Ugh", "Uh", "Um", "Mmm",
            "Oh", "Ooh", "Haha", "Ha ha", "Ahh", "Whoa", "Really", "I mean",
            "By the way", "Anyway", "So", "Actually", "Uh-huh", "Seriously",
            "Whatever", "Like", "But", "You know"
        };
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist;
    std::chrono::steady_clock::time_point last_command_time =
        std::chrono::steady_clock::now();

    // --- 21.29. Антипромпты ---
    //
    // Антипромпт — это строка, при появлении которой в конце
    // уже сгенерированного текста генерация останавливается.
    // Работает как мягкий стоп: если модель начала новый ход,
    // не завершив предыдущий, антипромпт ловит это.
    std::vector<std::string> antiprompts = preset_d.antiprompts;

    // PATCH 11 (v34): антипромпт "</end_of_turn>" для Gemma.
    //
    // WHY: Gemma 2/3 иногда генерирует "</end_of_turn>" (с закрывающим
    // слэшем) вместо правильного "<end_of_turn>". Стандартный
    // stop_sequence из Gemma3.json ("<end_of_turn>\n") этот баг
    // не ловит, и тег попадает в TTS. Добавляем явно.
    //
    // Дедупликация в конце раздела (std::sort + std::unique)
    // уберёт дубликат, если он уже был.
    antiprompts.push_back("</end_of_turn>");

    // PATCH 12 (v34): антипромпты "Имя:" и "Имя :" — защита от
    // продолжения диалога за пользователя.
    //
    // WHY: модель иногда пишет "Друг: ..." в ответе, что попадает
    // в TTS. Промпт просит этого не делать, но модель может
    // ослушаться. Антипромпт ловит это и обрывает генерацию.
    //
    // chat_symb = ": " (с пробелом). Значит:
    //   params.person + chat_symb        = "Друг: "
    //   params.person + " " + chat_symb  = "Друг : "
    // Оба варианта покрывают разные написания.
    antiprompts.push_back(params.person + chat_symb);
    antiprompts.push_back(params.person + " " + chat_symb);

    // Пользовательские стоп-слова из --stop-words.
    if (!params.stop_words.empty()) {
        size_t start = 0, end = params.stop_words.find(';');
        auto add_word = [&](std::string w) {
            if (w.length() >= 2) {
                w = string_replace_all(w, "\\r", "\r");
                w = string_replace_all(w, "\\n", "\n");
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
                std::string word = params.stop_words.substr(start, end - start);
                add_word(word);
                start = end + 1;
                end = params.stop_words.find(';', start);
                if (end == std::string::npos) end = params.stop_words.size();
            }
        }
    }

    // Дедупликация антипромптов.
    {
        std::sort(antiprompts.begin(), antiprompts.end());
        antiprompts.erase(std::unique(antiprompts.begin(), antiprompts.end()),
                          antiprompts.end());
    }

    // --- 21.30. Мягкий лимит токенов ---
    g_soft_limit_tokens.store(params.ctx_size / 3);
    if (params.verbose) {
        printf("[Limits] soft_limit_tokens=%d (ctx_size=%d / 3)\n",
               g_soft_limit_tokens.load(), params.ctx_size);
    }

    // --- 21.31. Запуск фоновых потоков ---
    std::thread input_thread(input_thread_func);
    std::thread shortcut_thread([&]() { keyboard_shortcut_func(params); });
    std::thread audio_thread([&]() {
        audio_input_thread_func(ctx_wsp, params, audio,
                                params.person, chat_symb);
    });
    std::thread tts_worker(tts_worker_func);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(g_display_mutex);
        g_display_text.clear();
    }

    llama_start_time = 0.0;
    float llama_start_generation_time = 0.0;
    llama_end_time = 0.0;
    llama_time_total = 0.0;
    std::string user_typed = "";
    bool user_typed_this = false;

    // --- 21.32. Заголовок окна ---
    update_console_title(params.bot_name, "Listening");

    // --- 21.33. Стартовое приглашение "Друг: " ---
    {
        std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
        std::string padded = make_padded_name(params.person);
        printf("%s%s%s: ", C_USER, padded.c_str(), C_RESET);
        fflush(stdout);
        g_initial_prompt_printed = true;
    }

    // ========================================================================
    // 22. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВВОДА И КОМАНД
    // ========================================================================
    while (is_running) {
        // --- 22.0. SDL-события (закрытие окна и т.п.) ---
        is_running = sdl_poll_events();
        if (!is_running) {
            printf("\n[Shutdown requested, cleaning up...]\n");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        int64_t t_ms = 0;
        std::string text_heard = "";
        user_typed_this = false;

        // --- 22.1. Автопродолжение после barge-in ---
        if (g_pending_continue_request.load()) {
            {
                std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                embd_inp = g_saved_embd_inp;
                n_past = g_saved_n_past;
                text_heard = "";
                user_typed = "";
                g_saved_was_interrupted = false;
            }
            text_heard = "";
            g_pending_continue_request.store(false);
            log_line("Auto-continue: state restored");
            force_speak = true;
        }

        // --- 22.2. Запрос из аудио-потока ---
        if (g_pending_llm_request.load()) {
            std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
            if (!g_pending_llm_text.empty()) {
                user_typed = g_pending_llm_text;
                user_typed_this = true;
                g_pending_llm_text.clear();
            }
            g_pending_llm_request.store(false);
        }

        // --- 22.3. Клавиатурный ввод ---
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

        // --- 22.4. Горячие клавиши ---
        std::string hk_copy;
        {
            std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
            hk_copy = g_hotkey_pressed;
            if (!(params.push_to_talk && hk_copy == "Alt")) {
                g_hotkey_pressed = "";
            }
        }
        if (!hk_copy.empty()) {
            if (hk_copy == "Ctrl+Space") { user_typed = "Stop"; user_typed_this = true; }
            else if (hk_copy == "Ctrl+Right") { user_typed = "Regenerate"; user_typed_this = true; }
            else if (hk_copy == "Ctrl+Delete") { user_typed = "Delete"; user_typed_this = true; }
            else if (hk_copy == "Ctrl+R") { user_typed = "Reset"; user_typed_this = true; }
        }

        // --- 22.5. Проверка наличия текста ---
        if (user_typed.empty() && !force_speak) continue;
        std::string display_text_for_ui = user_typed;
        trim(display_text_for_ui);
        text_heard = user_typed;
        user_typed = "";
        trim(text_heard);
        if (text_heard.empty() && !force_speak) continue;

        // --- 22.6. Wake command (если задан) ---
        if (!params.wake_cmd.empty() && !force_speak) {
            if (text_heard.find(params.wake_cmd) != 0) continue;
            text_heard = text_heard.substr(params.wake_cmd.length());
            trim(text_heard);
        }

        // --- 22.7. Удаление префикса имени пользователя ---
        if (!text_heard.empty()) {
            std::string heard_lower = LowerCase(text_heard);
            std::string person_lower = LowerCase(params.person);
            bool removed = false;
            if (heard_lower.find(person_lower + ":") == 0) {
                text_heard = text_heard.substr(person_lower.length() + 1);
                removed = true;
            }
            else if (heard_lower.find(person_lower + " :") == 0) {
                text_heard = text_heard.substr(person_lower.length() + 2);
                removed = true;
            }
            if (removed) trim(text_heard);
        }

        // --- 22.8. Хелпер печати реплики пользователя ---
        auto show_user_replica = [&](const std::string& text, const std::string& status) {
            if (g_initial_prompt_printed) {
                std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
                printf("%s", text.c_str());
                if (!status.empty()) {
                    printf(" %s[%s]%s", C_CMD, status.c_str(), C_RESET);
                }
                printf("\n");
                fflush(stdout);
                g_initial_prompt_printed = false;
                g_need_blank_before_next = true;
            } else {
                begin_new_pair();
                print_replica(C_USER, params.person, text, status);
            }
        };

        // --- 22.9. Команда «стоп» ---
        {
            std::string lower_text = LowerCase(text_heard);
            if (!force_speak && is_stop_command(lower_text)) {
                g_command_in_progress.store(true);
                std::string dummy;
                allow_xtts_file(dummy, 0);
                g_cancel_tts_requests.store(true);
                clear_tts_queue();
                show_user_replica(display_text_for_ui, "stop");
                audio.clear();
                g_force_vad_reset.store(true);
                if (g_bot_state.load() == BotState::GENERATING) {
                    g_interrupt_reason.store(InterruptReason::MANUAL_STOP);
                    g_interrupt_processed.store(true);
                }
                else {
                    g_interrupt_reason.store(InterruptReason::NONE);
                    g_interrupt_processed.store(false);
                    g_bot_state.store(BotState::IDLE);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                g_cancel_tts_requests.store(false);
                { std::string dummy2; allow_xtts_file(dummy2, 1); }
                update_console_title(params.bot_name, "Listening");
                g_command_in_progress.store(false);
                continue;
            }
        }

        // --- 22.10. Ответ heard_ok (если задан) ---
        if (!params.heard_ok.empty() && !force_speak) {
            TtsRequest hreq;
            hreq.text = params.heard_ok;
            hreq.voice = current_voice;
            hreq.language = params.language;
            hreq.url = params.xtts_url;
            hreq.stop_seq = preset_d.stop_sequence;
            hreq.bot_sfx = preset_d.bot_suffix;
            hreq.user_sfx = preset_d.user_suffix;
            hreq.bot_pfx = preset_d.bot_prefix;
            hreq.user_pfx = preset_d.user_prefix;
            hreq.chat_symb = chat_symb;
            hreq.person = params.person;
            hreq.bot = current_bot_name;
            enqueue_tts(std::move(hreq));
        }

        // --- 22.11. Очистка распознанного текста ---
        try {
            static const std::regex re_brackets(R"(\[[^\[\]]*\])");
            text_heard = std::regex_replace(text_heard, re_brackets, "");
        } catch (const std::regex_error&) {}
        if (params.language == "en" && !user_typed_this) {
            static const std::regex re_eng_only("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]");
            text_heard = std::regex_replace(text_heard, re_eng_only, "");
        }
        text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));
        text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
        text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, ",");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, ".");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "\xC2\xBB");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "[");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "]");
        text_heard = RemoveTrailingCharactersUtf8(text_heard, "\"");
        if (!text_heard.empty() && text_heard[0] == '.') text_heard.erase(0, 1);
        if (!text_heard.empty() && text_heard[0] == '[') text_heard.erase(0, 1);
        trim(text_heard);

        // --- 22.12. Детектор мусора ---
        bool is_garbage = false;
        if (!force_speak) {
            if (text_heard.empty() || text_heard == "!" || text_heard == "." ||
                text_heard == "?" || text_heard == "..." || text_heard == "!!" ||
                text_heard == "??") {
                is_garbage = true;
            }
            if (!is_garbage && is_hallucination(text_heard)) is_garbage = true;
            if (!is_garbage && text_heard == params.bot_name) is_garbage = true;
        }
        if (is_garbage) {
            text_heard = "";
            if (!force_speak) continue;
        }
        text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");
        text_heard_trimmed = text_heard;
        trim(text_heard_trimmed);
        if (!text_heard_trimmed.empty()) {
            if (text_heard_trimmed[0] == '.') text_heard_trimmed.erase(0, 1);
            if (!text_heard_trimmed.empty() && text_heard_trimmed[0] == '!')
                text_heard_trimmed.erase(0, 1);
        }
        if (!text_heard_trimmed.empty()) {
            size_t last_pos = text_heard_trimmed.length() - 1;
            if (text_heard_trimmed[last_pos] == '.' || text_heard_trimmed[last_pos] == '!')
                text_heard_trimmed.erase(last_pos, 1);
        }
        trim(text_heard);
        if (!text_heard.empty()) {
            if (text_heard[0] == '.' || text_heard[0] == '!') {
                text_heard.erase(0, 1);
                trim(text_heard);
            }
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

        // --- 22.13. TTS интро ---
        if (params.xtts_intro && !text_heard_trimmed.empty() && !force_speak) {
            dist = std::uniform_int_distribution<size_t>(0, tts_intros.size() - 1);
            rand_intro_text = tts_intros[dist(gen)];
            if (!rand_intro_text.empty()) {
                TtsRequest ireq;
                ireq.text = rand_intro_text;
                ireq.voice = current_voice;
                ireq.language = params.language;
                ireq.url = params.xtts_url;
                ireq.stop_seq = preset_d.stop_sequence;
                ireq.bot_sfx = preset_d.bot_suffix;
                ireq.user_sfx = preset_d.user_suffix;
                ireq.bot_pfx = preset_d.bot_prefix;
                ireq.user_pfx = preset_d.user_prefix;
                ireq.chat_symb = chat_symb;
                ireq.person = params.person;
                ireq.bot = current_bot_name;
                enqueue_tts(std::move(ireq));
            }
        }

        // --- 22.14. Определение команд ---
        std::string user_command;
        bool is_google = false;

        auto contains_word = [](const std::string& haystack,
                                const std::string& needle) -> bool {
            if (needle.empty()) return false;
            size_t pos = 0;
            while ((pos = haystack.find(needle, pos)) != std::string::npos) {
                bool left_ok = (pos == 0) ||
                    !std::isalnum(static_cast<unsigned char>(haystack[pos - 1]));
                size_t end_pos = pos + needle.size();
                bool right_ok = (end_pos >= haystack.size()) ||
                    !std::isalnum(static_cast<unsigned char>(haystack[end_pos]));
                if (left_ok && right_ok) return true;
                pos = end_pos;
            }
            return false;
        };

        auto starts_with_prefix = [](const std::string& haystack,
                                     const std::string& prefix) -> bool {
            if (prefix.empty() || haystack.size() < prefix.size()) return false;
            return haystack.compare(0, prefix.size(), prefix) == 0;
        };

        // ---- google ----
        static const std::vector<std::string> google_prefixes = {
            "погугли", "гугл", "гугли", "поищи", "найди", "поиск",
            "загугли", "угли",
            "google", "search", "find"
        };
        static const std::vector<std::string> google_word_only = {
            "погугли", "загугли", "нагугли"
        };
        for (const auto& var : google_prefixes) {
            if (starts_with_prefix(text_heard_trimmed, var)) {
                is_google = true;
                break;
            }
        }
        if (!is_google) {
            for (const auto& var : google_word_only) {
                if (contains_word(text_heard_trimmed, var)) {
                    is_google = true;
                    break;
                }
            }
        }
        if (is_google) user_command = "google";

        // ---- call ----
        if (user_command.empty()) {
            static const std::vector<std::string> call_words = {
                "позови", "вызови", "переключись", "call", "switch", "верни"
            };
            bool has_call_word = false;
            for (const auto& w : call_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    has_call_word = true;
                    break;
                }
            }
            if (has_call_word) {
                static const std::vector<std::string> leo_words = {
                    "лео", "лёва", "леона", "leo", "leva"
                };
                bool is_leo = false;
                for (const auto& w : leo_words) {
                    if (contains_word(text_heard_trimmed, w)) {
                        is_leo = true;
                        break;
                    }
                }
                if (is_leo) {
                    user_command = "call_leo";
                } else {
                    std::string bot_name_lower = LowerCase(default_bot_name);
                    std::string bot_name_accusative = bot_name_lower;
                    if (!bot_name_lower.empty()) {
                        char last = bot_name_lower.back();
                        if (last == 'a' || last == 'я') {
                            // Женские имена: Эмма → Эмму, Аня → Аню.
                            // (уже было в v33)
                            bot_name_accusative.pop_back();
                            bot_name_accusative += "у";
                        }
                        // PATCH 7 (v34): мужские имена на согласную.
                        //
                        // WHY: "Позови Иван" не распознавалось, потому
                        // что в винительном падеже имя звучит как
                        // "Ивана". Раньше обрабатывались только имена
                        // на -а/-я (Эмма → Эмму). Теперь добавлены
                        // окончания на согласную: Иван → Ивана,
                        // Пётр → Петра, Максим → Максима.
                        //
                        // Правило: если имя оканчивается на согласную
                        // (кроме й, ь, ъ), в винительном падеже
                        // добавляется "а".
                        else if (last == 'н' || last == 'р' || last == 'л' ||
                                 last == 'м' || last == 'в' || last == 'д' ||
                                 last == 'т' || last == 'с' || last == 'к' ||
                                 last == 'п' || last == 'б' || last == 'з' ||
                                 last == 'г' || last == 'х' || last == 'ж' ||
                                 last == 'ш' || last == 'щ' || last == 'ч' ||
                                 last == 'ц' || last == 'ф') {
                            bot_name_accusative += "а";
                        }
                    }
                    if (contains_word(text_heard_trimmed, bot_name_lower) ||
                        contains_word(text_heard_trimmed, bot_name_accusative)) {
                        user_command = "call_default";
                    }
                }
            }
        }

        // ---- regenerate ----
        if (user_command.empty()) {
            static const std::vector<std::string> regen_words = {
                "переделай", "переделаем", "заново", "перегенерируй",
                "перегенерировать", "regenerate"
            };
            for (const auto& w : regen_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    user_command = "regenerate";
                    break;
                }
            }
        }

        // ---- repeat ----
        if (user_command.empty()) {
            static const std::vector<std::string> repeat_words = { "повтори", "repeat" };
            for (const auto& w : repeat_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    user_command = "repeat";
                    break;
                }
            }
        }

        // ---- delete ----
        if (user_command.empty()) {
            static const std::vector<std::string> delete_words = {
                "удали", "удалить", "сотри", "стереть", "убери",
                "delete", "remove", "erase"
            };
            for (const auto& w : delete_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    user_command = "delete";
                    break;
                }
            }
        }

        // ---- reset ----
        if (user_command.empty()) {
            static const std::vector<std::string> reset_words = {
                "сброс", "сбросить", "обнули", "reset"
            };
            for (const auto& w : reset_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    user_command = "reset";
                    break;
                }
            }
        }

        // ---- time ----
        if (user_command.empty()) {
            static const std::vector<std::string> time_phrases = {
                "который час", "сколько времени", "сколько час",
                "какой час", "скажи время", "текущее время"
            };
            for (const auto& phrase : time_phrases) {
                if (text_heard_trimmed.find(phrase) != std::string::npos) {
                    user_command = "time";
                    break;
                }
            }
            if (user_command.empty()) {
                std::string t = text_heard_trimmed;
                trim(t);
                if (t == "время" || t == "время?" ||
                    t.find("время?") != std::string::npos) {
                    size_t vp = t.rfind("время");
                    if (vp != std::string::npos &&
                        vp + 5 >= t.size() - 1) {
                        user_command = "time";
                    }
                }
            }
        }

        // ---- date ----
        if (user_command.empty()) {
            static const std::vector<std::string> date_phrases = {
                "какая дата", "какое число", "какой день",
                "какой сегодня день", "скажи дату", "сегодняшняя дата",
                "какая сегодня дата", "которое число",
                "какое сегодня число", "какой день недели", "день недели",
                "what date", "what's the date", "current date",
                "what day is it", "tell me the date", "today's date"
            };
            for (const auto& phrase : date_phrases) {
                if (text_heard_trimmed.find(phrase) != std::string::npos) {
                    user_command = "date";
                    break;
                }
            }
            if (user_command.empty()) {
                std::string t = text_heard_trimmed;
                trim(t);
                if (t == "дата" || t == "дата?" || t == "число" || t == "число?") {
                    user_command = "date";
                }
            }
        }

        // ---- Ограничение частоты команд ----
        if (!user_command.empty() && !new_command_allowed) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - last_command_time).count();
            if (elapsed >= 2) new_command_allowed = 1;
        }

        // --- 22.15. Обработчики команд ---
        bool is_high_priority =
            (user_command == "reset" || user_command == "delete" ||
             user_command == "regenerate");

        // ---- google ----
        if (user_command == "google" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string query = ParseCommandAndGetKeyword(text_heard_trimmed, "google");
            trim(query);
            if (!query.empty()) {
                show_user_replica(text_heard, "google");
                std::string url = params.google_url + "google?q=" + UrlEncode(query);
                std::string google_resp = send_curl(url);
                std::string out_text;
                if (!google_resp.empty()) {
                    if (google_resp.length() > 500) {
                        google_resp = google_resp.substr(0, 500);
                        size_t last_space = google_resp.find_last_of(' ');
                        if (last_space != std::string::npos && last_space > 400)
                            google_resp = google_resp.substr(0, last_space);
                    }
                    out_text = "Поиск показал: " + google_resp;
                } else {
                    out_text = "Не удалось найти информацию";
                }
                print_replica(C_BOT, current_bot_name, out_text, "tts");
                TtsRequest req;
                req.text = out_text;
                req.voice = current_voice;
                req.language = params.language;
                req.url = params.xtts_url;
                req.stop_seq = preset_d.stop_sequence;
                req.bot_sfx = preset_d.bot_suffix;
                req.user_sfx = preset_d.user_suffix;
                req.bot_pfx = preset_d.bot_prefix;
                req.user_pfx = preset_d.user_prefix;
                req.chat_symb = chat_symb;
                req.person = params.person;
                req.bot = current_bot_name;
                enqueue_tts(std::move(req));
            }
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            update_console_title(params.bot_name, "Listening");
            fflush(stdout);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- call_leo ----
        if (user_command == "call_leo" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            current_bot_name = LEO_NAME;
            current_voice = LEO_VOICE;
            current_prompt = alt_prompt;
            g_whisper_prompt = params.person + ", " + current_bot_name + ".";
            if (params.verbose) {
                fprintf(stderr, "[Whisper] Prompt обновлён: '%s'\n",
                        g_whisper_prompt.c_str());
            }
            show_user_replica(text_heard, "call leo");
            TtsRequest req;
            req.text = "Я Лео, слушаю";
            req.voice = current_voice;
            req.language = params.language;
            req.url = params.xtts_url;
            req.stop_seq = preset_d.stop_sequence;
            req.bot_sfx = preset_d.bot_suffix;
            req.user_sfx = preset_d.user_suffix;
            req.bot_pfx = preset_d.bot_prefix;
            req.user_pfx = preset_d.user_prefix;
            req.chat_symb = chat_symb;
            req.person = params.person;
            req.bot = current_bot_name;
            enqueue_tts(std::move(req));
            update_console_title(current_bot_name, "Listening");
            fflush(stdout);
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- call_default ----
        if (user_command == "call_default" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            current_bot_name = default_bot_name;
            current_voice = default_voice;
            current_prompt = default_prompt;
            g_whisper_prompt = params.person + ", " + current_bot_name + ".";
            if (params.verbose) {
                fprintf(stderr, "[Whisper] Prompt обновлён: '%s'\n",
                        g_whisper_prompt.c_str());
            }
            std::string status = "call " + LowerCase(current_bot_name);
            show_user_replica(text_heard, status);
            TtsRequest req;
            req.text = "Слушаю";
            req.voice = current_voice;
            req.language = params.language;
            req.url = params.xtts_url;
            req.stop_seq = preset_d.stop_sequence;
            req.bot_sfx = preset_d.bot_suffix;
            req.user_sfx = preset_d.user_suffix;
            req.bot_pfx = preset_d.bot_prefix;
            req.user_pfx = preset_d.user_prefix;
            req.chat_symb = chat_symb;
            req.person = params.person;
            req.bot = current_bot_name;
            enqueue_tts(std::move(req));
            update_console_title(current_bot_name, "Listening");
            fflush(stdout);
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- regenerate ----
        if (user_command == "regenerate" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string dummy;
            allow_xtts_file(dummy, 0);
            g_cancel_tts_requests.store(true);
            clear_tts_queue();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            show_user_replica(text_heard, "regen");
            if (!past_prev_arr.empty()) {
                n_past_prev = past_prev_arr.back();
                past_prev_arr.pop_back();
                int rollback_num = static_cast<int>(embd_inp.size()) - n_past_prev;
                if (rollback_num > 0 && rollback_num <= static_cast<int>(embd_inp.size())) {
                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
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
                        TtsRequest req;
                        req.text = text_to_respeak_safe;
                        req.voice = current_voice;
                        req.language = params.language;
                        req.url = params.xtts_url;
                        req.stop_seq = preset_d.stop_sequence;
                        req.bot_sfx = preset_d.bot_suffix;
                        req.user_sfx = preset_d.user_suffix;
                        req.bot_pfx = preset_d.bot_prefix;
                        req.user_pfx = preset_d.user_prefix;
                        req.chat_symb = chat_symb;
                        req.person = params.person;
                        req.bot = current_bot_name;
                        enqueue_tts(std::move(req));
                    }
                }
            }
            g_cancel_tts_requests.store(false);
            { std::string dummy2; allow_xtts_file(dummy2, 1); }
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- repeat ----
        if (user_command == "repeat" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string last_text;
            {
                std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                last_text = g_last_tts_text;
            }
            show_user_replica(text_heard, "repeat");
            if (!last_text.empty()) {
                print_replica(C_BOT, current_bot_name, last_text, "tts");
                TtsRequest req;
                req.text = last_text;
                req.voice = current_voice;
                req.language = params.language;
                req.url = params.xtts_url;
                req.stop_seq = preset_d.stop_sequence;
                req.bot_sfx = preset_d.bot_suffix;
                req.user_sfx = preset_d.user_suffix;
                req.bot_pfx = preset_d.bot_prefix;
                req.user_pfx = preset_d.user_prefix;
                req.chat_symb = chat_symb;
                req.person = params.person;
                req.bot = current_bot_name;
                enqueue_tts(std::move(req));
            }
            else {
                print_replica(C_BOT, current_bot_name, "Нечего повторять", "");
            }
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- delete ----
        if (user_command == "delete" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            show_user_replica(text_heard, "del");
            if (!past_prev_arr.empty()) {
                n_past_prev = past_prev_arr.back();
                past_prev_arr.pop_back();
                int rollback_num = static_cast<int>(embd_inp.size()) - n_past_prev;
                if (rollback_num > 0 && rollback_num <= static_cast<int>(embd_inp.size())) {
                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    llama_memory_seq_rm(llama_get_memory(ctx_llama), 0,
                        static_cast<int>(embd_inp.size()), -1);
                    last_command_time = std::chrono::steady_clock::now();
                    new_command_allowed = 0;
                    TtsRequest req;
                    req.text = "Удалено";
                    req.voice = GOOGLE_VOICE;
                    req.language = params.language;
                    req.url = params.xtts_url;
                    req.stop_seq = preset_d.stop_sequence;
                    req.bot_sfx = preset_d.bot_suffix;
                    req.user_sfx = preset_d.user_suffix;
                    req.bot_pfx = preset_d.bot_prefix;
                    req.user_pfx = preset_d.user_prefix;
                    req.chat_symb = chat_symb;
                    req.person = params.person;
                    req.bot = current_bot_name;
                    enqueue_tts(std::move(req));
                }
            }
            else {
                TtsRequest req;
                req.text = "Нечего удалять";
                req.voice = GOOGLE_VOICE;
                req.language = params.language;
                req.url = params.xtts_url;
                req.stop_seq = preset_d.stop_sequence;
                req.bot_sfx = preset_d.bot_suffix;
                req.user_sfx = preset_d.user_suffix;
                req.bot_pfx = preset_d.bot_prefix;
                req.user_pfx = preset_d.user_prefix;
                req.chat_symb = chat_symb;
                req.person = params.person;
                req.bot = current_bot_name;
                enqueue_tts(std::move(req));
            }
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            fflush(stdout);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- reset ----
        if (user_command == "reset" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            g_reset_in_progress.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::string dummy;
            allow_xtts_file(dummy, 0);
            g_cancel_tts_requests.store(true);
            clear_tts_queue();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            show_user_replica(text_heard, "reset");
            if (g_bot_state.load() == BotState::GENERATING) {
                g_interrupt_reason.store(InterruptReason::MANUAL_STOP);
                g_interrupt_processed.store(true);
                g_bot_state.store(BotState::IDLE);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!past_prev_arr.empty()) {
                n_past_prev = past_prev_arr.front();
                past_prev_arr.clear();
                {
                    std::lock_guard<std::mutex> lock(g_llama_mutex);
                    llama_batch_free(batch);
                    g_ctx_llama_atomic.store(nullptr);
                    if (ctx_llama) {
                        llama_free(ctx_llama);
                        ctx_llama = nullptr;
                    }
                    ctx_llama = llama_init_from_model(model_llama, lcparams);
                    if (!ctx_llama) {
                        fprintf(stderr, "%s : ERROR: Failed to reinitialize llama context on reset\n", __func__);
                        g_reset_in_progress.store(false);
                        g_reset_cv.notify_all();
                        g_command_in_progress.store(false);
                        return 1;
                    }
                    g_ctx_llama_atomic.store(ctx_llama);
                    batch = llama_batch_init(params.batch_size, 0, 1);
                    embd_inp = ::llama_tokenize(ctx_llama, current_prompt, true);
                    if (embd_inp.empty()) {
                        fprintf(stderr, "%s : ERROR: Prompt size invalid\n", __func__);
                        g_reset_in_progress.store(false);
                        g_reset_cv.notify_all();
                        g_command_in_progress.store(false);
                        return 1;
                    }
                    size_t total = embd_inp.size();
                    size_t offset = 0;
                    while (offset < total) {
                        size_t chunk = std::min(
                            static_cast<size_t>(params.batch_size),
                            total - offset);
                        batch.n_tokens = static_cast<int>(chunk);
                        for (size_t i = 0; i < chunk; i++) {
                            batch.token[i] = embd_inp[offset + i];
                            batch.pos[i] = static_cast<int>(offset + i);
                            batch.n_seq_id[i] = 1;
                            batch.seq_id[i][0] = 0;
                            batch.logits[i] = ((offset + i) == total - 1) ? 1 : 0;
                        }
                        if (llama_decode(ctx_llama, batch)) {
                            fprintf(stderr, "%s : failed to decode after reset\n", __func__);
                            g_reset_in_progress.store(false);
                            g_reset_cv.notify_all();
                            g_command_in_progress.store(false);
                            return 1;
                        }
                        offset += chunk;
                    }
                }
                n_past = static_cast<int>(embd_inp.size());
                n_session_consumed = static_cast<int>(embd_inp.size());
                text_heard = "";
                text_heard_trimmed = "";

                // PATCH 5 (v34): переименованный счётчик.
                g_transcription_count.store(0);

                TtsRequest req;
                req.text = "Контекст сброшен";
                req.voice = GOOGLE_VOICE;
                req.language = params.language;
                req.url = params.xtts_url;
                req.stop_seq = preset_d.stop_sequence;
                req.bot_sfx = preset_d.bot_suffix;
                req.user_sfx = preset_d.user_suffix;
                req.bot_pfx = preset_d.bot_prefix;
                req.user_pfx = preset_d.user_prefix;
                req.chat_symb = chat_symb;
                req.person = params.person;
                req.bot = current_bot_name;
                enqueue_tts(std::move(req));
                { std::string dummy2; allow_xtts_file(dummy2, 1); }
                new_command_allowed = 0;
                last_command_time = std::chrono::steady_clock::now();
            }
            else {
                TtsRequest req;
                req.text = "Нечего сбрасывать";
                req.voice = GOOGLE_VOICE;
                req.language = params.language;
                req.url = params.xtts_url;
                req.stop_seq = preset_d.stop_sequence;
                req.bot_sfx = preset_d.bot_suffix;
                req.user_sfx = preset_d.user_suffix;
                req.bot_pfx = preset_d.bot_prefix;
                req.user_pfx = preset_d.user_prefix;
                req.chat_symb = chat_symb;
                req.person = params.person;
                req.bot = current_bot_name;
                enqueue_tts(std::move(req));
            }
            g_cancel_tts_requests.store(false);
            { std::string dummy2; allow_xtts_file(dummy2, 1); }
            g_reset_in_progress.store(false);
            g_reset_cv.notify_all();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            fflush(stdout);
            g_command_in_progress.store(false);
            continue;
        }

        // ---- time ----
        if (user_command == "time" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::time_t t_now = std::time(nullptr);
            std::tm tm_local_now{};
#ifdef _WIN32
            if (localtime_s(&tm_local_now, &t_now) != 0) {
                audio.clear();
                g_command_in_progress.store(false);
                continue;
            }
#else
            if (localtime_r(&t_now, &tm_local_now) == nullptr) {
                audio.clear();
                g_command_in_progress.store(false);
                continue;
            }
#endif
            char time_buffer[64];
            snprintf(time_buffer, sizeof(time_buffer), "Сейчас %02d:%02d",
                     tm_local_now.tm_hour, tm_local_now.tm_min);
            show_user_replica(text_heard, "time");
            print_replica(C_BOT, current_bot_name, time_buffer, "tts");
            TtsRequest req;
            req.text = time_buffer;
            req.voice = GOOGLE_VOICE;
            req.language = params.language;
            req.url = params.xtts_url;
            req.stop_seq = preset_d.stop_sequence;
            req.bot_sfx = preset_d.bot_suffix;
            req.user_sfx = preset_d.user_suffix;
            req.bot_pfx = preset_d.bot_prefix;
            req.user_pfx = preset_d.user_prefix;
            req.chat_symb = chat_symb;
            req.person = params.person;
            req.bot = current_bot_name;
            enqueue_tts(std::move(req));
            fflush(stdout);
            audio.clear();
            g_command_in_progress.store(false);
            continue;
        }

        // ---- date ----
        if (user_command == "date" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::time_t t_now = std::time(nullptr);
            std::tm tm_local_now{};
#ifdef _WIN32
            if (localtime_s(&tm_local_now, &t_now) != 0) {
                audio.clear();
                g_command_in_progress.store(false);
                continue;
            }
#else
            if (localtime_r(&t_now, &tm_local_now) == nullptr) {
                audio.clear();
                g_command_in_progress.store(false);
                continue;
            }
#endif
            static const char* months_ru[] = {
                "января", "февраля", "марта", "апреля", "мая", "июня",
                "июля", "августа", "сентября", "октября", "ноября", "декабря"
            };
            static const char* weekdays_ru[] = {
                "воскресенье", "понедельник", "вторник", "среда",
                "четверг", "пятница", "суббота"
            };
            char date_buffer[256];
            snprintf(date_buffer, sizeof(date_buffer), "%d %s %d года, %s",
                     tm_local_now.tm_mday, months_ru[tm_local_now.tm_mon],
                     tm_local_now.tm_year + 1900, weekdays_ru[tm_local_now.tm_wday]);
            show_user_replica(text_heard, "date");
            print_replica(C_BOT, current_bot_name, date_buffer, "tts");
            TtsRequest req;
            req.text = date_buffer;
            req.voice = GOOGLE_VOICE;
            req.language = params.language;
            req.url = params.xtts_url;
            req.stop_seq = preset_d.stop_sequence;
            req.bot_sfx = preset_d.bot_suffix;
            req.user_sfx = preset_d.user_suffix;
            req.bot_pfx = preset_d.bot_prefix;
            req.user_pfx = preset_d.user_prefix;
            req.chat_symb = chat_symb;
            req.person = params.person;
            req.bot = current_bot_name;
            enqueue_tts(std::move(req));
            fflush(stdout);
            audio.clear();
            g_command_in_progress.store(false);
            continue;
        }

        // ====================================================================
        // 22.16. НАЧАЛО ГЕНЕРАЦИИ LLaMA
        // ====================================================================
        int tokens_in_reply = 0;
        llama_start_generation_time = 0.0f;
        llama_start_time = get_current_time_ms();

        if (!force_speak) {
            if (text_heard.empty()) {
                audio.clear();
                { std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex); g_hotkey_pressed = ""; }
                continue;
            }
            trim(text_heard);
            text_heard_prev = text_heard;

            n_past_prev = static_cast<int>(embd_inp.size());
            past_prev_arr.push_back(static_cast<int>(embd_inp.size()));
            if (past_prev_arr.size() > MAX_PAST_PREV_SIZE)
                past_prev_arr.erase(past_prev_arr.begin());

            show_user_replica(text_heard, "");
        }
        force_speak = false;

        std::string text_heard_with_instruct;

        update_console_title(current_bot_name, "Generating...");
        g_bot_state.store(BotState::GENERATING);

        // --- 22.17. Формирование промпта (правильная пофрагментная токенизация) ---
        {
            std::time_t t_now = std::time(nullptr);
            std::tm tm_now{};
#ifdef _WIN32
            localtime_s(&tm_now, &t_now);
#else
            localtime_r(&t_now, &tm_now);
#endif
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%02d:%02d", tm_now.tm_hour, tm_now.tm_min);
            std::string fresh_time = buf;

            std::string updated_system_prompt = string_replace_all(current_prompt, "{2}", fresh_time);

            std::string safe_user_text = sanitize_user_text(text_heard);

            embd.clear();
            bool add_bos_here = true;

            auto append_part = [&](const std::string& part, bool parse_special) {
                if (part.empty()) return;
                std::vector<llama_token> part_tokens =
                    llama_tokenize_ex(ctx_llama, part, add_bos_here, parse_special);
                add_bos_here = false;
                embd.insert(embd.end(), part_tokens.begin(), part_tokens.end());
            };

            append_part(preset_d.sys_prefix, true);
            append_part(updated_system_prompt, false);
            append_part(preset_d.sys_suffix, true);

            append_part(preset_d.user_prefix, true);
            append_part(safe_user_text, false);
            append_part(preset_d.user_suffix, true);

            append_part(preset_d.bot_prefix, true);

            input_tokens_count = static_cast<int>(embd.size());
            if (!path_session.empty())
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
        }

        if (force_speak) {
            if (!embd_inp.empty()) {
                embd.clear();
                embd.push_back(embd_inp.back());
                n_past = static_cast<int>(embd_inp.size()) - 1;
            }
            input_tokens_count = 0;
        }
        else if (text_heard.empty()) {
            embd.clear();
            input_tokens_count = 0;
        }

        { std::string dummy; allow_xtts_file(dummy, 1); }

        (void)text_heard_with_instruct;

        // --- 22.19. Локальные переменные цикла генерации ---
        float temp_next = params.temp;
        bool done = false;
        std::string text_to_speak;
        std::string full_response_text;
        int new_tokens = 0;
        std::string tts_smart_buffer;
        bool was_interrupted = false;
        int loop_detection_counter = 0;
        int tokens_since_last_loop = 0;
        int seqrep_counter = 0;
        bool bot_prefix_printed = false;
        // Счётчик «пустых» итераций: если embd оказался пуст (сбой),
        // не крутимся вхолостую — выходим. Это защита от ситуации,
        // когда K-shift оставил embd пустым, и цикл ушёл в никуда.
        int empty_iterations = 0;
        const int MAX_EMPTY_ITERATIONS = 5;
        last_output_buffer.clear();
        last_output_needle.clear();
        token_accumulator.clear();
        tts_smart_buffer.clear();
        g_interrupt_processed.store(false);

        llama_sampler_reset(smpl);
        llama_sampler_reset(smpl_high_temp);

        // ====================================================================
        // 22.20. ОСНОВНОЙ ЦИКЛ ГЕНЕРАЦИИ ТОКЕНОВ
        // ====================================================================
        while (true) {
            // --- 22.20.1. Проверка прерывания ---
            InterruptReason reason = g_interrupt_reason.load();
            if (reason != InterruptReason::NONE) {
                std::string dummy;
                allow_xtts_file(dummy, 0);
                g_cancel_tts_requests.store(true);
                clear_tts_queue();
                done = true;
                was_interrupted = true;

                {
                    std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                    g_saved_generation_text = full_response_text;
                    g_saved_n_past = n_past;
                    g_saved_embd_inp = embd_inp;
                    g_saved_timestamp = get_current_time_ms();
                    g_saved_was_interrupted = true;
                }
                text_to_speak.clear();
                tts_smart_buffer.clear();

                if (reason == InterruptReason::VAD_SPEECH) {
                    if (bot_prefix_printed) {
                        append_status_to_current_line("int", C_STOP);
                    }
                } else if (reason == InterruptReason::HOTKEY_STOP) {
                    if (bot_prefix_printed) {
                        append_status_to_current_line("stop", C_STOP);
                    }
                } else if (reason == InterruptReason::MANUAL_STOP) {
                    if (bot_prefix_printed) {
                        append_status_to_current_line("stop", C_STOP);
                    }
                }
                bot_prefix_printed = false;
                g_bot_state.store(BotState::IDLE);
                break;
            }

            // --- 22.20.2. Проверка: ctx_llama не null ---
            llama_context* active_ctx = g_ctx_llama_atomic.load();
            if (!active_ctx) {
                fprintf(stderr, "\n[ERROR] ctx_llama is null during generation\n");
                done = true; was_interrupted = true; break;
            }

            if (g_reset_in_progress.load()) {
                done = true; was_interrupted = true; break;
            }

            // --- 22.20.3. Проверка горячей клавиши ---
            {
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (!g_hotkey_pressed.empty() && g_hotkey_pressed != "Alt") {
                    std::string dummy;
                    allow_xtts_file(dummy, 0);
                    g_cancel_tts_requests.store(true);
                    clear_tts_queue();
                    g_interrupt_reason.store(InterruptReason::HOTKEY_STOP);
                    g_interrupt_processed.store(true);
                    g_bot_state.store(BotState::IDLE);
                    done = true; was_interrupted = true;
                    text_to_speak = ""; tts_smart_buffer.clear();
                    g_hotkey_pressed = "";
                    break;
                }
            }

            // --- 22.20.4. Проверка лимита токенов ---
            if (new_tokens > params.n_predict) {
                if (params.verbose) {
                    fprintf(stderr, "\n[Generation] Лимит токенов (%d)\n",
                            params.n_predict);
                }
                done = true;
                break;
            }

            // --- 22.20.5. Обработка переполнения контекста (K-shift, v33) ---
            // КОМБИНИРОВАННЫЙ ПОДХОД:
            //   1. n_discard = n_left / 4 (как в старой версии, а не /3).
            //   2. Клампинг n_discard в [0, n_left - 1] — защита от
            //      underflow и от полного удаления контекста.
            //   3. Проверка llama_memory_can_shift() перед сдвигом.
            //      Если сдвиг невозможен — fallback на обрезку до n_keep.
            //   4. После сдвига embd НЕ очищается. В него кладутся
            //      последние n_prev = 64 токенов из embd_inp. Это
            //      устраняет холостое вращение цикла.
            //   5. Убран continue после сдвига — цикл продолжает работу
            //      с непустым embd.
            //   6. text_to_speak НЕ очищается — уже сгенерированный
            //      ответ пойдёт в TTS.
            if (embd.size() > 0) {
                if (n_past + static_cast<int>(embd.size()) > n_ctx) {
                    std::lock_guard<std::mutex> lock(g_llama_mutex);
                    active_ctx = g_ctx_llama_atomic.load();
                    if (!active_ctx) { done = true; break; }

                    llama_memory_t mem = llama_get_memory(active_ctx);

                    if (!llama_memory_can_shift(mem)) {
                        if (params.verbose) {
                            fprintf(stderr, "\n[KShift] can_shift=false, "
                                    "пропускаем сдвиг, fallback to n_keep\n");
                        }
                        // Fallback: обрезка KV-кэша и embd_inp до n_keep.
                        // ВАЖНО: embd НЕ трогаем — в нём лежат токены,
                        // которые ещё не декодированы. Они будут
                        // декодированы на позициях начиная с нового n_past.
                        embd_inp.resize(std::min(static_cast<size_t>(n_keep),
                                                 embd_inp.size()));
                        llama_memory_seq_rm(mem, 0, n_keep, -1);

                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;

                        token_accumulator.clear();
                        last_output_buffer.clear();
                        last_output_needle.clear();
                        past_prev_arr.clear();
                    }

                    // Параметры сдвига.
                    const int n_left_local = std::max(0, n_past - n_keep);
                    int n_discard_local = n_left_local / 4;  // как в старой версии

                    // Клампинг: не удаляем всё, оставляем минимум 1 токен
                    // после n_keep. Защита от underflow.
                    n_discard_local = std::clamp(n_discard_local, 0,
                                                 std::max(0, n_left_local - 1));

                    bool context_updated = false;

                    if (n_discard_local > 0 &&
                        n_keep + n_discard_local <= static_cast<int>(embd_inp.size()) &&
                        n_keep + n_discard_local <= n_past) {

                        // Удаляем диапазон [n_keep, n_keep+n_discard) из KV.
                        bool rm_ok = llama_memory_seq_rm(mem, 0,
                                                         n_keep,
                                                         n_keep + n_discard_local);
                        if (rm_ok) {
                            llama_memory_seq_add(mem, 0,
                                                 n_keep + n_discard_local,
                                                 n_past, -n_discard_local);

                            // Локальный буфер.
                            embd_inp.erase(embd_inp.begin() + n_keep,
                                           embd_inp.begin() + n_keep + n_discard_local);

                            // Сохранение сессии (если используется).
                            if (!path_session.empty()) {
                                if (session_tokens.size() > embd_inp.size())
                                    session_tokens.resize(embd_inp.size());
                                llama_state_save_file(active_ctx, path_session.c_str(),
                                    session_tokens.data(), session_tokens.size());
                            }

                            context_updated = true;
                            if (params.verbose) {
                                fprintf(stderr, "\n[KShift] Удал. %d ток, ост: %zu\n",
                                        n_discard_local, embd_inp.size());
                            }
                        }
                    }

                    if (!context_updated) {
                        // Fallback: обрезка до n_keep.
                        // ВАЖНО: embd остаётся пустым, n_past = размер embd_inp.
                        // KV-кэш содержит только префикс [0, n_keep).
                        // Первый же decode нового токена должен идти с позиции n_keep.
                        if (params.verbose) {
                            fprintf(stderr, "\n[KShift] Fallback: resize to n_keep=%d\n",
                                    n_keep);
                        }
                        embd_inp.resize(std::min(static_cast<size_t>(n_keep),
                                                 embd_inp.size()));

                        // Обрезка KV-кэша: оставляем только [0, n_keep).
                        // Это критично: после resize embd_inp мы ДОЛЖНЫ
                        // привести KV-кэш в согласованное состояние,
                        // иначе decode упадёт с inconsistent positions.
                        llama_memory_seq_rm(mem, 0, n_keep, -1);
                    }

                    // Обновляем счётчики.
                    // n_past теперь указывает на позицию СЛЕДУЮЩЕГО токена,
                    // который мы сгенерируем. Все токены [0, n_past) уже в KV-кэше.
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    need_to_save_session = true;

                    // Восстановление BOS, если нужно.
                    const llama_vocab* vocab_local = llama_model_get_vocab(model_llama);
                    if (vocab_local) {
                        const llama_token bos_token = llama_vocab_bos(vocab_local);
                        if (!embd_inp.empty() && embd_inp[0] != bos_token) {
                            embd_inp.insert(embd_inp.begin(), bos_token);
                            n_past = static_cast<int>(embd_inp.size());
                            n_session_consumed = n_past;
                        }
                    }

                    // Сброс вспомогательных буферов.
                    token_accumulator.clear();
                    last_output_buffer.clear();
                    last_output_needle.clear();
                    past_prev_arr.clear();

                    // === ИСПРАВЛЕНИЕ ===
                    // embd НЕ трогаем. В нём уже лежат токены, которые
                    // нужно декодировать на следующем шаге (обычно 1 —
                    // последний сэмплированный, либо N — собранный
                    // промпт). После K-shift эти токены ещё НЕ в KV,
                    // и будут декодированы с позиции нового n_past.
                    //
                    // Попытки «докладывать» сюда токены из embd_inp
                    // (или очищать embd) приводят к тому, что либо
                    // дублируются KV-записи, либо теряется токен —
                    // и в обоих случаях llama_decode падает с
                    // inconsistent sequence positions.
                    n_session_consumed = n_past;

                    // НЕ делаем continue: цикл сам декодирует embd.
                }
            }

            // Считаем итерацию генерации.
            new_tokens++;

            // --- 22.20.6. Восстановление сессии ---
            if (n_session_consumed < static_cast<int>(session_tokens.size())) {
                size_t i = 0;
                int max_check = std::min(static_cast<int>(embd.size()),
                    static_cast<int>(session_tokens.size()) - n_session_consumed);
                for (; i < static_cast<size_t>(max_check); i++) {
                    if (n_session_consumed >= static_cast<int>(session_tokens.size())) break;
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed); break;
                    }
                    embd_inp.push_back(embd[i]);
                    n_session_consumed++;
                    if (n_session_consumed >= static_cast<int>(session_tokens.size())) {
                        i++; break;
                    }
                }
                if (i > 0) embd.erase(embd.begin(), embd.begin() + static_cast<std::ptrdiff_t>(i));
                n_past = static_cast<int>(embd_inp.size());
            }
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = static_cast<int>(session_tokens.size());
            }

            // --- 22.20.7. Декодирование ---
            // v34: пустой embd — нормальная ситуация СРАЗУ ПОСЛЕ K-shift
            // (embd.clear() в 22.20.5). НЕ выходим из итерации через
            // continue — это блокировало бы семплер в 22.20.9 (он стоит
            // ПОСЛЕ этого блока). Вместо этого просто пропускаем
            // декодирование и идём дальше к семплеру: он сгенерирует
            // новый токен, и на СЛЕДУЮЩЕЙ итерации embd уже будет
            // содержать этот токен для декодирования.
            //
            // decode_failed=true по-прежнему означает реальный сбой
            // llama_decode и требует пропуска итерации через continue.
            bool decode_failed = false;
            bool decode_skipped = false;
            {
                std::lock_guard<std::mutex> lock(g_llama_mutex);
                active_ctx = g_ctx_llama_atomic.load();
                if (!active_ctx) { done = true; break; }

                if (embd.empty()) {
                    // Нечего декодировать. Это НЕ ошибка, а штатное
                    // состояние после K-shift. Идём к семплеру.
                    decode_skipped = true;
                } else {
                    if (embd.size() > static_cast<size_t>(params.batch_size)) {
                        embd.resize(params.batch_size);
                    }
                    batch.n_tokens = static_cast<int>(embd.size());
                    for (int i = 0; i < batch.n_tokens; ++i) {
                        batch.token[i] = embd[i];
                        batch.pos[i] = n_past + i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
                    }
                    if (llama_decode(active_ctx, batch)) {
                        fprintf(stderr, "%s : failed to decode\n", __func__);
                        llama_memory_seq_rm(llama_get_memory(active_ctx), 0, n_past, -1);
                        embd.clear();
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        decode_failed = true;
                    }
                }
            }

            if (decode_failed) {
                // Реальный сбой decode: пропускаем итерацию,
                // счётчик пустых итераций страхует от зацикливания.
                empty_iterations++;
                if (empty_iterations > MAX_EMPTY_ITERATIONS) {
                    if (params.verbose) {
                        fprintf(stderr, "\n[Generation] Слишком много сбоев "
                                "decode (%d), выход из цикла\n",
                                empty_iterations);
                    }
                    done = true;
                    break;
                }
                continue;
            }

            if (decode_skipped) {
                // embd пуст — декодировать нечего, но семплер должен
                // сработать. НЕ трогаем embd_inp, НЕ трогаем n_past:
                // они уже согласованы с KV-кэшем (см. K-shift в 22.20.5).
                // Счётчик empty_iterations сбрасываем, потому что это
                // штатный проход, а не сбой.
                empty_iterations = 0;
            } else {
                // Нормальный путь: embd был непуст, декодировали,
                // теперь фиксируем его в embd_inp и очищаем embd.
                empty_iterations = 0;
                embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                n_past = static_cast<int>(embd_inp.size());
                embd.clear();
            }

            if (done) break;

            std::string out_token_str = "";
            char out_token_symbol = 0;
            if (llama_start_generation_time == 0.0f)
                llama_start_generation_time = get_current_time_ms();

            {
                // --- 22.20.8. Сохранение сессии ---
                if (!path_session.empty() && need_to_save_session) {
                    need_to_save_session = false;
                    std::lock_guard<std::mutex> lock(g_llama_mutex);
                    active_ctx = g_ctx_llama_atomic.load();
                    if (active_ctx) {
                        llama_state_save_file(active_ctx, path_session.c_str(),
                            session_tokens.data(), session_tokens.size());
                    }
                }

                // --- 22.20.9. Выборка токена ---
                llama_token id = 0;
                {
                    std::lock_guard<std::mutex> lock(g_llama_mutex);
                    active_ctx = g_ctx_llama_atomic.load();
                    if (!active_ctx) { done = true; break; }
                    if (temp != temp_next) {
                        id = llama_sampler_sample(smpl_high_temp, active_ctx, -1);
                        temp = temp_next = params.temp;
                    } else {
                        id = llama_sampler_sample(smpl, active_ctx, -1);
                    }
                }

                // --- 22.20.10. Проверка стоп-строк ---
                {
                    std::string piece = llama_token_to_piece(active_ctx, id);
                    if (token_accumulator.size() > 256) {
                        token_accumulator = token_accumulator.substr(
                            token_accumulator.size() - 256);
                    }
                    token_accumulator += piece;

                    bool stop_found = false;

                    // 1. EOG (конец текста) — глобальный стоп.
                    if (llama_vocab_is_eog(vocab_llama, id)) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] EOG токен %d '%s'\n",
                                    id, piece.c_str());
                        }
                    }

                    // 2. Служебный токен (control) — стоп.
                    if (!stop_found && llama_vocab_is_control(vocab_llama, id)) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] Control токен %d '%s'\n",
                                    id, piece.c_str());
                        }
                    }

                    // 3. Страховка: "<|eo_id|>" — искажённый EOG-токен.
                    if (!stop_found && piece.find("<|eo_id|>") != std::string::npos) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] Неправильный EOG токен: '%s'\n",
                                    piece.c_str());
                        }
                    }

                    // 4. Стоп-строки из пресета — резервный слой.
                    if (!stop_found) {
                        for (const auto& s : preset_d.stop_strings) {
                            size_t pos = token_accumulator.find(s);
                            if (pos != std::string::npos) {
                                token_accumulator = token_accumulator.substr(0, pos);
                                stop_found = true;
                                if (params.verbose) {
                                    fprintf(stderr, "[Generation] Stop string '%s'\n",
                                            s.c_str());
                                }
                                break;
                            }
                        }
                    }

                    if (stop_found) {
                        done = true;
                        break;
                    }
                }

                // --- 22.20.11. Обработка выходного токена ---
                if (id != llama_vocab_eos(vocab_llama)) {
                    embd.push_back(id);
                    out_token_str = llama_token_to_piece(active_ctx, id);

                    // Подстановка плейсхолдеров {0}, {1}, {3}, {5}.
                    size_t pos0 = out_token_str.find("{0}");
                    if (pos0 != std::string::npos) out_token_str.replace(pos0, 3, params.person);
                    size_t pos1 = out_token_str.find("{1}");
                    if (pos1 != std::string::npos) out_token_str.replace(pos1, 3, current_bot_name);
                    size_t pos2 = out_token_str.find("{2}");
                    if (pos2 != std::string::npos) out_token_str.replace(pos2, 3, time_str);
                    size_t pos3 = out_token_str.find("{3}");
                    if (pos3 != std::string::npos) out_token_str.replace(pos3, 3, year_str);
                    size_t pos5 = out_token_str.find("{5}");
                    if (pos5 != std::string::npos) out_token_str.replace(pos5, 3, ymd);

                    std::string cleaned_token = strip_special_tokens(
                        out_token_str, preset_d, antiprompts);

                    // --- 22.20.12. Вывод токена в консоль ---
                    std::string console_token = sanitize_for_console(
                        out_token_str, params.person, current_bot_name, chat_symb);
                    std::string display_str = console_token;

                    bool only_newlines = true;
                    for (char c : display_str) {
                        if (c != '\n' && c != '\r') { only_newlines = false; break; }
                    }
                    bool has_content = false;
                    for (char c : display_str) {
                        if (c != '\n' && c != '\r' && c != ' ' && c != '\t') {
                            has_content = true;
                            break;
                        }
                    }

                    if (only_newlines && !display_str.empty()) {
                        printf("%s", display_str.c_str());
                        fflush(stdout);
                        tokens_in_reply++;
                    }
                    else if (has_content) {
                        if (!bot_prefix_printed) {
                            print_bot_prefix(current_bot_name);
                            bot_prefix_printed = true;
                        }
                        printf("%s", display_str.c_str());
                        fflush(stdout);
                        tokens_in_reply++;
                    }

                    std::string tts_check = cleaned_token;
                    trim(tts_check);
                    if (!tts_check.empty() &&
                        tts_check != current_bot_name &&
                        tts_check != params.person) {
                        text_to_speak += cleaned_token;
                    }

                    // --- 22.20.13. Проверка антипромптов ---
                    {
                        bool antiprompt_hit = false;
                        std::string antiprompt_matched = "";
                        std::string last_output_buffer_ap = text_to_speak;
                        if (last_output_buffer_ap.size() > 256) {
                            last_output_buffer_ap = last_output_buffer_ap.substr(
                                last_output_buffer_ap.size() - 256);
                        }
                        while (last_output_buffer_ap.find("  ") != std::string::npos) {
                            last_output_buffer_ap = string_replace_all(last_output_buffer_ap, "  ", " ");
                        }
                        for (const std::string& antiprompt : antiprompts) {
                            if (last_output_buffer_ap.length() >= antiprompt.length()) {
                                std::string end_part = last_output_buffer_ap.substr(
                                    last_output_buffer_ap.length() - antiprompt.length());
                                if (end_part == antiprompt) {
                                    antiprompt_hit = true;
                                    antiprompt_matched = antiprompt;
                                    break;
                                }
                            }
                        }
                        if (antiprompt_hit) {
                            if (params.min_tokens > 0 && tokens_in_reply < params.min_tokens) {
                                temp_next = params.temp + 0.3f;
                            } else {
                                size_t pos_in_speak = text_to_speak.find(antiprompt_matched);
                                if (pos_in_speak != std::string::npos) {
                                    text_to_speak = text_to_speak.substr(0, pos_in_speak);
                                    trim_spaces_only(text_to_speak);
                                }
                                done = true;
                                break;
                            }
                        }
                    }

                    // --- 22.20.14. Отправка в TTS по предложениям ---
                    tts_smart_buffer += text_to_speak;
                    text_to_speak.clear();
                    bool should_send = false;
                    size_t buf_len = tts_smart_buffer.size();
                    if (buf_len >= 1) {
                        char last_char = tts_smart_buffer[buf_len - 1];
                        if (last_char == '.' || last_char == '!' ||
                            last_char == '?' || last_char == '\n') {
                            should_send = true;
                        }
                        if (last_char == ':' && buf_len >= 2) {
                            char prev = tts_smart_buffer[buf_len - 2];
                            if (std::isspace(static_cast<unsigned char>(prev)) ||
                                buf_len <= 2) {
                                should_send = true;
                            }
                        }
                    }
                    if (!should_send && buf_len >= 2) {
                        char c1 = tts_smart_buffer[buf_len - 2];
                        char c2 = tts_smart_buffer[buf_len - 1];
                        if ((c1 == '.' || c1 == '!' || c1 == '?') &&
                            std::isspace(static_cast<unsigned char>(c2)))
                            should_send = true;
                    }
                    if (!should_send && buf_len > 300 &&
                        g_interrupt_reason.load() == InterruptReason::NONE) {
                        size_t search_end = (buf_len > 100) ? (buf_len - 100) : 0;
                        size_t cut_pos = std::string::npos;
                        int e_open_p = 0, e_open_b = 0, e_open_c = 0;
                        for (size_t ei = 0; ei < search_end; ++ei) {
                            char ec = tts_smart_buffer[ei];
                            if      (ec == '(') e_open_p++;
                            else if (ec == ')') e_open_p = std::max(0, e_open_p - 1);
                            else if (ec == '[') e_open_b++;
                            else if (ec == ']') e_open_b = std::max(0, e_open_b - 1);
                            else if (ec == '{') e_open_c++;
                            else if (ec == '}') e_open_c = std::max(0, e_open_c - 1);
                            else if ((ec == '.' || ec == '!' || ec == '?') &&
                                     e_open_p == 0 && e_open_b == 0 && e_open_c == 0) {
                                cut_pos = ei;
                            }
                        }
                        if (cut_pos != std::string::npos && cut_pos > 150) {
                            should_send = true;
                            std::string to_send_part = tts_smart_buffer.substr(0, cut_pos + 1);
                            tts_smart_buffer = tts_smart_buffer.substr(cut_pos + 1);
                            to_send_part = strip_dialog_prefixes(to_send_part,
                                params.person, current_bot_name, chat_symb);
                            if (!preset_d.stop_sequence.empty())
                                to_send_part = string_replace_all(to_send_part,
                                    preset_d.stop_sequence, "");
                            if (!preset_d.bot_suffix.empty())
                                to_send_part = string_replace_all(to_send_part,
                                    preset_d.bot_suffix, "");
                            trim_spaces_only(to_send_part);

                            int open_p = 0, open_b = 0, open_c = 0;
                            for (char c : to_send_part) {
                                if (c == '(') open_p++; else if (c == ')') open_p--;
                                else if (c == '[') open_b++; else if (c == ']') open_b--;
                                else if (c == '{') open_c++; else if (c == '}') open_c--;
                            }
                            if ((open_p > 0 || open_b > 0 || open_c > 0)) {
                                if (g_verbose_mode.load()) {
                                    fprintf(stderr,
                                        "[TTS] Отложен фрагмент с незакрытой скобкой: '%s'\n",
                                        to_send_part.c_str());
                                }
                                should_send = false;
                            }

                            if (!to_send_part.empty() && should_send) {
                                full_response_text += to_send_part;
                                TtsRequest req;
                                req.text = to_send_part;
                                req.voice = current_voice;
                                req.language = params.language;
                                req.url = params.xtts_url;
                                req.stop_seq = preset_d.stop_sequence;
                                req.bot_sfx = preset_d.bot_suffix;
                                req.user_sfx = preset_d.user_suffix;
                                req.bot_pfx = preset_d.bot_prefix;
                                req.user_pfx = preset_d.user_prefix;
                                req.chat_symb = chat_symb;
                                req.person = params.person;
                                req.bot = current_bot_name;
                                enqueue_tts(std::move(req));
                                update_console_title(current_bot_name, "Speaking...");
                                if (params.sleep_before_xtts) {
                                    int sleep_remaining = params.sleep_before_xtts;
                                    while (sleep_remaining > 0 &&
                                           g_interrupt_reason.load() == InterruptReason::NONE) {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                        sleep_remaining -= 20;
                                    }
                                }
                            }
                            continue;
                        }
                    }
                    if (should_send && !tts_smart_buffer.empty() &&
                        g_interrupt_reason.load() == InterruptReason::NONE) {
                        std::string to_send = tts_smart_buffer;
                        tts_smart_buffer.clear();
                        to_send = strip_dialog_prefixes(to_send,
                            params.person, current_bot_name, chat_symb);
                        if (!preset_d.stop_sequence.empty())
                            to_send = string_replace_all(to_send,
                                preset_d.stop_sequence, "");
                        if (!preset_d.bot_suffix.empty())
                            to_send = string_replace_all(to_send,
                                preset_d.bot_suffix, "");
                        trim_spaces_only(to_send);

                        int open_p = 0, open_b = 0, open_c = 0;
                        for (char c : to_send) {
                            if (c == '(') open_p++; else if (c == ')') open_p--;
                            else if (c == '[') open_b++; else if (c == ']') open_b--;
                            else if (c == '{') open_c++; else if (c == '}') open_c--;
                        }
                        if (open_p > 0 || open_b > 0 || open_c > 0) {
                            if (g_verbose_mode.load()) {
                                fprintf(stderr,
                                    "[TTS] Отложен фрагмент с незакрытой скобкой: '%s'\n",
                                    to_send.c_str());
                            }
                            should_send = false;
                        }

                        if (!to_send.empty() && should_send) {
                            full_response_text += to_send;
                            TtsRequest req;
                            req.text = to_send;
                            req.voice = current_voice;
                            req.language = params.language;
                            req.url = params.xtts_url;
                            req.stop_seq = preset_d.stop_sequence;
                            req.bot_sfx = preset_d.bot_suffix;
                            req.user_sfx = preset_d.user_suffix;
                            req.bot_pfx = preset_d.bot_prefix;
                            req.user_pfx = preset_d.user_prefix;
                            req.chat_symb = chat_symb;
                            req.person = params.person;
                            req.bot = current_bot_name;
                            enqueue_tts(std::move(req));
                            update_console_title(current_bot_name, "Speaking...");
                            if (params.sleep_before_xtts) {
                                int sleep_remaining = params.sleep_before_xtts;
                                while (sleep_remaining > 0 &&
                                       g_interrupt_reason.load() == InterruptReason::NONE) {
                                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                    sleep_remaining -= 20;
                                }
                            }
                        }
                    }

                    // --- 22.20.15. Seqrep — защита от повторов ---
                    if (params.seqrep) {
                        if (utf8_length(last_output_needle) > 25) {
                            last_output_needle = utf8_substr(last_output_needle, 5,
                                utf8_length(last_output_needle) - 5);
                        }
                        last_output_needle += out_token_str;
                        out_token_symbol = out_token_str[out_token_str.size() - 1];
                        if (out_token_symbol == ' ' || out_token_symbol == '.' ||
                            out_token_symbol == ',' || out_token_symbol == '!' ||
                            out_token_symbol == '?') {
                            if (utf8_length(last_output_buffer) > 300 &&
                                utf8_length(last_output_needle) >= 8 &&
                                last_output_buffer.find(last_output_needle) != std::string::npos) {
                                if (params.verbose)
                                    fprintf(stderr, "[Seqrep] Повтор '%s'\n",
                                            last_output_needle.c_str());
                                seqrep_counter++;
                                if (seqrep_counter >= 3) {
                                    if (params.verbose)
                                        fprintf(stderr, "[Seqrep] Стоп после %d срабатываний\n",
                                                seqrep_counter);
                                    done = true;
                                    break;
                                }
                                int symbols_to_delete = utf8_length(last_output_needle);
                                std::vector<llama_token> tokens_to_del =
                                    llama_tokenize(active_ctx, last_output_needle.c_str(), false);
                                int rollback_num = static_cast<int>(tokens_to_del.size());
                                if (rollback_num > 0 && rollback_num <= static_cast<int>(embd_inp.size())) {
                                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                                    n_past = static_cast<int>(embd_inp.size());
                                    n_session_consumed = n_past;
                                    {
                                        std::lock_guard<std::mutex> lock(g_llama_mutex);
                                        active_ctx = g_ctx_llama_atomic.load();
                                        if (active_ctx) {
                                            llama_memory_seq_rm(llama_get_memory(active_ctx), 0,
                                                static_cast<int>(embd_inp.size()), -1);
                                        }
                                    }
                                    if (symbols_to_delete <= utf8_length(text_to_speak)) {
                                        text_to_speak = utf8_substr(text_to_speak, 0,
                                            utf8_length(text_to_speak) - symbols_to_delete);
                                    } else {
                                        text_to_speak = "";
                                    }
                                    last_output_needle = utf8_substr(last_output_needle, 0,
                                        utf8_length(last_output_needle) - symbols_to_delete);
                                    last_output_buffer = utf8_substr(last_output_buffer, 0,
                                        utf8_length(last_output_buffer) - symbols_to_delete);
                                    temp_next = params.temp + 0.3f;
                                }
                            }
                        }
                        if (utf8_length(last_output_buffer) > 1000) {
                            last_output_buffer = utf8_substr(last_output_buffer, 100,
                                utf8_length(last_output_buffer) - 100);
                        }
                        last_output_buffer += out_token_str;
                    }

                    // --- 22.20.16. Loop detection (раз в 5 токенов) ---
                    static int loop_check_counter = 0;
                    loop_check_counter++;
                    bool run_loop_check = (loop_check_counter % 5 == 0);

                    if (run_loop_check) {
                        std::vector<std::string> recent_chars =
                            utf8_split_chars(token_accumulator);
                        const size_t RECENT_N = recent_chars.size();
                        if (RECENT_N > 128) {
                            recent_chars.erase(recent_chars.begin(),
                                               recent_chars.begin() + (RECENT_N - 128));
                        }
                        const size_t N = recent_chars.size();

                        bool pattern_found = false;
                        std::string pattern_str;
                        size_t pattern_count = 0;

                        for (size_t len = 6; len <= 20 && len <= N / 3; ++len) {
                            for (size_t start = 0; start + len * 3 <= N; ++start) {
                                size_t count = 0;
                                size_t pos = start;
                                while (pos + len <= N) {
                                    bool match = true;
                                    for (size_t k = 0; k < len; ++k) {
                                        if (recent_chars[pos + k] !=
                                            recent_chars[start + k]) {
                                            match = false;
                                            break;
                                        }
                                    }
                                    if (match) { count++; pos += len; }
                                    else       { pos++; }
                                }
                                if (count >= 4) {
                                    pattern_found = true;
                                    for (size_t k = 0; k < len; ++k)
                                        pattern_str += recent_chars[start + k];
                                    pattern_count = count;
                                    break;
                                }
                            }
                            if (pattern_found) break;
                        }

                        if (pattern_found) {
                            loop_detection_counter++;
                            tokens_since_last_loop = 0;
                            if (params.verbose) {
                                fprintf(stderr, "[Generation] Повтор '%s' (%zu)\n",
                                        pattern_str.c_str(), pattern_count);
                            }
                            if (loop_detection_counter >= 3) {
                                if (params.verbose) {
                                    fprintf(stderr,
                                        "[Generation] Стоп после %d срабатываний\n",
                                        loop_detection_counter);
                                }
                                done = true;
                                break;
                            }
                            if (loop_detection_counter >= 2) {
                                temp_next = params.temp + 0.3f;
                            }
                        } else {
                            tokens_since_last_loop++;
                            if (tokens_since_last_loop > 20) {
                                loop_detection_counter = 0;
                            }
                        }
                    }

                    // --- 22.20.17. Паттерны циклов ---
                    if (out_token_str.find("ineline") != std::string::npos ||
                        out_token_str.find("olen") != std::string::npos ||
                        out_token_str.find("elin") != std::string::npos ||
                        out_token_str.find("linelin") != std::string::npos) {
                        if (params.verbose)
                            fprintf(stderr, "[Generation] Паттерн цикла '%s'\n",
                                    out_token_str.c_str());
                        done = true;
                        break;
                    }

                    // --- 22.20.18. Проверка на повтор символов ---
                    if (out_token_str.length() > 10) {
                        bool all_same = true;
                        for (size_t i = 1; i < out_token_str.length(); ++i) {
                            if (out_token_str[i] != out_token_str[0]) {
                                all_same = false; break;
                            }
                        }
                        if (all_same) {
                            if (params.verbose)
                                fprintf(stderr, "[Generation] Повтор символов '%s'\n",
                                        out_token_str.c_str());
                            done = true;
                            break;
                        }
                    }
                }
            }
        }
        // ====================================================================
        // КОНЕЦ ЦИКЛА ГЕНЕРАЦИИ
        // ====================================================================

        // ====================================================================
        // 22.21. ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ
        // ====================================================================
        if (!tts_smart_buffer.empty()) {
            int open_p = 0, open_b = 0, open_c = 0;
            for (char c : tts_smart_buffer) {
                if (c == '(') open_p++; else if (c == ')') open_p--;
                else if (c == '[') open_b++; else if (c == ']') open_b--;
                else if (c == '{') open_c++; else if (c == '}') open_c--;
            }
            if (open_p <= 0 && open_b <= 0 && open_c <= 0) {
                text_to_speak = tts_smart_buffer + text_to_speak;
            } else {
                if (params.verbose) {
                    fprintf(stderr,
                        "[TTS] Финальный фрагмент с незакрытой скобкой, отброшен: '%s'\n",
                        tts_smart_buffer.c_str());
                }
            }
            tts_smart_buffer.clear();
        }

        InterruptReason local_reason = g_interrupt_reason.load();
        bool was_interrupted_final = (local_reason != InterruptReason::NONE ||
                                       g_interrupt_processed.load());
        if (was_interrupted_final) {
            text_to_speak = "";
            if (local_reason != InterruptReason::VAD_SPEECH) audio.clear();
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_bot_state.store(BotState::IDLE);
        }

        if (!was_interrupted_final) {
            std::string clean_full = full_response_text;
            if (!text_to_speak.empty()) clean_full += text_to_speak;
            trim_spaces_only(clean_full);
            {
                std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                g_last_tts_text = clean_full;
            }

            if (!text_to_speak.empty()) {
                if (g_interrupt_reason.load() == InterruptReason::NONE) {
                    TtsRequest req;
                    req.text = text_to_speak;
                    req.voice = current_voice;
                    req.language = params.language;
                    req.url = params.xtts_url;
                    req.stop_seq = preset_d.stop_sequence;
                    req.bot_sfx = preset_d.bot_suffix;
                    req.user_sfx = preset_d.user_suffix;
                    req.bot_pfx = preset_d.bot_prefix;
                    req.user_pfx = preset_d.user_prefix;
                    req.chat_symb = chat_symb;
                    req.person = params.person;
                    req.bot = current_bot_name;
                    enqueue_tts(std::move(req));
                    if (params.sleep_before_xtts) {
                        int sleep_remaining = params.sleep_before_xtts;
                        while (sleep_remaining > 0 &&
                               g_interrupt_reason.load() == InterruptReason::NONE) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                            sleep_remaining -= 20;
                        }
                    }
                }
            }

            if (bot_prefix_printed) {
                append_status_to_current_line("tts", C_TTS);
            } else {
                std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
                printf("\n");
                fflush(stdout);
                g_need_blank_before_next = true;
            }
            bot_prefix_printed = false;
        } else {
            bot_prefix_printed = false;
        }

        if (was_interrupted_final) {
            g_cancel_tts_requests.store(false);
            { std::string dummy2; allow_xtts_file(dummy2, 1); }
        }
        if (!was_interrupted_final) audio.clear();

        // --- 22.22. Статистика (verbose) ---
        llama_end_time = get_current_time_ms();
        if (params.verbose) {
            llama_time_input = llama_start_generation_time - llama_start_time;
            llama_time_output = llama_end_time - llama_start_generation_time;
            llama_time_total = llama_end_time - llama_start_time;
            printf("\n[Context: %d/%d. Tokens: %d in + %d out. Input %.3f s + output %.3f s = total: %.3f s]\n",
                n_past, n_ctx, input_tokens_count, new_tokens,
                llama_time_input, llama_time_output, llama_time_total);
            float input_speed = (llama_time_input > 0.001f) ?
                static_cast<float>(input_tokens_count) / llama_time_input : 0.0f;
            float output_speed = (llama_time_output > 0.001f) ?
                static_cast<float>(new_tokens) / llama_time_output : 0.0f;
            printf("[Speed: input %.2f t/s + output %.2f t/s]\n",
                input_speed, output_speed);
        }

        g_interrupt_reason.store(InterruptReason::NONE);
        g_interrupt_processed.store(false);
        g_bot_state.store(BotState::IDLE);
        llama_start_generation_time = 0.0;
        { std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex); g_hotkey_pressed = ""; }

        update_console_title(params.bot_name, "Listening");
        fflush(stdout);
    }
    // ========================================================================
    // КОНЕЦ ОСНОВНОГО ЦИКЛА while (is_running)
    // ========================================================================

    // ========================================================================
    // 23. ЗАВЕРШЕНИЕ РАБОТЫ
    // ========================================================================
    if (params.verbose) printf("Cleaning up TTS threads...\n");
    g_shutting_down.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (audio_thread.joinable()) {
        for (int i = 0; i < 20 && g_audio_thread_running.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        audio_thread.join();
        if (params.verbose) printf("[AudioInput] Thread stopped\n");
    }

    g_tts_worker_running.store(false);
    g_tts_queue_cv.notify_all();
    if (tts_worker.joinable()) tts_worker.join();

    audio.pause();
    audio.clear();

    keyboard_input_running.store(false);
    if (input_thread.joinable()) input_thread.join();

    g_shortcut_thread_running.store(false);
    if (shortcut_thread.joinable()) shortcut_thread.join();

    audio.pause();

    whisper_print_timings(ctx_wsp);
    if (ctx_llama) llama_perf_context_print(ctx_llama);

    g_ctx_llama_atomic.store(nullptr);

    whisper_free(ctx_wsp);

    if (smpl) {
        llama_perf_sampler_print(smpl);
        llama_sampler_free(smpl);
    }
    if (smpl_high_temp) llama_sampler_free(smpl_high_temp);

    llama_batch_free(batch);
    llama_free(ctx_llama);
    llama_model_free(model_llama);
    llama_backend_free();

    if (g_log_enabled.load()) {
        log_line("=== session ended ===");
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (g_log_file.is_open()) g_log_file.close();
    }

    if (params.verbose) printf("Goodbye!\n");
    return 0;
}


// ============================================================================
// 24. ТОЧКИ ВХОДА
// ============================================================================
#if _WIN32
// Windows: используем wmain, чтобы получить argv в UTF-16. Затем
// конвертируем в UTF-8 и вызываем run.
int wmain(int argc, const wchar_t** argv_UTF16LE) {
    console::init(true, true);
    atexit([] { console::cleanup(); });

    std::vector<std::string> buffer(argc);
    std::vector<char*> argv_UTF8(argc);
    for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = &buffer[i][0];
    }
    return run(argc, argv_UTF8.data());
}
#else
// POSIX: обычный main. argv уже в UTF-8.
int main(int argc, const char** argv_UTF8) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl" << std::endl;
        return 1;
    }
    const char* verbose_env = std::getenv("TALK_LLAMA_VERBOSE");
    if (verbose_env && (std::string(verbose_env) == "1" ||
                        std::string(verbose_env) == "true"))
        g_verbose_mode.store(true);
    console::init(true, true);
    atexit([] { console::cleanup(); curl_global_cleanup(); });
    return run(argc, argv_UTF8);
}
#endif