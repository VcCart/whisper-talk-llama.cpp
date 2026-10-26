// ============================================================================
// talk-llama.cpp — голосовой ассистент: Whisper + LLaMA + XTTS.
//
// Назначение:
//   Голосовой диалог с локальной моделью LLaMA (Saiga-YandexGPT 8B).
//   Распознавание речи — Whisper. Синтез — XTTS (HTTP). Управление —
//   микрофон (VAD), клавиатура, горячие клавиши.
//
// Потоки:
//   run()                    — основной цикл: ввод, команды, генерация.
//   audio_input_thread_func  — микрофон, VAD, Whisper, barge-in.
//   input_thread_func        — чтение строк с клавиатуры.
//   keyboard_shortcut_func   — опрос горячих клавиш (Windows).
//   tts_worker_func          — очередь TtsRequest → HTTP к XTTS.
//
// Инварианты:
//   • TtsRequest = один HTTP-запрос. Разбиение — до enqueue (см. 18.2).
//   • Прерывание ставит g_interrupt_reason и НЕ трогает TTS-очередь.
//     Чистка очереди и семафора — только в 23.17.1 (единая точка).
//   • «Стоп» — отдельный случай: g_tts_muted + ожидание g_tts_busy.
//   • strip_dialog_prefixes удаляет «Имя:» только в начале текста.
//   • sanitize_user_text удаляет «Имя:» в начале КАЖДОЙ строки user turn.
//   • Консоль и text_to_speak видят один и тот же visible_token.
//   • ctx_llama под std::shared_mutex: читатели — shared_lock,
//     reset/rebuild — unique_lock.
//   • get_current_time_ms() возвращает double: при uptime > ~4.6 ч float
//     теряет миллисекунды, что критично для auto-continue.
//   • Регенерация идёт через отдельный сэмплер smpl_regen (temp+0.5f),
//     чтобы ответ гарантированно отличался от предыдущего.
// ============================================================================


// ============================================================================
// 1. ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
// ============================================================================
// Назначение: системные заголовки, движки (Whisper, LLaMA), HTTP-клиент,
//   платформенные API.
// Инварианты: порядок включения не важен; console.cpp включается как .cpp —
//   это локальная реализация, без отдельной единицы трансляции.
// Используется: везде.

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
#include <shared_mutex>
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

// --- 1.2. Движки: Whisper + LLaMA ---
#include "common-sdl.h"
#include "common-whisper.h"
#include "whisper.h"
#include "llama.h"
#include "llama-chat.h"

// --- 1.3. Пользовательские модули ---
#include "console.h"
#include "console.cpp"

// --- 1.4. Сеть (XTTS, google-сервер) ---
#include <curl/curl.h>
#include "json.hpp"

// --- 1.5. Платформенные заголовки ---
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
// Назначение: ANSI-коды для раскраски реплик, статусов, команд.
// Инварианты: C_USER / C_BOT — только цвет имени; статусы — серый/циан/красный.
// Используется: print_replica, print_bot_prefix, print_user_prompt,
//   append_status_to_current_line.

#define C_USER          "\033[32m"
#define C_BOT           "\033[1;33m"
#define C_RESET         "\033[0m"
#define C_STATUS_TTS    "\033[90m"
#define C_STATUS_CMD    "\033[36m"
#define C_STATUS_STOP   "\033[31m"
#define C_STATUS_SPEECH "\033[33m"

#define C_TTS     C_STATUS_TTS
#define C_CMD     C_STATUS_CMD
#define C_STOP    C_STATUS_STOP


// ============================================================================
// 3. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И СИНХРОНИЗАЦИЯ
// ============================================================================
// Назначение: единое состояние приложения — бот, прерывания, TTS, ввод,
//   горячие клавиши, VAD, контекст LLaMA, сессия, флаги.
// Инварианты:
//   • Все флаги прерывания — std::atomic.
//   • g_ctx_llama_atomic защищён g_llama_rw_mutex (shared для чтения,
//     unique для reset/rebuild).
//   • g_saved_generation_mutex защищает g_saved_n_past / g_saved_embd_inp /
//     g_saved_was_interrupted / g_saved_timestamp.
//   • g_tts_queue_mutex защищает g_tts_queue; g_tts_queue_cv — для воркера.
// Используется: везде.

// --- 3.1. Состояние бота ---
// IDLE — ждём ввода; GENERATING — генерируем ответ;
// INTERRUPTED — прерывание обрабатывается.
enum class BotState : uint8_t {
    IDLE = 0,
    GENERATING,
    INTERRUPTED
};
std::atomic<BotState> g_bot_state{ BotState::IDLE };

// --- 3.2. Причина прерывания ---
enum class InterruptReason : uint8_t {
    NONE = 0,
    VAD_SPEECH,     // barge-in: пользователь заговорил
    HOTKEY_STOP,    // Ctrl+Space, либо внутренняя синхронизация
    HOTKEY_ALT,     // Alt (push-to-talk) — не прерывание, но слот есть
    MANUAL_STOP     // команда «стоп»
};
std::atomic<InterruptReason> g_interrupt_reason{ InterruptReason::NONE };
std::atomic<bool> g_interrupt_processed{ false };
std::atomic<bool> g_shutting_down{ false };
std::atomic<bool> g_cancel_tts_requests{ false };

// --- 3.3. Жёсткий запрет XTTS ---
// true — TTS-воркер не отправляет запросы, семафор = 0.
std::atomic<bool> g_tts_muted{ false };

// --- 3.4. Занятость TTS-воркера ---
// true — воркер выполняет HTTP-запрос; используется для ожидания.
std::atomic<bool> g_tts_busy{ false };

// --- 3.5. Атомарный указатель на ctx_llama ---
// Читается под shared_lock, меняется под unique_lock (reset/rebuild).
std::atomic<llama_context*> g_ctx_llama_atomic{ nullptr };

// --- 3.6. Файл-семафор XTTS ---
// Временный файл, в который пишется 1/0. XTTS-сервер читает его, чтобы
// не проигрывать аудио, пока пользователь говорит (barge-in).
std::string g_xtts_control_file_path = "";
std::mutex g_xtts_control_mutex;
bool g_last_semaphore_value = true;
bool g_last_semaphore_initialized = false;

// --- 3.7. Ввод с клавиатуры ---
std::queue<std::string> input_queue;
std::mutex input_mutex;
std::atomic<bool> keyboard_input_running{ true };

// --- 3.8. Горячие клавиши ---
std::string g_hotkey_pressed = "";
std::mutex g_hotkey_pressed_mutex;
std::atomic<bool> g_shortcut_thread_running{ true };

// --- 3.9. Read-Write мьютекс для ctx_llama ---
std::shared_mutex g_llama_rw_mutex;

// --- 3.10. Последний текст, отправленный в TTS ---
// Нужен для команды «повтори».
std::string g_last_tts_text = "";
std::mutex g_last_tts_mutex;

// --- 3.11. Verbose-режим ---
std::atomic<bool> g_verbose_mode{ false };

// --- 3.12. Режим печати спецтокенов ---
std::atomic<bool> g_print_special{ false };

// --- 3.13. Мьютекс консольного вывода ---
// recursive: print_replica может вызываться из вложенных лямбд.
std::recursive_mutex g_console_mutex;

// --- 3.14. Буфер накопленного текста от Whisper ---
// Сегменты речи, ещё не отправленные в LLaMA.
std::string g_accumulated_text;
std::mutex g_text_accumulator_mutex;

// --- 3.15. Буфер отображаемого текста ---
// Последний распознанный сегмент — для отладки.
std::string g_display_text;
std::mutex g_display_mutex;

// --- 3.16. Мягкий лимит токенов ---
// При превышении — принудительная отправка накопленного текста в LLaMA.
std::atomic<int>  g_soft_limit_tokens{ 2560 };
std::atomic<bool> g_audio_thread_running{ false };

// --- 3.17. Ожидающий запрос в LLaMA от аудио-потока ---
std::atomic<bool> g_pending_llm_request{ false };
std::string g_pending_llm_text;
std::mutex g_pending_llm_mutex;

// --- 3.18. Флаг принудительного сброса VAD ---
std::atomic<bool> g_force_vad_reset{ false };

// --- 3.19. Флаг сброса контекста ---
std::atomic<bool> g_reset_in_progress{ false };
std::mutex g_reset_mutex;
std::condition_variable g_reset_cv;

// --- 3.20. Whisper-промпт ---
// Bias для распознавания (имена участников).
std::string g_whisper_prompt = "";
std::mutex g_whisper_prompt_mutex;

// --- 3.21. Флаг «нужна пустая строка перед следующей парой» ---
static bool g_need_blank_before_next = false;

// --- 3.22. Флаг «приглашение пользователя уже напечатано» ---
static bool g_initial_prompt_printed = false;

// --- 3.23. Состояние генерации для auto-continue ---
// double — чтобы при uptime > ~4.6 ч не терять миллисекунды.
int g_saved_n_past = 0;
std::vector<llama_token> g_saved_embd_inp;
double g_saved_timestamp = 0.0;
bool g_saved_was_interrupted = false;
std::mutex g_saved_generation_mutex;
std::atomic<bool> g_pending_continue_request{ false };

// --- 3.23.1. Флаг «следующий ход — regen» ---
// При regen откатываем embd_inp до user turn и восстанавливаем text_heard,
// но чтобы получить ДРУГОЙ ответ, первый токен берём из smpl_regen
// (temp = params.temp + 0.5f). Флаг сбрасывается в 23.17.8.2 после
// первого токена.
std::atomic<bool> g_regen_pending{ false };

// --- 3.24. Константы ---
const std::string LEO_NAME  = "Лео";
const std::string LEO_VOICE = "Лео";
const std::string DEFAULT_CHAT_SYMB = ": ";

// --- 3.25. Именованные константы ---
constexpr int    N_KEEP_MIN             = 512;   // минимум system-токенов при K-shift
constexpr int    N_KEEP_SLACK           = 64;    // запас над размером промпта
constexpr size_t MAX_PAST_PREV_SIZE     = 100;   // история n_past для regen/delete
constexpr int    MAX_EMPTY_ITERATIONS   = 5;     // сбоев decode подряд
constexpr int    INTERRUPT_CHECK_MS     = 50;    // опрос прерываний
constexpr int    DEFAULT_INTERRUPT_MS   = 100;   // порог barge-in по умолчанию
constexpr int    LOOP_CHECK_INTERVAL    = 5;     // каждые N токенов — loop-check
constexpr int    STOP_COMMAND_MAX_WORDS = 4;     // «стоп» не длиннее 4 слов
constexpr int    STOP_COMMAND_MAX_CHARS = 32;    // и 32 символов
constexpr int    K_SHIFT_DIVISOR        = 8;     // удаляем 1/8 хвоста при K-shift
// Максимальная длина пользовательского ввода в символах. Защита от
// случайной вставки огромного текста (например, всего предыдущего
// диалога) в user turn. 2000 символов ≈ 600 токенов — с запасом для
// любой голосовой/печатной реплики.
constexpr size_t MAX_USER_INPUT_CHARS   = 2000;

// --- 3.26. Структура TTS-запроса ---
// Один TtsRequest = один HTTP-запрос. Разбиение — до enqueue.
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

// --- 3.27. Счётчик транскрипций ---
std::atomic<int> g_transcription_count{0};

// --- 3.28. Флаг «идёт обработка команды» ---
std::atomic<bool> g_command_in_progress{ false };

// --- 3.29. Заголовок консольного окна ---
static std::string g_last_console_title = "";
static std::mutex g_console_title_mutex;


// ============================================================================
// 4. ПРЕДВАРИТЕЛЬНЫЕ ОБЪЯВЛЕНИЯ
// ============================================================================
// Назначение: функции, определённые ниже, но используемые раньше.
// Инварианты: сигнатуры должны совпадать с определениями.
// Используется: в лямбдах и хелперах, объявленных до определений.

std::string getTempDir();
int utf8_length(const std::string& str);
std::string utf8_substr(const std::string& str, unsigned int start, unsigned int leng);
std::string normalize_template_token(const std::string& s);
std::string normalize_json_key(const std::string& key);

static void allow_xtts_file(std::string& path, int xtts_play_allowed);
static void print_replica(const std::string& color, const std::string& name,
                          const std::string& text, const std::string& status = "");
static void print_bot_prefix(const std::string& bot_name);
static void print_user_prompt(const std::string& person);


// ============================================================================
// 5. СТРОКОВЫЕ УТИЛИТЫ (UTF-8-aware)
// ============================================================================
// Назначение: trim, lowercase, подстроки, split, замена, детекторы границ
//   слов — всё с учётом UTF-8 (русские буквы — 2 байта).
// Инварианты: utf8_length/utf8_substr не режут многобайтные символы;
//   utf8_is_word_char считает кириллицу «словом», NBSP/тире — нет.
// Используется: везде, где есть текст.

namespace {

// --- 5.1. trim / ltrim / rtrim / trim_spaces_only ---
// trim удаляет все пробельные (включая \n, \r, \f, \v, 0xA0).
// trim_spaces_only — только ' ', '\t', 0xA0 (не трогает переводы строк).

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
// IsPunctuationMark — ASCII-пунктуация (для ParseCommandAndGetKeyword).
// StripPunctuationMarks — удаляет все такие символы из строки.
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


// --- 5.3. Lower-case (UTF-8-aware) ---
// Понимает кириллицу (U+0410..U+042F → U+0430..U+044F, U+0401 → U+0451).
// Остальные многобайтные символы копирует как есть.
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
            if (c == 0xD0 && c2 == 0x81) {
                // Ё → ё
                result += static_cast<char>(0xD1);
                result += static_cast<char>(0x91);
            }
            else if (c == 0xD0 && c2 >= 0x90 && c2 <= 0x9F) {
                // А..П → а..п
                result += static_cast<char>(0xD0);
                result += static_cast<char>(c2 + 0x20);
            }
            else if (c == 0xD0 && c2 >= 0xA0 && c2 <= 0xAF) {
                // Р..Я → р..я
                result += static_cast<char>(0xD1);
                result += static_cast<char>(c2 - 0x20);
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


// --- 5.4. Замена всех вхождений ---
// Защита от бесконечного цикла: если to содержит from — выходим после
// первой замены (иначе можно зациклиться).
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


// --- 5.5. Удаление завершающих UTF-8 символов ---
// Удаляет с конца inputString любые символы, входящие в targetCharacters
// (как множество UTF-8 символов). Кэш разбора targetCharacters — под mutex.
std::string RemoveTrailingCharactersUtf8(const std::string& inputString,
    const std::string& targetCharacters) {
    if (inputString.empty() || targetCharacters.empty()) return inputString;

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


// --- 5.6. Длина в UTF-8 символах ---
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
// start/leng — в символах, не в байтах. Границы не режут многобайтные.
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


// --- 5.8. Разбиение на UTF-8 символы ---
// Каждый элемент вектора — один UTF-8 символ (1–4 байта).
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


// --- 5.10. «Словесный» ли символ ---
// Буквы/цифры — слово. NBSP, длинное/короткое тире — нет.
static bool utf8_is_word_char(const std::string& ch) {
    if (ch.empty()) return false;
    unsigned char c = static_cast<unsigned char>(ch[0]);
    if (c < 0x80) {
        return std::isalnum(c) != 0;
    }
    if (ch == "\xC2\xA0") return false;
    if (ch == "\xE2\x80\x94") return false;
    if (ch == "\xE2\x80\x93") return false;
    return true;
}


// --- 5.10.1. UTF-8-aware граница слова ---
// true — перед позицией pos стоит не-слово (или начало строки).
static bool is_utf8_word_boundary(const std::string& haystack, size_t pos) {
    if (pos == 0) return true;
    if (pos > haystack.size()) return true;

    size_t char_start = pos - 1;
    while (char_start > 0 &&
        (static_cast<unsigned char>(haystack[char_start]) & 0xC0) == 0x80) {
        char_start--;
    }
    std::string left_char = haystack.substr(char_start, pos - char_start);
    return !utf8_is_word_char(left_char);
}


// --- 5.11. Нормализация JSON-ключа и шаблонного токена ---
// normalize_json_key — trim пробелов.
// normalize_template_token — trim + склейка «<| token |>» в «<|token|>».
std::string normalize_json_key(const std::string& key) {
    std::string result = key;
    trim(result);
    return result;
}

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


// --- 5.12. Парсинг списка float ---
// «0.5,0.3,0.2» → vector<float>. Ошибка — пустой вектор.
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


// --- 5.13. Тип KV-кэша из строки ---
static ggml_type parse_cache_type_str(const std::string& s) {
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


// --- 5.14. Монотонное время в секундах ---
// double, а не float: при uptime > ~4.6 ч float теряет миллисекунды.
// Для auto-continue (continue_max_ms до 10000) это критично.
double get_current_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    ) / 1000.0;
}


// --- 5.15. Директория для временных файлов ---
// Сначала std::filesystem, затем платформенный fallback.
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


// --- 5.16. Файл-семафор XTTS ---
// write_semaphore_instant — атомарная запись '1'/'0' с fsync/FlushFileBuffers.
// allow_xtts_file — публичная точка: кэширует последнее значение, чтобы
// не писать файл каждый тик audio-потока.
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

static void allow_xtts_file(std::string & path, int xtts_play_allowed) {
    std::lock_guard<std::mutex> lock(g_xtts_control_mutex);

    if (g_xtts_control_file_path.empty()) {
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

    path = g_xtts_control_file_path;

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


// --- 5.17. Токенизация ---
// llama_tokenize_ex — двухпроходная обёртка над llama_tokenize:
//   первый вызов может вернуть отрицательное число (сколько нужно),
//   второй — реальную длину.
// llama_token_to_piece — обёртка над llama_token_to_piece с ресайзом.
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


// --- 5.18. Санитизация пользовательского текста ---
// Назначение: превратить user turn в безопасный для модели текст.
// Удаляет:
//   • спецтокены <|...|> — чтобы модель не воспринимала их как
//     структурные маркеры диалога;
//   • начала строк "Имя: " — чтобы случайная вставка диалога
//     (например, "Адам: Привет\nЭмма: Как дела?") не воспринималась
//     моделью как готовая история и не продолжалась.
//
// Имена передаются из run() через глобалы: g_sanitize_person,
// g_sanitize_bot. Если пусты — только спецтокены.
// Инварианты: удаляем «Имя:» только в НАЧАЛЕ строки. «Привет, Адам:»
// остаётся как есть.
// Используется: 23.15 при формировании user turn.

static std::string g_sanitize_person;
static std::string g_sanitize_bot;

static std::string sanitize_user_text(const std::string& text) {
    std::string result = text;
    try {
        static const std::regex re_special_token(R"(<\|[^|]*\|>)", std::regex::ECMAScript);
        result = std::regex_replace(result, re_special_token, "");
    } catch (const std::regex_error&) {
        result = string_replace_all(result, "<|start_header_id|>", "");
        result = string_replace_all(result, "<|end_header_id|>", "");
        result = string_replace_all(result, "<|eot_id|>", "");
        result = string_replace_all(result, "<|eo_id|>", "");
    }

    // Удаляем начала строк вида "Имя: " (только в начале строки).
    if (!g_sanitize_person.empty() || !g_sanitize_bot.empty()) {
        std::string out;
        out.reserve(result.size());
        size_t i = 0;
        const size_t n = result.size();
        bool at_line_start = true;
        while (i < n) {
            if (at_line_start) {
                bool removed = false;
                for (const std::string* name : {&g_sanitize_person, &g_sanitize_bot}) {
                    if (name->empty()) continue;
                    std::string with_symb  = *name + ": ";
                    std::string with_colon = *name + ":";
                    if (result.compare(i, with_symb.size(), with_symb) == 0) {
                        i += with_symb.size();
                        removed = true;
                        break;
                    }
                    if (result.compare(i, with_colon.size(), with_colon) == 0) {
                        i += with_colon.size();
                        removed = true;
                        break;
                    }
                }
                if (removed) {
                    at_line_start = true;
                    continue;
                }
            }
            char c = result[i];
            out += c;
            i++;
            at_line_start = (c == '\n' || c == '\r');
        }
        result = std::move(out);
    }

    return result;
}

// ============================================================================
// 6. АУДИОУТИЛИТЫ (VAD)
// ============================================================================
// Назначение: энергетический VAD. Один проход по PCM-буферу: high-pass
//   фильтр + средняя энергия за последние last_ms.
//
// Возвращает:
//   0 — нет речи / речь продолжается;
//   1 — начало речи (энергия последних сэмплов > vad_start_thold);
//   2 — конец речи (энергия последних сэмплов < vad_thold * energy_all).
//
// Инварианты:
//   • vad_simple_int возвращает 0, если в буфере меньше last_ms — иначе
//     деление на n_samples_last даст мусор.
//   • high_pass_filter_local модифицирует буфер in-place (вызывающий
//     должен передавать копию, если оригинал нужен дальше).
// Используется:
//   • в audio_input_thread_func — сегментация речи;
//   • в основном цикле генерации — barge-in (быстрый прерыватель).

static void high_pass_filter_local(std::vector<float> & data,
                                   float cutoff,
                                   float sample_rate) {
    constexpr float PI_F = 3.14159265358979323846f;
    const float rc    = 1.0f / (2.0f * PI_F * cutoff);
    const float dt    = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];
    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

static int vad_simple_int(std::vector<float> & pcmf32,
                          int   sample_rate,
                          int   last_ms,
                          float vad_thold,
                          float freq_thold,
                          bool  verbose,
                          float vad_start_thold = 0.000270f) {
    const int n_samples      = static_cast<int>(pcmf32.size());
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        return 0;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter_local(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);
        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr,
                "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n",
                __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (vad_start_thold && energy_last > vad_start_thold) {
        if (verbose) printf("[speech started!]: energy_last > %f\n", vad_start_thold);
        return 1;
    }

    if (energy_last > vad_thold * energy_all) {
        return 0;
    }

    if (verbose) printf("[speech end]: energy_last < %f\n", vad_thold * energy_all);
    return 2;
}


// ============================================================================
// 7. ФИЛЬТР ЛОГОВ WHISPER
// ============================================================================
// Назначение: отсечь информационные сообщения Whisper, оставить только
//   ошибки.
// Инварианты: пропускаем только GGML_LOG_LEVEL_ERROR; null-text игнорируем.
// Используется: whisper_log_set(whisper_log_filtered, nullptr) в 22.6.

static void whisper_log_filtered(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (text == nullptr) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}


// ============================================================================
// 8. ШАБЛОНЫ И ВСТРОЕННЫЕ ПРОМПТЫ
// ============================================================================
// Назначение: few-shot шаблоны диалога, fallback-личность, Whisper-prompt.
// Инварианты:
//   • Шаблон разговора — это ФОРМАТ, не личность. Подставляется {0} = person,
//     {1} = bot_name.
//   • Fallback-личность используется, только если --prompt-file не задан
//     или не открылся.
//   • Whisper-prompt — bias для распознавания имён, не системный промпт.
// Используется: 22.4 (fallback), 22.12 (whisper), 22.19.1 (шаблон).

// --- 8.1. Шаблон разговора (few-shot, формат "Имя: текст") ---
static const std::string k_conv_template_ru =
    "{0}: Привет.\n"
    "{1}: Привет. Как ты?\n"
    "{0}: Нормально. Что делаешь?\n"
    "{1}: Да так, домашние дела. А ты?\n"
    "{0}: Который час?\n"
    "{1}: Около трёх. А что, куда-то опаздываешь?\n"
    "{0}: Что надеть на встречу?\n"
    "{1}: Тёмные джинсы и рубашка. Если официально — добавь пиджак.\n";

static const std::string k_conv_template_en =
    "{0}: Hi.\n"
    "{1}: Hi. How are you?\n"
    "{0}: Fine. What are you doing?\n"
    "{1}: Just some chores. You?\n"
    "{0}: What time is it?\n"
    "{1}: Around three. Are you late for something?\n"
    "{0}: What should I wear to a meeting?\n"
    "{1}: Dark jeans and a shirt. Add a jacket if it's formal.\n";

// --- 8.2. Fallback-личность (если --prompt-file не задан) ---
static const std::string k_fallback_persona_ru =
    "{0} разговаривает с {1}. {1} отвечает живо и коротко, "
    "без списков, без скобок и без фигурных меток.";

static const std::string k_fallback_persona_en =
    "{0} is talking with {1}. {1} replies briefly and lively, "
    "no lists, no brackets, no curly-brace labels.";

// --- 8.3. Whisper-prompt (bias для распознавания, не системный промпт) ---
static const std::string k_whisper_prompt_ru = "{0}, {1}.";
static const std::string k_whisper_prompt_en = "A conversation between {0} and {1}.";


// ============================================================================
// 9. СТРУКТУРА ПАРАМЕТРОВ
// ============================================================================
// Назначение: все настройки, которые приходят из argv или имеют дефолты.
// Инварианты: auto_* флаги отключаются, если пользователь задал
//   соответствующее значение явно (n_predict, batch_size, min_tokens,
//   n_keep).
// Используется: парсер 10.2, run() и все хелперы.

struct whisper_params {
    // --- 9.1. Производительность ---
    int32_t n_threads = 18;
    int32_t n_gpu_layers = 30;
    bool use_gpu = true;
    bool flash_attn = false;
    bool flash_attn_disable = false;
    int main_gpu = 0;
    std::string split_mode = "none";
    std::vector<float> tensor_split;

    // --- 9.2. Аудиозахват ---
    int32_t voice_ms = 10000;
    int32_t capture_id = -1;

    // --- 9.3. Энергетический VAD ---
    float vad_thold = 0.50f;
    float vad_start_thold = 0.000120f;
    float vad_last_ms = 800.0f;
    float freq_thold = 90.0f;
    int32_t interrupt_ms = DEFAULT_INTERRUPT_MS;

    // --- 9.4. LLaMA ---
    int32_t ctx_size = 8096;
    int32_t batch_size = 64;
    int32_t n_predict = 512;
    int32_t min_tokens = 0;
    float temp = 0.60f;
    int32_t top_k = 40;
    float top_p = 0.88f;
    float min_p = 0.0f;
    float repeat_penalty = 1.10f;
    int repeat_last_n = 128;
    int n_keep = N_KEEP_MIN;
    bool auto_n_keep = true;
    bool allow_newline = true;
    bool seqrep = true;

    // --- 9.5. Автонастройка ---
    bool auto_n_predict   = true;
    bool auto_batch_size  = true;
    bool auto_min_tokens  = true;
    std::string cache_type_k = "";
    std::string cache_type_v = "";

    // --- 9.6. TTS ---
    std::string xtts_voice = "Эмма";
    std::string xtts_url = "http://localhost:8020/";
    int sleep_before_xtts = 400;
    int split_after = 0;
    bool multi_chars = false;

    // --- 9.7. Имена и пол ---
    std::string person = "Адам";
    std::string person_gender = "";   // "male" | "female" | ""
    std::string bot_name = "Эмма";
    std::string bot_gender = "";      // "male" | "female" | ""
    std::string wake_cmd = "";

    // --- 9.8. Язык ---
    std::string language = "ru";
    bool translate = false;

    // --- 9.9. Модели ---
    std::string model_wsp = "";
    std::string model_llama = "";

    // --- 9.10. Флаги ---
    bool print_special = false;
    bool print_energy = false;
    bool verbose_prompt = false;
    bool verbose = false;
    bool push_to_talk = false;
    std::string google_url = "http://localhost:8003/";
    std::string prompt = "";
    std::string alt_prompt_file = "";
    std::string instruct_preset = "";
    std::string path_session = "";
    std::string stop_words = "";

    // --- 9.11. Автопродолжение ---
    bool auto_continue = true;
    int32_t continue_max_ms = 10000;

    // --- 9.12. Instruct-пресет ---
    std::map<std::string, std::string> instruct_preset_data = {
        {"system_prompt_prefix", ""},
        {"system_prompt_suffix", ""},
        {"user_message_prefix", ""},
        {"user_message_suffix", ""},
        {"bot_message_prefix", ""},
        {"bot_message_suffix", ""},
        {"stop_sequence", ""}
    };

    // --- 9.13. ChatML-шаблон ---
    // Ручная сборка из пресета (22.15). llm_chat_apply_template не используется.
    llm_chat_template chat_template = LLM_CHAT_TEMPLATE_UNKNOWN;
};


// ============================================================================
// 10. ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
// ============================================================================
// Назначение: разобрать argv в whisper_params, напечатать --help.
// Инварианты:
//   • Любой неизвестный аргумент → печать usage + return false.
//   • Ошибка stoi/stof → печать usage + return false.
//   • --person-gender / --bot-gender: только "male" | "female" | "".
// Используется: run() в самом начале (22.1).

// --- 10.1. Печать справки ---
void whisper_print_usage(int argc, char** argv, const whisper_params & params) {
    (void)argc;
    fprintf(stderr, "\nusage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads (CPU)\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] max voice segment length, ms\n", params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n", params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] energy vad threshold\n", params.vad_thold);
    fprintf(stderr, "  -vths N,  --vad-start-thold[%-7.6f] energy vad min level to start\n", params.vad_start_thold);
    fprintf(stderr, "  -vlm N,   --vad-last-ms N  [%-7.2f] energy vad min silence after speech\n", params.vad_last_ms);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  --interrupt-ms N           [%-7d] ms of sustained speech to trigger barge-in\n", params.interrupt_ms);
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name\n", params.person.c_str());
    fprintf(stderr, "  --person-gender G          [%-7s] male|female (optional, for fallback prompt)\n", params.person_gender.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name\n", params.bot_name.c_str());
    fprintf(stderr, "  --bot-gender G             [%-7s] male|female (optional, for fallback prompt)\n", params.bot_gender.c_str());
    fprintf(stderr, "  -w CMD,   --wake-command   [%-7s] wake word (optional)\n", params.wake_cmd.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -tr,      --translate                translate from source language to english\n");
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n", params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n", params.model_llama.c_str());
    fprintf(stderr, "  --ctx_size N               [%-7d] size of the prompt context\n", params.ctx_size);
    fprintf(stderr, "  -b N,     --batch-size N   [auto: ctx/8, 128..2048] logical batch size\n");
    fprintf(stderr, "  -n N,     --n_predict N    [auto: ctx/16, 256..1024] max tokens to predict\n");
    fprintf(stderr, "  --temp N                   [%-7.2f] temperature\n", params.temp);
    fprintf(stderr, "  --top_k N                  [%-7d] top_k\n", params.top_k);
    fprintf(stderr, "  --top_p N                  [%-7.2f] top_p\n", params.top_p);
    fprintf(stderr, "  --min_p N                  [%-7.2f] min_p\n", params.min_p);
    fprintf(stderr, "  --repeat_penalty N         [%-7.2f] repetition penalty\n", params.repeat_penalty);
    fprintf(stderr, "  --repeat_last_n N          [%-7d] window for repetition penalty\n", params.repeat_last_n);
    fprintf(stderr, "  --n_keep N                 [auto: max(512, prompt+64), <= ctx/2]\n");
    fprintf(stderr, "  --min-tokens N             [auto: max(20, ctx/256)] min new tokens\n");
    fprintf(stderr, "  --cache-type-k NAME        [default: f16] K cache type\n");
    fprintf(stderr, "  --cache-type-v NAME        [default: f16] V cache type\n");
    fprintf(stderr, "  --xtts-voice NAME          [%-7s] xtts voice without .wav\n", params.xtts_voice.c_str());
    fprintf(stderr, "  --xtts-url TEXT            [%-7s] xtts server URL\n", params.xtts_url.c_str());
    fprintf(stderr, "  --sleep-before-xtts N      [%-7d] sleep llama inference before xtts, ms\n", params.sleep_before_xtts);
    fprintf(stderr, "  --split-after N            [%-7d] chunk size for xtts, 0=off\n", params.split_after);
    fprintf(stderr, "  --multi-chars                       switch xtts voice on 'Name:' in reply\n");
    fprintf(stderr, "  --google-url TEXT          [%-7s] google-serper server URL\n", params.google_url.c_str());
    fprintf(stderr, "  --stop-words TEXT          [%-7s] llama stop words separated by ;\n", params.stop_words.c_str());
    fprintf(stderr, "  --prompt-file FNAME                 file with custom prompt (REQUIRED)\n");
    fprintf(stderr, "  --alt-prompt-file FNAME             file with prompt for Leo\n");
    fprintf(stderr, "  --instruct-preset TEXT              instruct preset without .json\n");
    fprintf(stderr, "  --session FNAME                     file to cache model state (KV)\n");
    fprintf(stderr, "  --auto-continue                     enable auto-continue (default: on)\n");
    fprintf(stderr, "  --no-auto-continue                  disable auto-continue\n");
    fprintf(stderr, "  --continue-max-ms N                 max ms after barge-in\n");
    fprintf(stderr, "  --allow-newline                     allow newlines in llama output\n");
    fprintf(stderr, "  --seqrep                            enable sequence repetition penalty\n");
    fprintf(stderr, "  --verbose                           print speed, debug info and write log file\n");
    fprintf(stderr, "  --verbose-prompt                    print the full prompt at start\n");
    fprintf(stderr, "  --print-energy                      print energy levels for vad debug\n");
    fprintf(stderr, "  --print-special                     print special tokens (LLaMA) in console\n");
    fprintf(stderr, "  --push-to-talk                      hold Alt to speak (default: off)\n");
    fprintf(stderr, "  --no-flash-attn                     force-disable flash attention\n");
    fprintf(stderr, "\n");
}


// --- 10.2. Основной парсер ---
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
            else if (arg == "--no-flash-attn") {
                params.flash_attn_disable = true;
                params.flash_attn = false;
            }
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
            else if (arg == "--interrupt-ms") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.interrupt_ms = std::stoi(argv[++i]);
            }
            else if (arg == "--ctx_size") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.ctx_size = std::stoi(argv[++i]);
            }
            else if (arg == "-b" || arg == "--batch-size") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.batch_size = std::stoi(argv[++i]);
                params.auto_batch_size = false;
            }
            else if (arg == "-n" || arg == "--n_predict") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.n_predict = std::stoi(argv[++i]);
                params.auto_n_predict = false;
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
                params.auto_n_keep = false;
            }
            else if (arg == "--min-tokens") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.min_tokens = std::stoi(argv[++i]);
                params.auto_min_tokens = false;
            }
            else if (arg == "--cache-type-k") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.cache_type_k = argv[++i];
            }
            else if (arg == "--cache-type-v") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.cache_type_v = argv[++i];
            }
            else if (arg == "--allow-newline") { params.allow_newline = true; }
            else if (arg == "--multi-chars") { params.multi_chars = true; }
            else if (arg == "--seqrep") { params.seqrep = true; }
            else if (arg == "--xtts-voice") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.xtts_voice = argv[++i];
            }
            else if (arg == "--xtts-url") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.xtts_url = argv[++i];
            }
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
            else if (arg == "--person-gender") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                std::string g = LowerCase(argv[++i]);
                if (g != "male" && g != "female" && !g.empty()) {
                    std::cerr << "Error: --person-gender must be 'male' or 'female'" << std::endl;
                    return false;
                }
                params.person_gender = g;
            }
            else if (arg == "-bn" || arg == "--bot-name") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.bot_name = argv[++i];
            }
            else if (arg == "--bot-gender") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                std::string g = LowerCase(argv[++i]);
                if (g != "male" && g != "female" && !g.empty()) {
                    std::cerr << "Error: --bot-gender must be 'male' or 'female'" << std::endl;
                    return false;
                }
                params.bot_gender = g;
            }
            else if (arg == "-w" || arg == "--wake-command") {
                if (i + 1 >= argc) { std::cerr << "Error: missing value after " << arg << std::endl; return false; }
                params.wake_cmd = argv[++i];
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
                const char* prompt_path = argv[++i];
                std::ifstream file(prompt_path);
                if (!file.is_open()) {
                    std::cerr << "Warning: failed to open prompt file: "
                              << prompt_path
                              << ". Using built-in fallback prompt.\n";
                } else {
                    std::copy(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>(),
                              std::back_inserter(params.prompt));
                    if (!params.prompt.empty() && params.prompt.back() == '\n')
                        params.prompt.pop_back();
                }
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
// 11. ГОЛОСОВЫЕ КОМАНДЫ (ДЕТЕКТОРЫ)
// ============================================================================
// Назначение: распознать команду в тексте, извлечь ключевое слово.
// Инварианты:
//   • is_stop_command — только короткие фразы (≤4 слов, ≤32 символов).
//   • looks_like_command — не отбрасывать короткие реплики, которые могут
//     быть командами.
//   • ParseCommandAndGetKeyword — для "google" убирает префикс
//     «погугли»/«гугл»/...; для "call" — приводит имя к именительному.
// Используется: 23.9 (стоп), 20.11 (audio), 23.13 (google).

// --- 11.1. Команда «стоп» ---
static bool is_stop_command(const std::string& text_lower) {
    if (text_lower.empty()) return false;
    if (text_lower.size() > STOP_COMMAND_MAX_CHARS) return false;

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
    if (word_count > STOP_COMMAND_MAX_WORDS) return false;

    if (clean == "стоп" || clean == "stop" || clean == "хватит" ||
        clean == "тихо" || clean == "стой" || clean == "остановись" ||
        clean == "прекрати" || clean == "замолчи") return true;
    if (clean.find("стоп") == 0 || clean.find("stop") == 0) return true;
    return false;
}


// --- 11.2. Похоже ли на команду ---
// Используется в audio-потоке, чтобы не отбрасывать короткие реплики,
// которые могут быть командами («стоп», «сброс», «время»).
static bool looks_like_command(const std::string& text_lower) {
    if (text_lower.empty()) return false;
    static const std::vector<std::string> needles = {
        "стоп", "stop", "хватит", "тихо", "стой", "остановись",
        "прекрати", "замолчи",
        "сброс", "сбро", "обнул", "reset",
        "повтор", "repeat",
        "удали", "сотри", "убери", "delete",
        "перегенерир", "regenerate",
        "позови", "вызови", "call",
        "погугли", "загугли", "поищи", "найди",
        "время", "дата", "число"
    };
    for (const auto& n : needles) {
        if (text_lower.find(n) != std::string::npos) return true;
    }
    return false;
}


// --- 11.3. Извлечение ключевого слова из команды ---
// Для "google": удаляет вводные («пожалуйста», «погугли», «гугл»...)
// и возвращает остаток.
// Для "call": приводит имя к именительному падежу (для русского).
std::string ParseCommandAndGetKeyword(std::string textHeardTrimmed,
    const std::string& command = "google") {
    textHeardTrimmed = StripPunctuationMarks(textHeardTrimmed);
    std::string sanitizedInput = textHeardTrimmed;
    size_t pos = 0;
    bool startsWithPrefix = false;

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
// 12. HTTP-КЛИЕНТ (curl)
// ============================================================================
// Назначение: URL-кодирование, GET-запрос, преобразование URL в речь.
// Инварианты:
//   • send_curl — только GET, таймауты 30/10 с.
//   • progress_callback отменяет запрос при g_cancel_tts_requests или
//     g_tts_muted.
// Используется: 23.13 (google), 17.3 (TTS).

// --- 12.1. URL-кодирование ---
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


// --- 12.2. Callback записи HTTP-ответа ---
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    if (userp) {
        ((std::string*)userp)->append(static_cast<char*>(contents), realsize);
    }
    return realsize;
}


// --- 12.3. Callback прогресса ---
// Возвращает 1, если запрос нужно отменить. Используется для
// прерывания долгих TTS-запросов.
static int progress_callback(void* /*clientp*/,
    curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
    curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    if (g_cancel_tts_requests.load()) return 1;
    if (g_tts_muted.load()) return 1;
    return 0;
}


// --- 12.4. Простой GET-запрос ---
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

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (res != CURLE_OK) {
            if (g_verbose_mode.load()) {
                fprintf(stderr, "[cURL] Ошибка GET-запроса: %s\n",
                        curl_easy_strerror(res));
            }
            readBuffer.clear();
        } else if (http_code != 200) {
            if (g_verbose_mode.load()) {
                fprintf(stderr, "[cURL] HTTP %ld для %s\n", http_code, url.c_str());
            }
            readBuffer.clear();
        }

        curl_easy_cleanup(curl);
    }
    return readBuffer;
}


// --- 12.5. URL → произносимый текст ---
// «https://example.com/path?x=1» → «example dot com slash path».
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
// 13. INSTRUCT-ПРЕСЕТ (PresetDerived)
// ============================================================================
// Назначение: все стоп-строки, антипромпты и префиксы выводятся из 7 полей
//   JSON-пресета. Новый пресет добавляется без перекомпиляции.
// Инварианты:
//   • stop_strings = {stop_sequence, bot_suffix, user_suffix} —
//     проверяются при генерации токенов (23.17.8.3).
//   • antiprompts = stop_strings + {bot_prefix, user_prefix, stop_words} —
//     проверяются по хвосту text_to_speak (23.17.8.7).
//   • statuses — список статусов для консоли (tts/int/stop/...).
// Используется: 22.14, 22.15, 22.23, 22.31, 23.13, 23.17.5.

struct PresetDerived {
    std::string sys_prefix;
    std::string sys_suffix;
    std::string user_prefix;
    std::string user_suffix;
    std::string bot_prefix;
    std::string bot_suffix;
    std::string stop_sequence;

    std::vector<std::string> stop_strings;
    std::vector<std::string> antiprompts;

    std::unordered_set<std::string> statuses;
};


// --- 13.1. Сборка PresetDerived из JSON-данных ---
static PresetDerived build_preset_derived(
    const std::map<std::string, std::string>& data)
{
    PresetDerived d;

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

    d.statuses = {
        "google", "time", "date", "reset", "del", "regen", "repeat",
        "tts", "int", "stop", "call leo"
    };

    return d;
}


// --- 13.2. Универсальная обработка BOS ---
// Назначение: гарантировать, что embd_inp начинается с BOS-токена,
//   если модель его требует.
// Инварианты: BOS добавляется только если его нет ни в sys_prefix,
//   ни в user_prefix, ни в bot_prefix.
// Используется: 22.23, 22.24 (rebuild), 23.13 (reset).
static void ensure_bos_in_prompt(
    std::vector<llama_token>& embd_inp,
    llama_context* ctx,
    const llama_vocab* vocab,
    bool add_bos_token,
    const PresetDerived& preset)
{
    if (!add_bos_token || !vocab || !ctx) return;

    const llama_token bos_token = llama_vocab_bos(vocab);
    if (bos_token < 0) return;

    const std::string* probes[] = {
        &preset.sys_prefix,
        &preset.user_prefix,
        &preset.bot_prefix,
    };
    for (const std::string* p : probes) {
        if (!p || p->empty()) continue;
        std::vector<llama_token> toks =
            llama_tokenize_ex(ctx, *p, /*add_special=*/false, /*parse_special=*/true);
        for (llama_token t : toks) {
            if (t == bos_token) return;
        }
    }

    if (embd_inp.empty() || embd_inp[0] != bos_token) {
        embd_inp.insert(embd_inp.begin(), bos_token);
    }
}

// ============================================================================
// 14. WHISPER-ТРАНСКРИПЦИЯ
// ============================================================================
// Назначение: обёртка над whisper_full. Внутренний VAD Whisper отключён —
//   сегментация делается энергетическим VAD (секция 6).
//
// Параметры whisper_full:
//   beam_size=1, best_of=1, temperature_inc=0, no_timestamps=true,
//   audio_ctx=0, max_tokens=64, no_speech_thold=0.4.
//
// Выходы:
//   prob            — средняя вероятность токенов (legacy, fallback)
//   no_speech_prob  — вероятность «нет речи» для первого сегмента
//                     (точнее для фильтра галлюцинаций)
//   t_ms            — время выполнения, мс
//
// Инварианты:
//   • carry_initial_prompt = true — prompt передаётся в следующий сегмент.
//   • prompt_tokens_vec живёт до конца whisper_full — не освобождать раньше.
// Используется: 20.9 (warmup), 20.10 (аварийное сегментирование), 20.11 (конец речи).

static std::string transcribe(
    whisper_context* ctx,
    const whisper_params& params,
    const std::vector<float>& pcmf32,
    const std::string& prompt_text,
    float& prob,
    float& no_speech_prob,
    int64_t& t_ms,
    bool translate = false) {
    prob = 0.0f;
    no_speech_prob = 0.0f;
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

    wparams.vad = false;

    wparams.beam_search.beam_size = 1;
    wparams.greedy.best_of = 1;
    wparams.temperature_inc = 0.0f;

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
            wparams.carry_initial_prompt = true;
        } else {
            wparams.prompt_tokens = nullptr;
            wparams.prompt_n_tokens = 0;
        }
    } else {
        wparams.prompt_tokens = nullptr;
        wparams.prompt_n_tokens = 0;
    }

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

    wparams.temperature = 0.2f;
    wparams.entropy_thold = 2.4f;
    wparams.logprob_thold = -1.0f;
    wparams.no_speech_thold = 0.4f;
    wparams.length_penalty = 0.5f;
    wparams.max_len = 96;

    {
        int model_text_ctx = static_cast<int>(whisper_n_text_ctx(ctx));
        int mt = 64;
        if (mt > model_text_ctx) mt = model_text_ctx;
        wparams.max_tokens = mt;
    }
    wparams.audio_ctx = 0;

    // Whisper: пустая строка и "auto" означают автоматическое определение языка.
    wparams.language =
        (params.language.empty() || params.language == "auto")
            ? nullptr
            : params.language.c_str();
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

    if (n_segments > 0) {
        no_speech_prob = whisper_full_get_segment_no_speech_prob(ctx, 0);
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


// ============================================================================
// 15. ДЕТЕКТОР ГАЛЛЮЦИНАЦИЙ WHISPER
// ============================================================================
// Назначение: отсеивать мусорные выходы Whisper — субтитры, музыку,
//   повторы. НЕ должен рубить короткие легитимные реплики («да», «нет»,
//   «привет»).
// Инварианты:
//   • is_short_valid_reply проверяется ДО is_hallucination — иначе
//     «да»/«нет» будут отброшены как слишком короткие.
//   • no_speech_prob проверяется отдельно в audio_input_thread_func.
// Используется: 20.10, 20.11, 23.11.

// --- 15.1. Легитимный короткий ответ ---
// Whitelist коротких слов + проверка «1–3 буквы без пунктуации».
static bool is_short_valid_reply(const std::string& text_lower) {
    static const std::unordered_set<std::string> whitelist = {
        "да", "нет", "неа", "ага", "угу", "ок", "окей", "okay",
        "привет", "хай", "здравствуй", "здрасьте", "пока",
        "спасибо", "ну", "эй", "эй!", "хм", "ммм", "давай",
        "хватит", "ладно", "хорошо", "ясно", "понятно",
        "yes", "no", "yeah", "nope", "yep", "ok", "okay", "hi",
        "hello", "hey", "bye", "thanks", "thank you", "sure",
        "stop", "go", "right", "wrong", "maybe"
    };
    if (whitelist.count(text_lower)) return true;

    int len = utf8_length(text_lower);
    if (len >= 1 && len <= 3) {
        bool only_letters = true;
        size_t i = 0;
        const size_t n = text_lower.size();
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(text_lower[i]);
            if (c < 0x80) {
                if (!std::isalpha(c)) { only_letters = false; break; }
                i += 1;
            } else if ((c & 0xE0) == 0xC0) { i += 2; }
            else if ((c & 0xF0) == 0xE0) { i += 3; }
            else if ((c & 0xF8) == 0xF0) { i += 4; }
            else { only_letters = false; break; }
        }
        if (only_letters) return true;
    }
    return false;
}


// --- 15.2. Детектор галлюцинаций ---
// Три уровня проверки:
//   1) точное совпадение с известным мусором;
//   2) подстроки «Субтитры», «Спасибо за просмотр» и т.п.;
//   3) повторяющийся паттерн (≥4 раза) + слово-доминанта (≥50%).
static bool is_hallucination(const std::string& text) {
    std::string text_lower = LowerCase(text);
    trim(text_lower);

    static const std::unordered_set<std::string> exact_matches_lower = {
        "thank you", "thanks for watching", "thank you for watching",
        "please subscribe", "like and subscribe", "subscribe to my channel",
        "see you next time", "bye", "you", "the end",
        "amara.org", "subtitles by the amara org community",
        "продолжение следует",
        "субтитры подогнал симон",
        "редактор субтитров а.семкин корректор а.егорова",
        "[музыка]",
        "sous-titres", "sottotitoli", "untertitel",
        "www.mooji.org", "ご視聴ありがとうございました", "자막 제공", "mbc 뉴스",
        " ", "  ", "!", ".", "...", "♪", "[blank_audio]",
    };

    if (exact_matches_lower.count(text_lower)) return true;

    static const std::vector<std::string> substrings = {
        "Редактор субтитров", "Спасибо за внимание",
        "Продолжение следует", "Translated by",
        "Thanks for watching", "Thank you for watching",
        "Субтитры", "До новых встреч", "Спасибо за просмотр",
        "СПОКОЙНАЯ МУЗЫКА", "ЗВОНОК В ДВЕРЬ",
        "Мы с вами поздравляем вас с праздником!",
        "заставка", "интро", "аутро"
    };

    static const std::vector<std::string> substrings_lower = []{
        std::vector<std::string> v;
        for (const auto& s : substrings) {
            v.push_back(LowerCase(s));
        }
        return v;
    }();
    for (const auto& sub : substrings_lower) {
        if (text_lower.find(sub) != std::string::npos) return true;
    }

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

    if (is_short_valid_reply(text_lower)) {
        return false;
    }
    if (utf8_length(text) < 3) {
        return true;
    }

    if (text.length() > 3 && text.length() < 20) {
        bool all_same = true;
        for (size_t i = 1; i < text.length(); ++i) {
            if (text[i] != text[0]) { all_same = false; break; }
        }
        if (all_same && text.length() >= 3) return true;
    }

    return false;
}


// --- 15.3. Учёт транскрипции ---
// Просто счётчик для статистики. Не влияет на поток.
static void note_transcription(const std::string& text) {
    if (text.empty()) return;
    (void)text;
    g_transcription_count.fetch_add(1);
}


// ============================================================================
// 16. ОЧИСТКА ТЕКСТА ДЛЯ TTS
// ============================================================================
// Назначение: превратить вывод модели в чистый текст для XTTS.
// Правила:
//   (круглые)     → ". " (содержимое удаляется, граница предложения)
//   [квадратные]  → ", содержимое" (содержимое сохраняется)
//   {фигурные}    → "" (удаляются полностью)
//   <|...|>       → "" (любые спецтокены)
//   markdown      → очищается
//   HTML-теги     → пробел
//
// ВАЖНО: на этапе ГЕНЕРАЦИИ ТОКЕНА эта очистка НЕ вызывается. Только
//   в send_tts_async — когда буфер text_to_speak готов к отправке в XTTS.
//
// Инварианты:
//   • clean_text_for_tts вызывается ТОЛЬКО из send_tts_async.
//   • Если после очистки осталась пустая строка — не отправляем.
// Используется: 17.4 (send_tts_async).

// --- 16.1. Проверка: маркер списка? ---
// Возвращает длину маркера (для удаления) или 0.
// Распознаёт: "- ", "* ", "+ ", "1. ", "1) ", "1: ", "• ", "· ", кириллица + точка.
static size_t detect_list_marker(const std::string& line, size_t start) {
    if (start >= line.size()) return 0;
    size_t p = start;

    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
    if (p >= line.size()) return 0;

    size_t marker_end = p;
    unsigned char c0 = static_cast<unsigned char>(line[p]);

    if ((line[p] == '-' || line[p] == '*' || line[p] == '+') &&
        p + 1 < line.size() &&
        (line[p + 1] == ' ' || line[p + 1] == '\t')) {
        marker_end = p + 2;
        while (marker_end < line.size() &&
               (line[marker_end] == ' ' || line[marker_end] == '\t')) {
            marker_end++;
        }
        return marker_end;
    }

    if (p + 2 < line.size() &&
        static_cast<unsigned char>(line[p]) == 0xE2 &&
        static_cast<unsigned char>(line[p + 1]) == 0x80 &&
        static_cast<unsigned char>(line[p + 2]) == 0xA2) {
        marker_end = p + 3;
        while (marker_end < line.size() &&
               (line[marker_end] == ' ' || line[marker_end] == '\t')) {
            marker_end++;
        }
        return marker_end;
    }
    if (p + 1 < line.size() &&
        static_cast<unsigned char>(line[p]) == 0xC2 &&
        static_cast<unsigned char>(line[p + 1]) == 0xB7) {
        marker_end = p + 2;
        while (marker_end < line.size() &&
               (line[marker_end] == ' ' || line[marker_end] == '\t')) {
            marker_end++;
        }
        return marker_end;
    }

    if (std::isdigit(c0)) {
        size_t q = p;
        while (q < line.size() &&
               std::isdigit(static_cast<unsigned char>(line[q]))) q++;
        size_t n_digits = q - p;
        if (n_digits >= 1 && n_digits <= 2 &&
            q < line.size() &&
            (line[q] == '.' || line[q] == ')' || line[q] == ':')) {
            size_t r = q + 1;
            while (r < line.size() &&
                   (line[r] == ' ' || line[r] == '\t')) r++;
            if (r > q + 1) return r;
        }
    }

    if (c0 >= 0x80) {
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
                   (line[r] == ' ' || line[r] == '\t')) { r++; spaces++; }
            if (spaces > 0 || r >= line.size()) {
                if (p + char_len < line.size() &&
                    (line[p + char_len] == '.' ||
                     line[p + char_len] == ')' ||
                     line[p + char_len] == ':')) {
                    return r;
                }
            }
        }
    }

    return 0;
}


// --- 16.2. Удаление маркеров списков в строках ---
static std::string strip_list_markers_lines(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    std::istringstream iss(text);
    std::string line;
    bool first_line = true;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        size_t marker_len = detect_list_marker(line, 0);
        if (marker_len > 0) {
            line = line.substr(marker_len);
        }

        if (!first_line) result += "\n";
        result += line;
        first_line = false;
    }
    return result;
}


// --- 16.3. Удаление маркеров внутри строки ---
// Осторожно: «1.5» — не маркер (после цифры идёт цифра, не пробел+буква).
static std::string strip_list_markers_inline(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (std::isdigit(c)) {
            size_t digit_start = i;
            size_t digit_end = i;
            while (digit_end < n &&
                   std::isdigit(static_cast<unsigned char>(text[digit_end]))) {
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
                    std::string left_char = utf8_last_char(text.substr(0, prev));
                    if (left_char == "?" || left_char == "!" ||
                        left_char == "." || left_char == "\n" ||
                        left_char == "\r" || left_char == ":") {
                        left_ok = true;
                    }
                }
            }

            bool middle_ok = false;
            if (digit_end < n) {
                unsigned char m = static_cast<unsigned char>(text[digit_end]);
                if (m == '.' || m == ')' || m == ':') middle_ok = true;
            }

            bool right_ok = false;
            size_t after_punct = digit_end + 1;
            if (middle_ok && after_punct < n) {
                size_t spaces = 0;
                size_t j = after_punct;
                while (j < n && (text[j] == ' ' || text[j] == '\t')) {
                    j++; spaces++;
                }
                if (spaces > 0 && j < n) {
                    unsigned char next = static_cast<unsigned char>(text[j]);
                    if (std::isalpha(next) || next >= 0xC0) {
                        right_ok = true;
                    }
                }
            }

            if (n_digits >= 1 && n_digits <= 2 && left_ok && middle_ok && right_ok) {
                result += ' ';
                i = after_punct;
                while (i < n && (text[i] == ' ' || text[i] == '\t')) i++;
                continue;
            }

            for (size_t k = digit_start; k < digit_end; ++k) result += text[k];
            i = digit_end;
            continue;
        }

        result += text[i];
        i++;
    }
    return result;
}


// --- 16.4. Полная очистка маркеров списков ---
static std::string strip_all_list_markers(const std::string& text) {
    std::string result = strip_list_markers_lines(text);
    result = strip_list_markers_inline(result);
    return result;
}


// --- 16.5. Вставка пробела после пунктуации перед заглавной ---
// «Привет.Мир» → «Привет. Мир». Не трогает «1.5» (после точки цифра).
static std::string insert_space_after_punct_before_capital(
    const std::string& text) {
    std::string result;
    result.reserve(text.size() + 16);

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        bool is_punct = false;
        size_t punct_len = 1;

        if (c == '.' || c == ',' || c == '!' || c == '?' ||
            c == ';' || c == ':') {
            is_punct = true;
        }
        else if (c == 0xE2 && i + 2 < n &&
                 static_cast<unsigned char>(text[i + 1]) == 0x80 &&
                 (static_cast<unsigned char>(text[i + 2]) == 0x93 ||
                  static_cast<unsigned char>(text[i + 2]) == 0x94)) {
            is_punct = true;
            punct_len = 3;
        }

        for (size_t k = 0; k < punct_len && i + k < n; ++k) {
            result += text[i + k];
        }
        i += punct_len;

        if (!is_punct) continue;

        if (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) {
            continue;
        }
        if (i < n && (text[i] == ' ' || text[i] == '\t')) {
            continue;
        }

        bool need_space = false;
        if (i < n) {
            unsigned char nc = static_cast<unsigned char>(text[i]);
            if (nc >= 'A' && nc <= 'Z') {
                need_space = true;
            }
            else if (nc >= '0' && nc <= '9') {
                need_space = true;
            }
            else if (nc == 0xD0 && i + 1 < n) {
                unsigned char nc2 = static_cast<unsigned char>(text[i + 1]);
                if ((nc2 >= 0x90 && nc2 <= 0xAF) || nc2 == 0x81) {
                    need_space = true;
                }
            }
        }
        if (need_space) result += ' ';
    }
    return result;
}


// --- 16.6. Базовая очистка ---
// Вызывается ТОЛЬКО из send_tts_async.
static std::string clean_text_for_tts(const std::string& text) {
    std::string result = text;

    {
        std::string stripped = text;
        trim_spaces_only(stripped);
        while (!stripped.empty() &&
               (stripped.front() == '\n' || stripped.front() == '\r')) {
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

    try {
        // WHY: содержимое круглых скобок НЕ удаляем. Раньше было
        // re_round → ". " (содержимое удалялось): «Я пришёл (как обычно)
        // поздно» → «Я пришёл .  поздно» — смысл терялся.
        //
        // Сервер (_prepare_text_for_synthesis, шаг 5) заменяет
        // '(' ')' на ', ', содержимое СОХРАНЯЕТСЯ: «Я пришёл (как обычно)
        // поздно» → «Я пришёл, как обычно, поздно». XTTS получает две
        // короткие паузы, содержимое произнесено.
        //
        // Клиент НЕ трогает круглые скобки — отдаёт их серверу как есть.
        // Если сервер по каким-то причинам недоступен (_prepare отключён),
        // XTTS может «понтить» на скобках — но это худший сценарий, а
        // стандартный путь — сервер заменит на ','.
        //
        // static const std::regex re_round(R"(\([^()]{1,300}\))", std::regex::ECMAScript);
        // result = std::regex_replace(result, re_round, ". ");  // ← ОТКЛЮЧЕНО

        // WHY: квадратные скобки → "" (удаляются ЦЕЛИКОМ, включая
        // содержимое). В TTS-тексте '[...]' — это артефакт:
        //   • таймкоды Whisper: [14:32], [00:00:00];
        //   • метки звука: [Смех], [Музыка], [Аплодисменты];
        //   • имитация субтитров LLaMA: [пауза], [вздыхает].
        //
        // XTTS на '[...]' либо произносит содержимое («смех»,
        // «музыка», «четырнадцать тридцать две»), либо «понтит»
        // на самих скобках. И то, и другое — не то, что нужно.
        //
        // Раньше re_square превращал '[...]' в ', содержимое'
        // (содержимое сохранялось). Это давало ложные срабатывания
        // на таймкодах и метках звука — XTTS их произносил.
        //
        // ВАЖНО: чистим ТОЛЬКО здесь, на пути в TTS. Консоль
        // (23.17.8.6) показывает сырой visible_token — там '[...]'
        // остаётся, чтобы вы видели, что сгенерировала модель.
        //
        // {0,300} — покрывает и пустые '[]', и длинные
        // '[как обычно в такой ситуации]'. Если внутри больше 300
        // символов — это не метка, а содержимое; оставляем как есть
        // (regex не сработает — лучше пропустить, чем съесть смысл).
        static const std::regex re_square(R"(\[[^\[\]]{0,300}\])", std::regex::ECMAScript);
        result = std::regex_replace(result, re_square, "");

        // Чистка пробелов после удаления '[...]':
        //   «Привет  мир»       → «Привет мир» (двойной пробел → одинарный)
        //   «Привет , мир»      → «Привет, мир» (пробел перед ',' убирается)
        //   «Привет .»          → «Привет.» (пробел перед '.' убирается)
        //   «( Привет)»         → «(Привет)» (пробел после '(' убирается)
        //   «Привет )»          → «Привет)» (пробел перед ')' убирается)
        //
        // Это нужно, потому что удаление '[...]' оставляет «дыры»
        // из пробелов. XTTS на двойных пробелах делает лишние паузы,
        // а пробел перед ',' звучит как «запятая с паузой».
        try {
            static const std::regex re_spaces_after_square(R"([ \t]{2,})", std::regex::ECMAScript);
            result = std::regex_replace(result, re_spaces_after_square, " ");

            static const std::regex re_space_before_punct(R"([ \t]+([,\.!\?;:]))", std::regex::ECMAScript);
            result = std::regex_replace(result, re_space_before_punct, "$1");

            static const std::regex re_space_after_open(R"(([\(\[])[ \t]+)", std::regex::ECMAScript);
            result = std::regex_replace(result, re_space_after_open, "$1");

            static const std::regex re_space_before_close(R"([ \t]+([\)\]}])", std::regex::ECMAScript);
            result = std::regex_replace(result, re_space_before_close, "$1");
        } catch (const std::regex_error&) {
            // Fallback: если regex не поддерживается — грубая схлопка.
            while (result.find("  ") != std::string::npos) {
                result = string_replace_all(result, "  ", " ");
            }
            result = string_replace_all(result, " ,", ",");
            result = string_replace_all(result, " .", ".");
            result = string_replace_all(result, " !", "!");
            result = string_replace_all(result, " ?", "?");
            result = string_replace_all(result, " ;", ";");
            result = string_replace_all(result, " :", ":");
            result = string_replace_all(result, "( ", "(");
            result = string_replace_all(result, ") ", ")");
        }

        trim_spaces_only(result);

        // WHY: фигурные скобки → "" (удаляются полностью). Это правильно:
        // '{...}' обычно служебные метки (плейсхолдеры, шаблоны), не смысл.
        static const std::regex re_curly(R"(\{[^{}]*\})", std::regex::ECMAScript);
        result = std::regex_replace(result, re_curly, "");

        static const std::regex re_bold_double(R"(\*\*([^*]+)\*\*)", std::regex::ECMAScript);
        static const std::regex re_bold_under(R"(__([^_]+)__)", std::regex::ECMAScript);
        static const std::regex re_italic_star(R"(\*([^*]+)\*)", std::regex::ECMAScript);
        static const std::regex re_italic_under(R"(_([^_]+)_)", std::regex::ECMAScript);
        static const std::regex re_strike(R"(~~([^~]+)~~)", std::regex::ECMAScript);
        // WHY: '...' (три точки) — это пауза-раздумье, XTTS её понимает.
        // Схлопывание в '. ' даёт завершённую интонацию на месте
        // незавершённой — «зажёвывание» на конце фразы.
        // Сервер (_get_silence_duration) даёт паузу 0.20 для '...',
        // сервер (_prepare_text_for_synthesis, шаг 9) НЕ снимает '...'
        // в конце. Всё это работает только если клиент оставляет '...'.
        // Четыре и более точек — артефакт модели; их схлопываем до '...',
        // чтобы XTTS не «понтил».
        static const std::regex re_multidot4(R"(\.{4,})", std::regex::ECMAScript);
        static const std::regex re_md_noise(R"([#*_~`>])", std::regex::ECMAScript);
        static const std::regex re_spaces_tabs(R"([ \t]+)", std::regex::ECMAScript);

        result = std::regex_replace(result, re_bold_double, "$1 ");
        result = std::regex_replace(result, re_bold_under, "$1 ");
        result = std::regex_replace(result, re_italic_star, " ");
        result = std::regex_replace(result, re_italic_under, "$1 ");
        result = std::regex_replace(result, re_strike, "$1 ");
        result = std::regex_replace(result, re_multidot4, "...");
        result = std::regex_replace(result, re_md_noise, " ");
        result = std::regex_replace(result, re_spaces_tabs, " ");

        static const std::regex re_pps(R"(\bP\.P\.S\.?)", std::regex::ECMAScript | std::regex::icase);
        static const std::regex re_ps (R"(\bP\.S\.?)",    std::regex::ECMAScript | std::regex::icase);
        result = std::regex_replace(result, re_pps, "Пост-постскриптум");
        result = std::regex_replace(result, re_ps,  "Постскриптум");

        trim(result);
    } catch (const std::regex_error& e) {
        if (g_verbose_mode.load()) {
            fprintf(stderr, "[clean_text_for_tts] regex error: %s\n", e.what());
        }
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

    result = strip_all_list_markers(result);

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

    // WHY: знаки препинания в начале фрагмента — это АРТЕФАКТ
    // (модель сгенерировала знак отдельным токеном перед фразой,
    // или split_after оставил знак в начале нового фрагмента).
    // XTTS v2 обучен на текстах, где фраза начинается с БУКВЫ.
    // Знак в начале → модель «понтит» («точка», «восклицательный
    // знак», «многоточие») или сбивает интонацию.
    //
    // Удаляем ВСЕ знаки в начале:
    //   '.'  ','  '!'  '?'  ';'  ':'  — ASCII-пунктуация
    //   '...'  '…'                     — многоточие
    //   '—'  '–'                        — тире
    //
    // '...' в СЕРЕДИНЕ и в КОНЦЕ — сохраняется (см. К1, К2).
    // '—' '–' '…' в СЕРЕДИНЕ и в КОНЦЕ — сохраняется (см. К4).
    while (!result.empty()) {
        unsigned char c = static_cast<unsigned char>(result[0]);

        // ASCII-пунктуация
        if (c == '.' || c == ',' || c == '!' || c == '?' ||
            c == ';' || c == ':') {
            result.erase(0, 1);
            trim_spaces_only(result);
            continue;
        }

        // UTF-8: '…' (U+2026) — 3 байта E2 80 A6
        if (result.size() >= 3 &&
            static_cast<unsigned char>(result[0]) == 0xE2 &&
            static_cast<unsigned char>(result[1]) == 0x80 &&
            static_cast<unsigned char>(result[2]) == 0xA6) {
            result.erase(0, 3);
            trim_spaces_only(result);
            continue;
        }

        // UTF-8: '—' (U+2014) — 3 байта E2 80 94
        if (result.size() >= 3 &&
            static_cast<unsigned char>(result[0]) == 0xE2 &&
            static_cast<unsigned char>(result[1]) == 0x80 &&
            static_cast<unsigned char>(result[2]) == 0x94) {
            result.erase(0, 3);
            trim_spaces_only(result);
            continue;
        }

        // UTF-8: '–' (U+2013) — 3 байта E2 80 93
        if (result.size() >= 3 &&
            static_cast<unsigned char>(result[0]) == 0xE2 &&
            static_cast<unsigned char>(result[1]) == 0x80 &&
            static_cast<unsigned char>(result[2]) == 0x93) {
            result.erase(0, 3);
            trim_spaces_only(result);
            continue;
        }

        break;
    }

    result = insert_space_after_punct_before_capital(result);

    return result;
}


// --- 16.7. Безопасное удаление фрагмента ---
// Удаляет fragment, только если он стоит на границе слова
// (иначе «<|eot_id|>» может «съесть» часть слова).
static std::string safe_remove_fragment(const std::string& text,
                                         const std::string& fragment) {
    if (fragment.empty()) return text;

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

    std::string result = text;
    size_t pos = 0;
    while ((pos = result.find(fragment, pos)) != std::string::npos) {
        bool left_ok = is_utf8_word_boundary(result, pos);
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


// --- 16.8. Схлопка знаков препинания ---
// «!!!» → «!», «?!» → «?», «!!» → «!», «??» → «?».
//
// ВАЖНО: «...» (три точки) НЕ схлопывается. Это пауза-раздумье,
// XTTS её понимает. Схлопывание «...» → «.» даёт завершённую
// интонацию на месте незавершённой — «зажёвывание» на конце фразы.
// «....» и более (4+ точек) — артефакт модели, схлопывается до «...».
//
// Защита «...» сделана через placeholder из Private Use Area
// (\uE000-\uF8FF) — эти символы не встречаются в обычном тексте
// и не пересекаются с логикой схлопывания парных знаков.
static std::string collapse_punctuation_marks(const std::string& text) {
    std::string result = text;

    // Шаг 1. Артефакт модели: 4+ точек → «...». Делаем ДО защиты,
    // чтобы placeholder не пришлось защищать дважды.
    {
        static const std::regex re_dots4plus(R"(\.{4,})", std::regex::ECMAScript);
        result = std::regex_replace(result, re_dots4plus, "...");
    }

    // Шаг 2. Защищаем «...» placeholder-ом, чтобы re_punct_group
    // не схлопнул три точки в одну.
    static const std::string ellipsis_marker = "\uE000ELLIPSIS\uE000";
    result = string_replace_all(result, "...", ellipsis_marker);

    // Шаг 3. Схлопываем парные знаки (кроме защищённого «...»).
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
        // Fallback: placeholder «\uE000ELLIPSIS\uE000» не содержит
        // последовательности «..», поэтому «..» здесь безопасно.
        while (result.find("!!") != std::string::npos)
            result = string_replace_all(result, "!!", "!");
        while (result.find("??") != std::string::npos)
            result = string_replace_all(result, "??", "?");
        while (result.find("..") != std::string::npos)
            result = string_replace_all(result, "..", ".");
    }

    // Шаг 4. Возвращаем «...» на место.
    result = string_replace_all(result, ellipsis_marker, "...");

    return result;
}


// --- 16.9. Удаление префикса имени ---
// «Адам: Привет» → «Привет». Работает только для начала строки.
static std::string strip_dialog_prefixes(const std::string& text,
                                          const std::string& person,
                                          const std::string& bot,
                                          const std::string& symb) {
    std::string r = text;

    size_t start = 0;
    while (start < r.size()) {
        unsigned char c = static_cast<unsigned char>(r[start]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xA0) ++start;
        else break;
    }
    if (start > 0) r.erase(0, start);

    std::vector<std::string> names = { person, bot };
    std::sort(names.begin(), names.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() > b.size();
              });
    for (const auto& name : names) {
        std::string with_symb  = name + symb;
        std::string with_colon = name + ":";
        if (r.compare(0, with_symb.size(), with_symb) == 0) {
            r = r.substr(with_symb.size());
            break;
        } else if (r.compare(0, with_colon.size(), with_colon) == 0) {
            r = r.substr(with_colon.size());
            break;
        }
    }

    trim_spaces_only(r);
    return r;
}


// ============================================================================
// 17. АСИНХРОННАЯ ОТПРАВКА В XTTS
// ============================================================================
// Назначение: send_tts_async — единственная точка, где видимый текст
//   проходит нормализацию под XTTS. Один вызов = один HTTP-запрос.
//   Разбиение на чанки делается до очереди (enqueue_tts_maybe_split).
//
// Инварианты:
//   • g_tts_muted / g_cancel_tts_requests → немедленный выход.
//   • progress_callback отменяет HTTP при g_cancel_tts_requests.
//   • После всех замен — если text пуст, не отправляем.
// Используется: 18.4 (TTS-воркер).

// --- 17.1. Вставка пробела между цифрой и буквой ---
// «2х2» → «2 х2» (для корректного произношения XTTS).
static std::string insert_space_between_digit_and_letter(const std::string& text) {
    std::string fixed;
    fixed.reserve(text.size() + 16);
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t step = 1;
        if ((c & 0xE0) == 0xC0) step = 2;
        else if ((c & 0xF0) == 0xE0) step = 3;
        else if ((c & 0xF8) == 0xF0) step = 4;

        bool cur_is_digit = (c >= '0' && c <= '9');
        bool next_is_letter = false;
        if (i + step < n) {
            unsigned char nc = static_cast<unsigned char>(text[i + step]);
            if ((nc >= 'a' && nc <= 'z') || (nc >= 'A' && nc <= 'Z')) {
                next_is_letter = true;
            }
            else if ((nc == 0xD0 || nc == 0xD1) && i + step + 1 < n) {
                unsigned char nc2 = static_cast<unsigned char>(text[i + step + 1]);
                if (nc2 >= 0x80 && nc2 <= 0xBF) {
                    next_is_letter = true;
                }
            }
        }

        for (size_t k = 0; k < step && i + k < n; ++k) {
            fixed += text[i + k];
        }
        i += step;

        if (cur_is_digit && next_is_letter) {
            fixed += ' ';
        }
    }
    return fixed;
}


// --- 17.2. Удаление скобок с датой/временем ---
// «[14:32]» → удаляется. «[музыка]» → остаётся (не дата).
static std::string strip_datetime_brackets(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (text[i] == '[') {
            size_t close = text.find(']', i + 1);
            if (close != std::string::npos && close > i + 1) {
                std::string inside = text.substr(i + 1, close - i - 1);
                bool is_datetime = false;

                for (size_t p = 0; p + 2 < inside.size(); ++p) {
                    if (inside[p] != ':') continue;
                    if (!std::isdigit(static_cast<unsigned char>(inside[p + 1])) ||
                        !std::isdigit(static_cast<unsigned char>(inside[p + 2]))) {
                        continue;
                    }
                    if (p == 0 ||
                        !std::isdigit(static_cast<unsigned char>(inside[p - 1]))) {
                        continue;
                    }
                    size_t left_digits = 0;
                    size_t q = p;
                    while (q > 0 &&
                           std::isdigit(static_cast<unsigned char>(inside[q - 1]))) {
                        left_digits++;
                        q--;
                    }
                    if (left_digits >= 1 && left_digits <= 2) {
                        is_datetime = true;
                        break;
                    }
                }

                if (!is_datetime &&
                    inside.find("года") != std::string::npos &&
                    inside.find_first_of("0123456789") != std::string::npos) {
                    is_datetime = true;
                }
                if (is_datetime) {
                    i = close + 1;
                    continue;
                }
            }
        }
        cleaned += text[i];
        ++i;
    }
    return cleaned;
}


// --- 17.3. Отправка одного чанка в XTTS ---
// Делает полную нормализацию текста + один HTTP-запрос.
static void send_tts_single_chunk(
    std::string text,
    const std::string& speaker_wav,
    const std::string& language,
    const std::string& tts_url,
    const std::string& stop_sequence,
    const std::string& bot_suffix,
    const std::string& user_suffix,
    const std::string& /*bot_prefix*/,
    const std::string& /*user_prefix*/,
    const std::string& chat_symb,
    const std::string& person_name,
    const std::string& bot_name)
{
    if (text.empty()) return;

    text = insert_space_between_digit_and_letter(text);
    text = strip_datetime_brackets(text);

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

    text = string_replace_all(text, "\xE2\x80\x9C", "\"");
    text = string_replace_all(text, "\xE2\x80\x9D", "\"");
    text = string_replace_all(text, "\xE2\x80\x98", "'");
    text = string_replace_all(text, "\xE2\x80\x99", "'");
    // WHY: '—' (U+2014, длинное тире), '–' (U+2013, короткое тире)
    // и '…' (U+2026, многоточие) НЕ заменяем.
    //
    // '—' и '–': XTTS v2 понимает их как паузу с интонацией
    // (противопоставление, вставка). Сервер (_get_silence_duration)
    // даёт паузу 0.20 для '—', 0.16 для '–'. Если клиент заменит
    // на '-' (дефис) — XTTS в середине фрагмента может прочитать
    // «дефис» или «чёрточка», интонация ломается, хотя пауза
    // формально правильная. Оставляем '—' и '–' как есть.
    //
    // '…': XTTS понимает как паузу-раздумье (как '...').
    // Сервер (_get_silence_duration) даёт паузу 0.20 для '…'.
    // Замена на '...' не критична (оба варианта работают), но
    // лишняя — оставляем '…' как есть.
    //
    // text = string_replace_all(text, "\xE2\x80\x93", "-");  // ← ОТКЛЮЧЕНО
    // text = string_replace_all(text, "\xE2\x80\x94", "-");  // ← ОТКЛЮЧЕНО
    text = string_replace_all(text, "\xC2\xA0", " ");
    // text = string_replace_all(text, "\xE2\x80\xA6", "...");  // ← ОТКЛЮЧЕНО

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

    text = strip_all_list_markers(text);

    text = string_replace_all(text, "|", " ");
    text = string_replace_all(text, "\\", " ");

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

    text = collapse_punctuation_marks(text);

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

    for (const auto& p : protected_patterns) {
        text = string_replace_all(text, p.first, p.second);
    }

    text = strip_dialog_prefixes(text, person_name, bot_name, chat_symb);
    if (text.empty()) return;

    if (!stop_sequence.empty()) text = safe_remove_fragment(text, stop_sequence);
    if (!bot_suffix.empty())    text = safe_remove_fragment(text, bot_suffix);
    if (!user_suffix.empty())   text = safe_remove_fragment(text, user_suffix);
    trim_spaces_only(text);
    if (text.empty()) return;

    // WHY: ',' в конце фрагмента НЕ удаляем.
    //
    // ', ' в конце — это незавершённая интонация (продолжение мысли).
    // XTTS v2 понимает её как продолжение, а не как конец фразы.
    // Сервер (_prepare_text_for_synthesis, шаг 9) специально
    // ОСТАВЛЯЕТ ',' в конце, а _get_silence_duration даёт паузу
    // 0.08 — короткую, как у запятой.
    //
    // Если удалить ',' здесь — XTTS получит завершённую интонацию
    // (как точку), а пауза 0.08 слишком коротка, чтобы это звучало
    // естественно — «зажёвывание» на конце фразы.
    //
    // Запятую перед '.', '!', '?' всё равно убираем — там она мусор
    // (порядок «,.» «,!» «,?» не имеет смысла в русском).
    //
    // if (!text.empty() && text.back() == ',') text.pop_back();  // ← ОТКЛЮЧЕНО
    text = string_replace_all(text, ",.", ".");
    text = string_replace_all(text, ",!", "!");
    text = string_replace_all(text, ",?", "?");
    trim_spaces_only(text);
    if (text.empty()) return;

    if (text == "eo" || text == "Eo" || text == "t_id" || text == "id" ||
        text == "eot" || text == speaker_wav) {
        return;
    }

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
        curl_easy_setopt(http_handle, CURLOPT_POSTFIELDSIZE, (long)data.size());
        curl_easy_setopt(http_handle, CURLOPT_VERBOSE, 0L);
        std::string responseData;
        curl_easy_setopt(http_handle, CURLOPT_WRITEDATA, &responseData);
        curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, WriteCallback);
        CURLcode res = curl_easy_perform(http_handle);
        if (res == CURLE_ABORTED_BY_CALLBACK) {
            if (g_verbose_mode.load()) {
                fprintf(stderr, "[TTS] Запрос отменён (stop/barge-in)\n");
            }
        } else if (res != CURLE_OK && g_verbose_mode.load()) {
            fprintf(stderr, "[TTS] cURL ошибка: %s\n", curl_easy_strerror(res));
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(http_handle);
    }
}


// --- 17.4. Публичная отправка: очистка + один HTTP ---
static void send_tts_async(const TtsRequest& req) {
    if (req.text.empty()) return;
    if (g_tts_muted.load()) return;
    if (g_cancel_tts_requests.load()) return;

    std::string text = clean_text_for_tts(req.text);
    if (text.empty()) return;

    try {
        static const std::regex re_special_token(R"(<\|[^|]*\|>)",
                                                  std::regex::ECMAScript);
        text = std::regex_replace(text, re_special_token, "");
    } catch (const std::regex_error&) {}

    if (text.empty()) return;

    send_tts_single_chunk(text, req.voice, req.language, req.url,
        req.stop_seq, req.bot_sfx, req.user_sfx, req.bot_pfx, req.user_pfx,
        req.chat_symb, req.person, req.bot);
}


// ============================================================================
// 18. TTS-ОЧЕРЕДЬ И ВОРКЕР
// ============================================================================
// Назначение: один фоновый поток обрабатывает запросы из g_tts_queue.
//   Основной цикл не блокируется на HTTP-запросах и продолжает генерировать.
//
// Инварианты:
//   • TtsRequest = один HTTP-запрос. Разбиение по '\n' — до очереди (18.2).
//   • g_tts_busy = true на время HTTP; g_tts_queue_cv.wait_for(100ms).
//   • При g_cancel_tts_requests или g_tts_muted — запрос пропускается.
// Используется: 22.33 (запуск воркера), 23.17.8.8 (enqueue).

// --- 18.1. Постановка запроса в очередь ---
static void enqueue_tts(TtsRequest req) {
    std::lock_guard<std::mutex> lock(g_tts_queue_mutex);
    if (g_cancel_tts_requests.load()) return;
    if (g_tts_muted.load()) return;
    g_tts_queue.push_back(std::move(req));
    g_tts_queue_cv.notify_one();
}


// --- 18.2. Постановка с разбиением по '\n' ---
// Один TtsRequest = один HTTP. Если текст содержит '\n',
// разбиваем на несколько TtsRequest ДО очереди.
static void enqueue_tts_maybe_split(TtsRequest req) {
    if (req.text.find('\n') == std::string::npos) {
        enqueue_tts(std::move(req));
        return;
    }
    std::string current;
    for (char c : req.text) {
        if (c == '\n') {
            if (!current.empty()) {
                TtsRequest r = req;
                r.text = current;
                enqueue_tts(std::move(r));
            }
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        req.text = std::move(current);
        enqueue_tts(std::move(req));
    }
}


// --- 18.3. Очистка очереди ---
static void clear_tts_queue() {
    std::lock_guard<std::mutex> lock(g_tts_queue_mutex);
    g_tts_queue.clear();
}


// --- 18.4. TTS-воркер ---
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
        if (g_tts_muted.load()) continue;

        g_tts_busy.store(true);
        send_tts_async(req);
        g_tts_busy.store(false);
    }
}


// ============================================================================
// 19. КОНСОЛЬНЫЙ ВЫВОД
// ============================================================================
// Назначение: единая точка печати в консоль — имени, реплики, статуса,
//   заголовка окна.
//
// Инварианты:
//   • g_console_mutex — recursive: print_replica может вызываться из вложенных.
//   • g_need_blank_before_next = true ставится после append_status_to_current_line.
//   • print_user_prompt выставляет g_initial_prompt_printed = true.
// Используется: 22.35, 23.8, 23.9, 23.18, 23.21.

// --- 19.1. Выравнивание имени до 4 символов ---
static std::string make_padded_name(const std::string& name) {
    std::string padded = name;
    while (utf8_length(padded) < 4) padded += " ";
    return padded;
}


// --- 19.2. Разделитель пар реплик ---
// Печатает пустую строку, только если g_need_blank_before_next == true.
static void begin_new_pair() {
    if (g_need_blank_before_next) {
        std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
        printf("\n");
        fflush(stdout);
    }
}


// --- 19.3. Печать реплики ---
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


// --- 19.4. Печать префикса бота ---
// Печатает «Эмма: » без перевода строки — токены дописываются в ту же строку.
static void print_bot_prefix(const std::string& bot_name) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    std::string padded = make_padded_name(bot_name);
    printf("%s%s%s: ", C_BOT, padded.c_str(), C_RESET);
    fflush(stdout);
}


// --- 19.5. Печать приглашения пользователя ---
// Печатает «Адам: » без перевода строки — реплика допишется в 23.8.
static void print_user_prompt(const std::string& person) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    if (g_need_blank_before_next) {
        printf("\n");
        fflush(stdout);
    }
    std::string padded = make_padded_name(person);
    printf("%s%s%s: ", C_USER, padded.c_str(), C_RESET);
    fflush(stdout);
    g_initial_prompt_printed = true;
    g_need_blank_before_next = false;
}


// --- 19.6. Обновление заголовка окна ---
// Windows — SetConsoleTitleW; POSIX — ANSI escape.
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


// --- 19.7. Дописать статус в текущую строку ---
// «Эмма: Привет! [tts]\n». После — g_need_blank_before_next = true.
static void append_status_to_current_line(const std::string& status,
                                            const std::string& color) {
    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
    printf(" %s[%s]%s\n", color.c_str(), status.c_str(), C_RESET);
    fflush(stdout);
    g_need_blank_before_next = true;
}

// ============================================================================
// 20. АУДИО-ПОТОК
// ============================================================================
// Назначение: читает микрофон, детектирует речь энергетическим VAD,
//   транскрибирует Whisper, отправляет текст в основной поток через
//   g_pending_llm_request.
//
// ВАЖНО: аудио-поток НЕ трогает TTS-очередь напрямую. При barge-in он
//   только ставит g_interrupt_reason = VAD_SPEECH. Основной цикл в 23.17.1
//   централизованно делает clear_tts_queue / cancel / semaphore.
//
// Инварианты:
//   • Все временные метки — double (см. 5.14).
//   • barge_in_start_ms == 0.0 — нет активного barge-in-кандидата.
//   • speak_len > 0.5 — иначе короткий шум не идёт в Whisper.
//   • no_speech < 0.6 — иначе сегмент считается «нет речи».
// Используется: 22.33 (запуск потока).

void audio_input_thread_func(whisper_context* ctx_wsp,
                             const whisper_params& params,
                             audio_async& audio_ref) {
    g_audio_thread_running.store(true);

    // --- 20.1. Локальное состояние VAD ---
    // double: сравнения с get_current_time_ms() (double) без warning'ов.
    bool speech_active = false;
    double speech_start_ms = 0.0;
    double last_speech_end_ms = get_current_time_ms();
    double barge_in_start_ms = 0.0;

    double silence_timeout_ms = static_cast<double>(params.vad_last_ms);
    if (silence_timeout_ms < 1500.0) silence_timeout_ms = 1500.0;

    const double max_segment_ms = 30000.0;

    bool tts_allowed_prev = true;
    {
        std::string dummy;
        allow_xtts_file(dummy, 1);
    }

    if (g_verbose_mode.load()) {
        fprintf(stderr, "[AudioInput] Поток запущен: silence=%.0f мс, "
                "max_segment=%.0f мс, soft_limit=%d токенов, "
                "interrupt_ms=%d\n",
                silence_timeout_ms, max_segment_ms,
                g_soft_limit_tokens.load(), params.interrupt_ms);
    }

    std::vector<float> pcmf32_cur;
    pcmf32_cur.reserve(2000 * WHISPER_SAMPLE_RATE / 1000);

    while (!g_shutting_down.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(INTERRUPT_CHECK_MS));
        if (g_shutting_down.load()) break;

        // --- 20.2. Принудительный сброс VAD (после команды «стоп») ---
        if (g_force_vad_reset.load()) {
            speech_active = false;
            speech_start_ms = 0.0;
            barge_in_start_ms = 0.0;
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
            continue;
        }

        // --- 20.3. Если идёт сброс контекста — ждём ---
        if (g_reset_in_progress.load()) {
            std::unique_lock<std::mutex> lock(g_reset_mutex);
            g_reset_cv.wait(lock, [] { return !g_reset_in_progress.load(); });
            continue;
        }

        // --- 20.4. Push-to-talk ---
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

        double current_time = get_current_time_ms();

        pcmf32_cur.clear();
        audio_ref.get(2000, pcmf32_cur);
        if (pcmf32_cur.empty()) continue;

        // --- 20.5. Управление семафором XTTS ---
        bool should_allow_tts = !speech_active &&
                                g_interrupt_reason.load() == InterruptReason::NONE &&
                                !g_reset_in_progress.load() &&
                                !g_tts_muted.load();
        if (should_allow_tts != tts_allowed_prev) {
            std::string dummy;
            allow_xtts_file(dummy, should_allow_tts ? 1 : 0);
            tts_allowed_prev = should_allow_tts;
        }

        BotState state = g_bot_state.load();

        // --- 20.6. Энергетический VAD ---
        int vad_result = vad_simple_int(pcmf32_cur, WHISPER_SAMPLE_RATE,
            static_cast<int>(params.vad_last_ms), params.vad_thold,
            params.freq_thold, params.print_energy, params.vad_start_thold);

        // --- 20.7. Мгновенная остановка XTTS при VAD-хите ---
        if (vad_result == 1) {
            std::string dummy;
            allow_xtts_file(dummy, 0);
        }

        // --- 20.8. Barge-in во время генерации ---
        if (state == BotState::GENERATING) {
            if (vad_result == 1) {
                if (barge_in_start_ms == 0.0) {
                    barge_in_start_ms = current_time;
                }
                double barge_duration_ms = (current_time - barge_in_start_ms) * 1000.0;
                if (barge_duration_ms >= static_cast<double>(params.interrupt_ms)) {
                    g_interrupt_reason.store(InterruptReason::VAD_SPEECH);
                    g_bot_state.store(BotState::INTERRUPTED);
                    barge_in_start_ms = 0.0;

                    speech_active = false;
                    speech_start_ms = 0.0;
                    last_speech_end_ms = current_time;
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        g_accumulated_text.clear();
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_display_mutex);
                        g_display_text.clear();
                    }

                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Barge-in: %.0f мс речи\n",
                                barge_duration_ms);
                    }
                }
            } else {
                barge_in_start_ms = 0.0;
            }
            continue;
        }

        barge_in_start_ms = 0.0;

        if (state != BotState::IDLE) continue;

        // --- 20.9. Начало речи ---
        if (vad_result == 1 && !speech_active) {
            speech_active = true;
            speech_start_ms = current_time;

            std::string dummy;
            allow_xtts_file(dummy, 0);

            if (!pcmf32_cur.empty()) {
                float warmup_prob = 0.0f;
                float warmup_no_speech = 0.0f;
                int64_t warmup_t_ms = 0;
                std::string warmup_prompt;
                {
                    std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
                    warmup_prompt = g_whisper_prompt;
                }
                (void)transcribe(ctx_wsp, params, pcmf32_cur,
                                 warmup_prompt, warmup_prob, warmup_no_speech,
                                 warmup_t_ms, params.translate);
            }

            if (g_verbose_mode.load()) {
                fprintf(stderr, "[AudioInput] Начало речи (%.3f)\n", current_time);
            }
        }

        // --- 20.10. Аварийное сегментирование длинной речи ---
        // Если речь идёт > max_segment_ms — режем и отправляем накопленное.
        if (speech_active && (vad_result == 0 || vad_result == 1)) {
            double speech_duration = current_time - speech_start_ms;

            if (speech_duration > max_segment_ms) {
                std::vector<float> pcmf32_seg;
                audio_ref.get(params.voice_ms, pcmf32_seg);
                if (!pcmf32_seg.empty()) {
                    float prob = 0.0f;
                    float no_speech = 0.0f;
                    int64_t t_ms = 0;
                    std::string whisper_prompt_copy;
                    {
                        std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
                        whisper_prompt_copy = g_whisper_prompt;
                    }
                    std::string seg_text = transcribe(ctx_wsp, params,
                        pcmf32_seg, whisper_prompt_copy, prob, no_speech, t_ms,
                        params.translate);
                    trim(seg_text);
                    if (!seg_text.empty() && !is_hallucination(seg_text) &&
                        no_speech < 0.6f) {
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

        // --- 20.11. Конец речи ---
        if (vad_result == 2 && speech_active) {
            speech_active = false;
            double speech_len = current_time - speech_start_ms;
            last_speech_end_ms = current_time;

            if (speech_len > 0.5) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                std::vector<float> pcmf32_full;
                audio_ref.get(params.voice_ms, pcmf32_full);

                const int padding = (WHISPER_SAMPLE_RATE * 300) / 1000;
                pcmf32_full.insert(pcmf32_full.end(), padding, 0.0f);

                if (!pcmf32_full.empty()) {
                    float prob = 0.0f;
                    float no_speech = 0.0f;
                    int64_t t_ms = 0;
                    std::string whisper_prompt_copy;
                    {
                        std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
                        whisper_prompt_copy = g_whisper_prompt;
                    }
                    std::string text = transcribe(ctx_wsp, params, pcmf32_full,
                        whisper_prompt_copy, prob, no_speech, t_ms, params.translate);
                    trim(text);
                    if (!text.empty() && !is_hallucination(text) && no_speech < 0.6f) {
                        std::string text_lower = LowerCase(text);
                        bool command_like = looks_like_command(text_lower);
                        bool short_reply = is_short_valid_reply(text_lower);

                        if (!command_like && !short_reply &&
                            speech_len > 2.0 && utf8_length(text) < 5) {
                            if (g_verbose_mode.load()) {
                                fprintf(stderr, "[Whisper] Отброшено: %.1f сек речи, "
                                                "транскрипция '%s' (%d символов)\n",
                                        speech_len, text.c_str(), utf8_length(text));
                            }
                            text = "";
                        } else if ((command_like || short_reply) && g_verbose_mode.load()) {
                            fprintf(stderr, "[Whisper] Оставлено: '%s' (%s)\n",
                                    text.c_str(),
                                    command_like ? "команда" : "короткий ответ");
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

        // --- 20.12. Тишина — проверка таймаута ---
        if (!speech_active) {
            double silence_ms = (current_time - last_speech_end_ms) * 1000.0;
            if (silence_ms > silence_timeout_ms) {
                std::string acc;
                {
                    std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                    acc = g_accumulated_text;
                }

                bool should_continue = false;
                if (params.auto_continue && !g_command_in_progress.load()) {
                    std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                    double time_since_interrupt =
                        (current_time - g_saved_timestamp) * 1000.0;
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
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Автопродолжение\n");
                    }
                }
                else if (!acc.empty() && !g_pending_llm_request.load()) {
                    {
                        std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
                        g_pending_llm_text = acc;
                        g_pending_llm_request.store(true);
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        g_accumulated_text.clear();
                    }
                    last_speech_end_ms = current_time;
                    if (g_verbose_mode.load()) {
                        fprintf(stderr, "[AudioInput] Отправка %zu символов в LLaMA\n",
                                acc.size());
                    }
                }
            }

            // --- 20.13. Мягкий лимит токенов (по символам, ×3) ---
            {
                std::string snapshot;
                {
                    std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                    if (!g_accumulated_text.empty() &&
                        !g_pending_llm_request.load()) {
                        snapshot = g_accumulated_text;
                    }
                }
                const int soft_limit_chars =
                    g_soft_limit_tokens.load() * 3;
                if (!snapshot.empty() &&
                    static_cast<int>(snapshot.size()) >= soft_limit_chars &&
                    !g_reset_in_progress.load()) {
                    std::string to_send;
                    {
                        std::lock_guard<std::mutex> lock(g_text_accumulator_mutex);
                        to_send = g_accumulated_text;
                        g_accumulated_text.clear();
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
                        g_pending_llm_text = to_send;
                        g_pending_llm_request.store(true);
                    }
                    last_speech_end_ms = current_time;
                    if (g_verbose_mode.load()) {
                        fprintf(stderr,
                            "[AudioInput] Мягкий лимит: отправка %zu символов\n",
                            to_send.size());
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
// 21. ПОТОКИ ВВОДА И ГОРЯЧИХ КЛАВИШ
// ============================================================================
// Назначение:
//   input_thread_func      — читает stdin построчно, кладёт в input_queue.
//   keyboard_shortcut_func — опрашивает GetAsyncKeyState (Windows) для
//                            Ctrl+Space / Ctrl+Right / Ctrl+Delete / Ctrl+R /
//                            Alt (push-to-talk).
//
// Инварианты:
//   • input_thread_func: пустые строки не кладутся в очередь.
//   • keyboard_shortcut_func: обрабатывает нажатие «фронтом» (edge),
//     а не «уровнем» — чтобы не срабатывало каждый тик.
// Используется: 22.33 (запуск потоков).

// --- 21.1. Чтение клавиатуры ---
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


// --- 21.2. Проверка фокуса консоли (Windows) ---
// На POSIX всегда true — фокус не отслеживаем.
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


// --- 21.3. Горячие клавиши ---
// Windows: опрос GetAsyncKeyState. POSIX: заглушка (только sleep).
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
                b_ctr_space_prev = b_ctr_space;
                b_ctr_right_prev = b_ctr_right;
                b_ctr_delete_prev = b_ctr_delete;
                b_ctr_r_prev = b_ctr_r;
                std::this_thread::sleep_for(std::chrono::milliseconds(INTERRUPT_CHECK_MS));
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
        std::this_thread::sleep_for(std::chrono::milliseconds(INTERRUPT_CHECK_MS));
    }
#else
    while (g_shortcut_thread_running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#endif
}

// ============================================================================
// 22. ОСНОВНАЯ ФУНКЦИЯ RUN() — ИНИЦИАЛИЗАЦИЯ
// ============================================================================
// Назначение: точка входа приложения. Настраивает Whisper, LLaMA, аудио,
//   запускает фоновые потоки, входит в основной цикл обработки реплик.
//
// Ключевые инварианты:
//   • Системный промпт токенизируется ОДИН РАЗ (22.23).
//   • Каждый ход в 23.15 добавляет ТОЛЬКО user turn + bot prefix.
//   • BOS добавляется через ensure_bos_in_prompt — универсально для всех
//     пресетов.
//   • n_keep = max(512, prompt_size + 64), ограничение ctx/2.
//   • Сэмплеры: smpl (temp), smpl_high_temp (temp=2.0, антипромпты,
//     loop detection), smpl_regen (temp+0.5f, regen).
// Используется: 25 (wmain/main).

int run(int argc, char** argv) {
    whisper_params params;
    int input_tokens_count = 0;
    // double: статистика от get_current_time_ms() (double), без warning'ов.
    double llama_time_input = 0.0;
    double llama_time_output = 0.0;
    llama_sampler* smpl = nullptr;
    llama_sampler* smpl_high_temp = nullptr;
    llama_sampler* smpl_regen = nullptr;

    // --- 22.1. Парсинг аргументов ---
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }
        if (params.language != "auto" &&
        whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n",
                params.language.c_str());
        whisper_print_usage(argc, argv, params);
        return 1;
    }

    // --- 22.2. Автонастройка параметров от ctx_size ---
    {
        if (params.ctx_size < 2048) params.ctx_size = 2048;

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

        if (params.verbose) {
            fprintf(stderr, "[Auto] n_predict=%d, batch_size=%d, "
                    "min_tokens=%d, n_keep=%s, "
                    "cache_k='%s', cache_v='%s'\n",
                    params.n_predict, params.batch_size, params.min_tokens,
                    params.auto_n_keep ? "auto" : std::to_string(params.n_keep).c_str(),
                    params.cache_type_k.empty() ? "f16(default)" : params.cache_type_k.c_str(),
                    params.cache_type_v.empty() ? "f16(default)" : params.cache_type_v.c_str());
        }
    }

    // --- 22.3. Валидация cache_type при K-shift ---
    // На CUDA кэш Q8_0/Q4_0/... не поддерживает сдвиг — включаем
    // hard-truncate вместо K-shift.
    bool k_shift_enabled = true;
    if (!params.cache_type_k.empty() || !params.cache_type_v.empty()) {
        auto is_shift_incompatible = [](const std::string& s) {
            if (s.empty()) return false;
            std::string low = s;
            std::transform(low.begin(), low.end(), low.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            return low == "q8_0" || low == "q4_0" || low == "q4_1" ||
                   low == "q5_0" || low == "q5_1";
        };
        if (is_shift_incompatible(params.cache_type_k) ||
            is_shift_incompatible(params.cache_type_v)) {
            if (params.verbose) {
                fprintf(stderr, "[KShift] cache-type-k='%s', cache-type-v='%s' "
                        "несовместимы с K-shift на CUDA. "
                        "Включаю hard-truncate вместо K-shift.\n",
                        params.cache_type_k.c_str(),
                        params.cache_type_v.c_str());
            }
            k_shift_enabled = false;
        }
    }

    // --- 22.4. Fallback-промпт (если --prompt-file не задан) ---
    if (params.prompt.empty()) {
        fprintf(stderr,
            "\033[33m"
            "Warning: --prompt-file not provided or not loaded. "
            "Using built-in fallback prompt.\n"
            "\033[0m");

        const bool en = (params.language == "en");
        std::string fb = en ? k_fallback_persona_en : k_fallback_persona_ru;

        if (params.bot_gender == "female") {
            fb += en ? " {1} is a woman." : " {1} — женщина.";
        } else if (params.bot_gender == "male") {
            fb += en ? " {1} is a man." : " {1} — мужчина.";
        }

        if (params.person_gender == "male") {
            fb += en ? " {0} is a man, use masculine forms."
                     : " {0} — мужчина, обращайся в мужском роде.";
        } else if (params.person_gender == "female") {
            fb += en ? " {0} is a woman, use feminine forms."
                     : " {0} — женщина, обращайся в женском роде.";
        }

        params.prompt = fb;
    }

    // --- 22.5. Начальное значение семафора XTTS ---
    {
        std::string control_path;
        allow_xtts_file(control_path, 1);
    }

    // --- 22.6. Подмена логгера Whisper ---
    whisper_log_set(whisper_log_filtered, nullptr);

    // --- 22.7. Инициализация Whisper ---
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

    // --- 22.8. Инициализация LLaMA ---
    llama_backend_init();

    // Фильтр логов: пропускаем только ошибки. Убирает из консоли
    // llama_model_loader, sched_reserve, print_info, sched_graph_realloc.
    llama_log_set([](ggml_log_level level, const char * text, void * /*ud*/) {
        if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

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

    llama_context_params lcparams = llama_context_default_params();
    lcparams.n_ctx = params.ctx_size;
    lcparams.n_batch = params.batch_size;
    lcparams.n_ubatch = std::min(params.batch_size, 512);
    if (params.verbose) {
        fprintf(stdout, "n_ctx %d, n_batch %d, n_ubatch %d\n",
                lcparams.n_ctx, lcparams.n_batch, lcparams.n_ubatch);
    }
    lcparams.n_threads = params.n_threads;

    if (params.flash_attn_disable) {
        lcparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (params.verbose) fprintf(stderr, "[FA] DISABLED (--no-flash-attn)\n");
    } else if (params.flash_attn) {
        lcparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (params.verbose) fprintf(stderr, "[FA] ENABLED (--flash-attn)\n");
    } else {
        lcparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (params.verbose) fprintf(stderr, "[FA] DISABLED (default)\n");
    }

    if (!params.cache_type_k.empty()) {
        lcparams.type_k = parse_cache_type_str(params.cache_type_k);
    }
    if (!params.cache_type_v.empty()) {
        lcparams.type_v = parse_cache_type_str(params.cache_type_v);
    }

    struct llama_context* ctx_llama = llama_init_from_model(model_llama, lcparams);
    if (!ctx_llama) {
        fprintf(stderr, "error: failed to initialize llama context\n");
        llama_model_free(model_llama);
        whisper_free(ctx_wsp);
        llama_backend_free();
        return 1;
    }

    g_ctx_llama_atomic.store(ctx_llama);

    // --- 22.9. Проверка мультиязычности Whisper ---
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

    // --- 22.10. Инициализация аудио-захвата ---
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

    // --- 22.11. Промпт LLaMA (базовый) ---
    std::string prompt_llama = params.prompt;

    // --- 22.12. Whisper-промпт (языкозависимый) ---
    // В режиме auto не подсказываем Whisper имена и язык диалога:
    // иначе prompt может смещать автоматическое распознавание в сторону русского.
    {
        std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);

        if (params.language == "auto") {
            g_whisper_prompt.clear();
        } else {
            std::string tmpl = (params.language == "en")
                ? k_whisper_prompt_en
                : k_whisper_prompt_ru;

            g_whisper_prompt = string_replace_all(tmpl, "{0}", params.person);
            g_whisper_prompt = string_replace_all(
                g_whisper_prompt, "{1}", params.bot_name);
        }
    }
    if (params.verbose) {
        std::string wp;
        {
            std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
            wp = g_whisper_prompt;
        }
        fprintf(stderr, "[Whisper] Prompt (initial_prompt for bias): '%s'\n",
                wp.c_str());
    }

    // --- 22.13. Загрузка instruct-пресета ---
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

    // --- 22.14. Сборка производных из пресета ---
    PresetDerived preset_d = build_preset_derived(params.instruct_preset_data);

    {
        preset_d.statuses.insert("call " + LowerCase(params.bot_name));
    }

    // --- 22.15. Ручная сборка из пресета ---
    params.chat_template = LLM_CHAT_TEMPLATE_UNKNOWN;
    if (params.verbose) {
        fprintf(stderr, "[ChatML] Ручная сборка из пресета '%s'.\n",
                params.instruct_preset.c_str());
    }

    // --- 22.16. Дата и время ---
    // Текущие дата и время передаются модели при каждом ходе динамически вместе с репликой пользователя.
    const std::string ymd;

    // --- 22.17. Текущие значения (изменяются командой call) ---
    std::string default_bot_name = params.bot_name;
    std::string default_voice = params.xtts_voice;
    std::string default_prompt = prompt_llama;
    std::string current_bot_name = default_bot_name;
    std::string current_voice = default_voice;
    std::string current_prompt = default_prompt;

    // --- 22.18. Альтернативный промпт (Лео) ---
    std::string alt_prompt = default_prompt;
    if (!params.alt_prompt_file.empty()) {
        std::ifstream file(params.alt_prompt_file);
        if (file.is_open()) {
            std::copy(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>(),
                      std::back_inserter(alt_prompt));
            alt_prompt = string_replace_all(alt_prompt, "{0}", params.person);
            alt_prompt = string_replace_all(alt_prompt, "{1}", LEO_NAME);
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

    // --- 22.19. Подстановка плейсхолдеров в основной промпт ---
    prompt_llama = string_replace_all(prompt_llama, "{0}", params.person);
    prompt_llama = string_replace_all(prompt_llama, "{1}", default_bot_name);
    prompt_llama = string_replace_all(prompt_llama, "{4}", chat_symb);
    prompt_llama = string_replace_all(prompt_llama, "{5}", ymd);

    // --- 22.19.1. Шаблон разговора (few-shot) ---
    // Добавляется ВСЕГДА, независимо от источника личности.
    {
        const std::string& conv = (params.language == "en")
            ? k_conv_template_en : k_conv_template_ru;
        std::string conv_sub = string_replace_all(conv, "{0}", params.person);
        conv_sub = string_replace_all(conv_sub, "{1}", default_bot_name);
        prompt_llama += "\n\n";
        prompt_llama += conv_sub;
    }

    default_prompt = prompt_llama;
    current_prompt = default_prompt;

    // --- 22.20. Батч LLaMA ---
    llama_batch batch = llama_batch_init(params.batch_size, 0, 1);
    if (params.verbose) fprintf(stdout, "llama_n_ctx %d\n", llama_n_ctx(ctx_llama));

    // --- 22.21. Сэмплеры ---
    // Три цепочки:
    //   smpl            — обычная (temp = params.temp), конец — dist(seed).
    //   smpl_high_temp  — temp = 2.0, для антипромптов и loop detection.
    //   smpl_regen      — temp = params.temp + 0.5f, для regen.
    const float top_k = static_cast<float>(params.top_k);
    const float top_p = params.top_p;
    const float min_p = params.min_p;
    float temp = params.temp;
    const int seed = 0;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab_llama);
    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);
    smpl_high_temp = llama_sampler_chain_init(sparams);
    smpl_regen = llama_sampler_chain_init(sparams);

    if (temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            n_vocab,
            params.repeat_last_n,
            params.repeat_penalty,
            0.0f,
            0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_penalties(
            n_vocab,
            params.repeat_last_n,
            params.repeat_penalty,
            0.0f,
            0.0f));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_temp(2.00f));
        llama_sampler_chain_add(smpl_high_temp, llama_sampler_init_dist(seed));

        // smpl_regen: temp = params.temp + 0.5f — для regen на один ход.
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_penalties(
            n_vocab,
            params.repeat_last_n,
            params.repeat_penalty,
            0.0f,
            0.0f));
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_temp(params.temp + 0.5f));
        llama_sampler_chain_add(smpl_regen, llama_sampler_init_dist(seed));
    }
    else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        // smpl_regen не инициализирован — при greedy regen бессмысленен.
        llama_sampler_free(smpl_regen);
        smpl_regen = nullptr;
    }

    // --- 22.22. ВЫЧИСЛЕНИЕ n_keep ---
    // n_keep = max(512, prompt_size + 64), ограничение ctx/2.
    int n_keep = params.n_keep + (add_bos_token ? 1 : 0);
    auto recalc_n_keep = [&](int prompt_tokens) {
        if (!params.auto_n_keep) {
            n_keep = params.n_keep + (add_bos_token ? 1 : 0);
            return;
        }
        int computed = std::max(N_KEEP_MIN, prompt_tokens + N_KEEP_SLACK);
        int cap = params.ctx_size / 2;
        if (computed > cap) computed = cap;
        params.n_keep = computed;
        n_keep = params.n_keep + (add_bos_token ? 1 : 0);
        if (params.verbose) {
            fprintf(stderr, "[Auto] n_keep=%d (prompt_size=%d, ctx=%d, cap=%d)\n",
                    params.n_keep, prompt_tokens, params.ctx_size, cap);
        }
    };

    // --- 22.23. НАЧАЛЬНАЯ ТОКЕНИЗАЦИЯ СИСТЕМНОГО ПРОМПТА ---
    std::vector<llama_token> embd_inp;
    {
        std::string updated_system_prompt = prompt_llama;

        auto append_part = [&](const std::string& part, bool parse_special) {
            if (part.empty()) return;
            std::vector<llama_token> part_tokens =
                llama_tokenize_ex(ctx_llama, part, /*add_special=*/false, parse_special);
            embd_inp.insert(embd_inp.end(), part_tokens.begin(), part_tokens.end());
        };

        append_part(preset_d.sys_prefix, true);
        append_part(updated_system_prompt, false);
        append_part(preset_d.sys_suffix, true);

        ensure_bos_in_prompt(embd_inp, ctx_llama, vocab_llama,
                             add_bos_token, preset_d);
    }

    // --- 22.23.1. Пересчёт n_keep по факту токенизации ---
    recalc_n_keep(static_cast<int>(embd_inp.size()));
    if (params.auto_n_keep &&
        static_cast<int>(embd_inp.size()) > params.n_keep) {
        fprintf(stderr, "[Auto] WARNING: system prompt (%zu) > n_keep (%d). "
                        "Будет обрезка в 22.24.\n",
                embd_inp.size(), params.n_keep);
    }

    // --- 22.24. Обрезка слишком большого промпта ---
    if (static_cast<int>(embd_inp.size()) > params.ctx_size - 512) {
        int keep = std::min(n_keep, static_cast<int>(embd_inp.size()));
        if (static_cast<int>(embd_inp.size()) > keep + 256) {
            embd_inp.erase(embd_inp.begin() + keep, embd_inp.end() - 256);
        }
        std::cerr << "[warn] Context trimmed: " << embd_inp.size()
                  << " tokens (ctx limit " << params.ctx_size << ")\n";
    }

    // --- 22.25. Загрузка сессии ---
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

    // --- 22.26. ПРОГРЕВ КОНТЕКСТА ---
    if (params.verbose) printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);
    double llama_start_time = get_current_time_ms();
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
    double llama_end_time = get_current_time_ms();
    double llama_time_total = llama_end_time - llama_start_time;
    if (llama_time_total > 0.001) {
        printf("\nLlama start prompt: %zu/%d tokens in %.3f s at %.0f t/s\n",
               embd_inp.size(), params.ctx_size,
               llama_time_total,
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

    // --- 22.27. Сравнение сессии ---
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
    if (need_to_save_session && !path_session.empty()) {
        session_tokens = embd_inp;
    }

    // --- 22.28. Приветствие ---
    printf("%s : done! start speaking in the microphone\n", __func__);
    const std::string wake_cmd = params.wake_cmd;
    if (!wake_cmd.empty())
        printf("%s : the wake-up command is: '%s'\n", __func__, wake_cmd.c_str());
    printf("\n\033[90mVoice commands: Stop(Ctrl+Space), Regenerate(Ctrl+Right), "
           "Delete(Ctrl+Delete), Reset(Ctrl+R)\033[0m\n\n");
    if (params.push_to_talk)
        printf("\033[90mPush-to-talk: hold 'Alt' to speak\033[0m\n\n");
    fflush(stdout);

    // --- 22.29. Флаги состояния ---
    g_bot_state.store(BotState::IDLE);
    g_interrupt_reason.store(InterruptReason::NONE);
    g_interrupt_processed.store(false);
    g_shutting_down.store(false);
    g_reset_in_progress.store(false);
    g_cancel_tts_requests.store(false);
    g_tts_muted.store(false);
    g_tts_busy.store(false);
    g_print_special.store(params.print_special);
    // Инициализируем глобалы для sanitize_user_text: чтобы случайная
    // вставка "Адам: ..." / "Эмма: ..." в user turn не воспринималась
    // моделью как готовая история диалога.
    g_sanitize_person = params.person;
    g_sanitize_bot = current_bot_name;
    audio.clear();
    const int n_ctx = llama_n_ctx(ctx_llama);
    n_past = static_cast<int>(embd_inp.size());

    std::vector<int> past_prev_arr{};
    int n_past_prev = 0;
    int n_session_consumed = !path_session.empty() && session_tokens.size() > 0
        ? static_cast<int>(session_tokens.size()) : 0;
    std::vector<llama_token> embd;
    std::string text_heard_prev;
    std::string text_heard_trimmed;
    int new_command_allowed = 1;

    // --- 22.30. Локальные переменные для генерации ---
    std::chrono::steady_clock::time_point last_command_time =
        std::chrono::steady_clock::now();

    // --- 22.31. Антипромпты ---
    // stop_strings (из пресета) + хардкод-маркеры eot + (опционально '\n')
    // + "Адам: " + stop_words.
    std::vector<std::string> antiprompts = preset_d.antiprompts;

    antiprompts.push_back("</end_of_turn>");

    for (const char * ap : {"<|eot_id|>", "<|im_end|>", "<end_of_turn>",
                            "<|end|>", "<|endoftext|>"}) {
        if (std::find(antiprompts.begin(), antiprompts.end(), ap)
            == antiprompts.end()) {
            antiprompts.push_back(ap);
        }
    }

    if (!params.allow_newline) {
        antiprompts.push_back("\n");
        if (params.verbose) {
            fprintf(stderr, "[Antiprompts] Добавлен '\\n' (нет --allow-newline)\n");
        }
    }

    antiprompts.push_back(params.person + chat_symb);

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

    {
        std::sort(antiprompts.begin(), antiprompts.end());
        antiprompts.erase(std::unique(antiprompts.begin(), antiprompts.end()),
                          antiprompts.end());
    }

    // --- 22.32. Мягкий лимит токенов ---
    g_soft_limit_tokens.store(params.ctx_size / 3);
    if (params.verbose) {
        printf("[Limits] soft_limit_tokens=%d (ctx_size=%d / 3)\n",
               g_soft_limit_tokens.load(), params.ctx_size);
    }

    // --- 22.33. Запуск фоновых потоков ---
    std::thread input_thread(input_thread_func);
    std::thread shortcut_thread([&]() { keyboard_shortcut_func(params); });
    std::thread audio_thread([&]() {
        audio_input_thread_func(ctx_wsp, params, audio);
    });
    std::thread tts_worker(tts_worker_func);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(g_display_mutex);
        g_display_text.clear();
    }

    llama_start_time = 0.0;
    double llama_start_generation_time = 0.0;
    llama_end_time = 0.0;
    llama_time_total = 0.0;
    std::string user_typed = "";
    bool user_typed_this = false;

    // --- 22.34. Заголовок окна ---
    update_console_title(current_bot_name, "Listening");

    // --- 22.35. Стартовое приглашение ---
    print_user_prompt(params.person);

    // ========================================================================
    // 22.36. ХЕЛПЕРЫ ДЛЯ ОСНОВНОГО ЦИКЛА
    // ========================================================================
    // Назначение: make_tts_req, wait_for_tts_idle, rebuild_kv_from_current_prompt.
    // Инварианты:
    //   • make_tts_req заполняет все поля TtsRequest из текущих значений.
    //   • wait_for_tts_idle ждёт не дольше sleep_before_xtts и прерывается
    //     при g_interrupt_reason != NONE.
    //   • rebuild_kv_from_current_prompt берёт unique_lock, чистит KV,
    //     пересобирает embd_inp из current_prompt.

    auto make_tts_req = [&](const std::string& text) {
        TtsRequest req;
        req.text = text;
        req.voice = current_voice;
        // "auto" относится к распознаванию Whisper, а не к языку XTTS.
        // До полноценной передачи определённого языка в TTS используем русский
        // как язык приложения по умолчанию.
        req.language = (params.language == "auto")
            ? "ru"
            : params.language;
        req.url = params.xtts_url;
        req.stop_seq = preset_d.stop_sequence;
        req.bot_sfx = preset_d.bot_suffix;
        req.user_sfx = preset_d.user_suffix;
        req.bot_pfx = preset_d.bot_prefix;
        req.user_pfx = preset_d.user_prefix;
        req.chat_symb = chat_symb;
        req.person = params.person;
        req.bot = current_bot_name;
        return req;
    };

    auto wait_for_tts_idle = [&]() {
        if (params.sleep_before_xtts <= 0) return;
        if (!g_tts_busy.load()) return;

        int remaining = params.sleep_before_xtts;
        while (g_tts_busy.load() && remaining > 0 &&
               g_interrupt_reason.load() == InterruptReason::NONE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            remaining -= 20;
        }
    };

    auto rebuild_kv_from_current_prompt = [&]() -> bool {
        std::unique_lock<std::shared_mutex> wlock(g_llama_rw_mutex);
        llama_context* active = g_ctx_llama_atomic.load();
        if (!active) {
            fprintf(stderr, "[rebuild_kv] ERROR: ctx_llama is null\n");
            return false;
        }

        llama_memory_seq_rm(llama_get_memory(active), 0, -1, -1);

        embd_inp.clear();
        auto append_part = [&](const std::string& part, bool parse_special) {
            if (part.empty()) return;
            std::vector<llama_token> toks =
                llama_tokenize_ex(active, part, /*add_special=*/false, parse_special);
            embd_inp.insert(embd_inp.end(), toks.begin(), toks.end());
        };
        append_part(preset_d.sys_prefix, true);
        append_part(current_prompt, false);
        append_part(preset_d.sys_suffix, true);
        ensure_bos_in_prompt(embd_inp, active, vocab_llama, add_bos_token, preset_d);

        if (embd_inp.empty()) {
            fprintf(stderr, "[rebuild_kv] ERROR: empty embd_inp\n");
            return false;
        }

        size_t total = embd_inp.size();
        size_t offset = 0;
        while (offset < total) {
            size_t chunk = std::min(static_cast<size_t>(params.batch_size),
                                    total - offset);
            batch.n_tokens = static_cast<int>(chunk);
            for (size_t i = 0; i < chunk; ++i) {
                batch.token[i] = embd_inp[offset + i];
                batch.pos[i] = static_cast<int>(offset + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = ((offset + i) == total - 1) ? 1 : 0;
            }
            if (llama_decode(active, batch)) {
                fprintf(stderr, "[rebuild_kv] decode failed at offset %zu\n", offset);
                return false;
            }
            offset += chunk;
        }
        n_past = static_cast<int>(embd_inp.size());
        n_session_consumed = n_past;
        session_tokens = embd_inp;

        recalc_n_keep(static_cast<int>(embd_inp.size()));

        past_prev_arr.clear();
        text_heard_prev.clear();

        return true;
    };

    // ========================================================================
    // 23. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВВОДА И КОМАНД
    // ========================================================================
    // Назначение: каждая итерация —
    //   1) проверяет auto-continue после barge-in;
    //   2) забирает текст из audio-потока (g_pending_llm_request);
    //   3) забирает текст с клавиатуры (input_queue);
    //   4) обрабатывает горячие клавиши (g_hotkey_pressed);
    //   5) разбирает команды;
    //   6) формирует промпт user turn + bot prefix;
    //   7) генерирует ответ;
    //   8) отправляет видимый текст в TTS-буфер;
    //   9) печатает новое приглашение пользователя.
    //
    // Инварианты:
    //   • g_pending_continue_request обрабатывается ДО ввода с клавиатуры.
    //   • «Стоп» — отдельный случай (23.9), до очистки текста.
    //   • Команды (google/call/regen/repeat/delete/reset/time/date)
    //     обрабатываются в 23.13, до генерации.
    //   • show_user_replica — единственная точка печати реплики Адама.

    while (is_running) {

        // --- 23.0. SDL-события ---
        is_running = sdl_poll_events();
        if (!is_running) {
            printf("\n[Shutdown requested, cleaning up...]\n");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::string text_heard = "";
        user_typed_this = false;

        // --- 23.1. Автопродолжение после barge-in ---
        // g_pending_continue_request ставится audio-потоком (20.12) при
        // тишине после barge-in. Восстанавливаем сохранённый снимок
        // embd_inp/n_past — но ТОЛЬКО если он синхронизирован (см. 23.17.1).
        if (g_pending_continue_request.load()) {
            bool can_continue = false;
            {
                std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                can_continue = g_saved_was_interrupted && !g_saved_embd_inp.empty();
                if (can_continue) {
                    embd_inp = g_saved_embd_inp;
                    n_past = g_saved_n_past;
                    g_saved_was_interrupted = false;
                }
            }
            g_pending_continue_request.store(false);
            if (!can_continue) {
                // Рассинхронизация или пустой снимок — auto-continue
                // невозможен. Пропускаем ход, ждём новой реплики.
                // n_past и embd_inp остаются согласованными.
                continue;
            }
            text_heard = "";
            user_typed = "";
            force_speak = true;
            // Пользователь не говорил — реплики Адама нет. Сбрасываем
            // флаг "приглашение уже напечатано", чтобы print_bot_prefix
            // не приклеился к пустому "Адам: ". g_need_blank_before_next
            // уже true (после [int]) — пустая строка между парами сохранится.
            g_initial_prompt_printed = false;
        }

        // --- 23.2. Запрос из audio-потока ---
        {
            std::lock_guard<std::mutex> lock(g_pending_llm_mutex);
            if (g_pending_llm_request.load() && !g_pending_llm_text.empty()) {
                user_typed = g_pending_llm_text;
                user_typed_this = true;
                g_pending_llm_text.clear();
                g_pending_llm_request.store(false);
            }
        }

        // --- 23.3. Клавиатурный ввод ---
        // Если пользователь ввёл с клавиатуры — отменяем pending_llm_request,
        // чтобы не смешивать голос и текст.
        bool user_input_from_keyboard = false;
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
                    user_input_from_keyboard = true;
                    std::lock_guard<std::mutex> pl(g_pending_llm_mutex);
                    g_pending_llm_request.store(false);
                    g_pending_llm_text.clear();
                }
            }
        }

        // --- 23.4. Горячие клавиши ---
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

        // --- 23.5. Проверка наличия текста ---
        if (user_typed.empty() && !force_speak) continue;
        std::string display_text_for_ui = user_typed;
        trim(display_text_for_ui);
        text_heard = user_typed;
        user_typed = "";
        trim(text_heard);
        if (text_heard.empty() && !force_speak) continue;

        // Защита от случайной вставки огромного текста (например,
        // всего предыдущего диалога) в user turn. Обрезаем до
        // MAX_USER_INPUT_CHARS с сохранением UTF-8 границ.
        if (!force_speak &&
            utf8_length(text_heard) > static_cast<int>(MAX_USER_INPUT_CHARS)) {
            int orig_len = utf8_length(text_heard);
            text_heard = utf8_substr(text_heard, 0,
                static_cast<unsigned int>(MAX_USER_INPUT_CHARS));
            if (params.verbose) {
                fprintf(stderr,
                    "[Input] Обрезано: %d -> %zu символов (MAX_USER_INPUT_CHARS)\n",
                    orig_len, MAX_USER_INPUT_CHARS);
            }
        }

        // --- 23.6. Wake command (если задан) ---
        if (!params.wake_cmd.empty() && !force_speak) {
            if (text_heard.find(params.wake_cmd) != 0) continue;
            text_heard = text_heard.substr(params.wake_cmd.length());
            trim(text_heard);
        }

        // --- 23.7. Удаление префикса имени пользователя ---
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

        // --- 23.8. Хелпер печати реплики пользователя ---
        // Если префикс "Адам: " уже напечатан — дописываем текст в ту же
        // строку (даже при typed), иначе строка останется пустой и
        // следующий print_bot_prefix приклеится ("Адам:Эмма:").
        // Если префикса нет — begin_new_pair + print_replica.
        auto show_user_replica = [&](const std::string& text,
                                      const std::string& status,
                                      bool typed) {
            if (g_initial_prompt_printed) {
                std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
                // При вводе с клавиатуры readline уже напечатал text и '\n'
                // после префикса "Адам: ". Повторно печатать text, status
                // и '\n' не нужно — иначе получаем пустую строку
                // ("Адам: привет\n\nЭмма: ...") и оторванный статус
                // ("Адам: текст\n [time]").
                // При голосовом вводе (typed == false) — печатаем всё сами.
                if (!typed) {
                    if (!text.empty()) {
                        printf("%s", text.c_str());
                    }
                    if (!status.empty()) {
                        printf(" %s[%s]%s", C_CMD, status.c_str(), C_RESET);
                    }
                    printf("\n");
                }
                fflush(stdout);
                g_initial_prompt_printed = false;
            } else {
                begin_new_pair();
                print_replica(C_USER, params.person, text, status);
            }
        };

        // ====================================================================
        // 23.9. КОМАНДА «СТОП»
        // ====================================================================
        // Отдельный случай: g_tts_muted + ожидание g_tts_busy.
        // g_saved_was_interrupted = false — auto-continue не сработает.
        // g_force_vad_reset = true — audio-поток сбросит VAD.
        // Если идёт генерация — MANUAL_STOP, иначе NONE + IDLE.
        {
            std::string lower_text = LowerCase(text_heard);
            if (!force_speak && is_stop_command(lower_text)) {
                g_command_in_progress.store(true);

                {
                    std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                    g_saved_was_interrupted = false;
                }

                g_tts_muted.store(true);
                g_cancel_tts_requests.store(true);
                clear_tts_queue();

                {
                    std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
                    if (g_initial_prompt_printed) {
                        printf("%s %s[stop]%s\n",
                               display_text_for_ui.c_str(),
                               C_STOP, C_RESET);
                        g_initial_prompt_printed = false;
                    } else {
                        printf("\n");
                        std::string padded = make_padded_name(params.person);
                        printf("%s%s%s: %s %s[stop]%s\n",
                               C_USER, padded.c_str(), C_RESET,
                               display_text_for_ui.c_str(),
                               C_STOP, C_RESET);
                    }
                    fflush(stdout);
                    g_need_blank_before_next = true;
                }

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

                std::string dummy;
                allow_xtts_file(dummy, 0);

                for (int i = 0; i < 60 && g_tts_busy.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }

                update_console_title(current_bot_name, "Listening");
                g_command_in_progress.store(false);
                print_user_prompt(params.person);
                continue;
            }
        }

        // --- 23.10. Очистка распознанного текста ---
        try {
            static const std::regex re_brackets(R"(\[[^\[\]]*\])");
            text_heard = std::regex_replace(text_heard, re_brackets, "");
        } catch (const std::regex_error&) {}
        if (params.language == "en" && !user_typed_this) {
            static const std::regex re_eng_only("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]");
            text_heard = std::regex_replace(text_heard, re_eng_only, "");
        }
        {
            size_t nl = text_heard.find('\n');
            if (nl != std::string::npos) text_heard.resize(nl);
        }
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

        // --- 23.11. Детектор мусора ---
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
            if (text_heard_trimmed[last_pos] == '.' ||
                text_heard_trimmed[last_pos] == '!')
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

        // --- 23.12. Определение команд ---
        std::string user_command;
        bool is_google = false;

        auto contains_word = [](const std::string& haystack,
                                const std::string& needle) -> bool {
            if (needle.empty()) return false;
            size_t pos = 0;
            while ((pos = haystack.find(needle, pos)) != std::string::npos) {
                bool left_ok = is_utf8_word_boundary(haystack, pos);
                size_t end_pos = pos + needle.size();
                bool right_ok = true;
                if (end_pos < haystack.size()) {
                    std::string right_char = utf8_first_char(haystack.substr(end_pos));
                    right_ok = !utf8_is_word_char(right_char);
                }
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
                            bot_name_accusative.pop_back();
                            bot_name_accusative += "у";
                        }
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
                "перегенерируй",
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
                "удали", "удалить", "delete", "remove", "erase"
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
                "сброс", "сбросить", "reset"
            };
            for (const auto& w : reset_words) {
                if (contains_word(text_heard_trimmed, w)) {
                    user_command = "reset";
                    break;
                }
            }
        }


        // Ограничение частоты команд.
        if (!user_command.empty() && !new_command_allowed) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - last_command_time).count();
            if (elapsed >= 2) new_command_allowed = 1;
        }

        // ====================================================================
        // 23.13. ОБРАБОТЧИКИ КОМАНД
        // ====================================================================
        // Каждая команда обрабатывается своим блоком. После — continue.
        // Исключение: regenerate проваливается в 23.14 (генерация).
        if (!user_command.empty()) {
            g_tts_muted.store(false);
            g_cancel_tts_requests.store(false);
            std::string dummy; allow_xtts_file(dummy, 1);
        }

        bool is_high_priority =
            (user_command == "reset" || user_command == "delete" ||
             user_command == "regenerate");

        // ---- google ----
        if (user_command == "google" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string query = ParseCommandAndGetKeyword(text_heard_trimmed, "google");
            trim(query);
            if (!query.empty()) {
                show_user_replica(text_heard, "google", user_input_from_keyboard);
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
                enqueue_tts_maybe_split(make_tts_req(out_text));
            }
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            update_console_title(current_bot_name, "Listening");
            fflush(stdout);
            g_command_in_progress.store(false);
            print_user_prompt(params.person);
            continue;
        }

        // ---- call_leo ----
        if (user_command == "call_leo" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            current_bot_name = LEO_NAME;
            current_voice = LEO_VOICE;
            current_prompt = alt_prompt;
            g_sanitize_bot = LEO_NAME;
            {
                std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);

                if (params.language == "auto") {
                    g_whisper_prompt.clear();
                } else {
                    std::string tmpl = (params.language == "en")
                        ? k_whisper_prompt_en
                        : k_whisper_prompt_ru;

                    g_whisper_prompt = string_replace_all(tmpl, "{0}", params.person);
                    g_whisper_prompt = string_replace_all(
                        g_whisper_prompt, "{1}", current_bot_name);
                }
            }
            if (params.verbose) {
                std::string wp;
                {
                    std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
                    wp = g_whisper_prompt;
                }
                fprintf(stderr, "[Whisper] Prompt обновлён: '%s'\n", wp.c_str());
            }
            show_user_replica(text_heard, "call leo", user_input_from_keyboard);

            g_reset_in_progress.store(true);
            bool rebuild_ok = rebuild_kv_from_current_prompt();
            g_reset_in_progress.store(false);
            g_reset_cv.notify_all();

            if (!rebuild_ok) {
                fprintf(stderr, "[call_leo] ERROR: KV rebuild failed\n");
                g_command_in_progress.store(false);
                print_user_prompt(params.person);
                continue;
            }

            enqueue_tts(make_tts_req("Я Лео, слушаю"));
            update_console_title(current_bot_name, "Listening");
            fflush(stdout);
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            print_user_prompt(params.person);
            continue;
        }

        // ---- call_default ----
        if (user_command == "call_default" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            current_bot_name = default_bot_name;
            current_voice = default_voice;
            current_prompt = default_prompt;
            g_sanitize_bot = default_bot_name;
            {
                std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);

                if (params.language == "auto") {
                    g_whisper_prompt.clear();
                } else {
                    std::string tmpl = (params.language == "en")
                        ? k_whisper_prompt_en
                        : k_whisper_prompt_ru;

                    g_whisper_prompt = string_replace_all(tmpl, "{0}", params.person);
                    g_whisper_prompt = string_replace_all(
                        g_whisper_prompt, "{1}", current_bot_name);
                }
            }
            if (params.verbose) {
                std::string wp;
                {
                    std::lock_guard<std::mutex> lock(g_whisper_prompt_mutex);
                    wp = g_whisper_prompt;
                }
                fprintf(stderr, "[Whisper] Prompt обновлён: '%s'\n", wp.c_str());
            }
            std::string status = "call " + LowerCase(current_bot_name);
            show_user_replica(text_heard, status, user_input_from_keyboard);

            g_reset_in_progress.store(true);
            bool rebuild_ok = rebuild_kv_from_current_prompt();
            g_reset_in_progress.store(false);
            g_reset_cv.notify_all();

            if (!rebuild_ok) {
                fprintf(stderr, "[call_default] ERROR: KV rebuild failed\n");
                g_command_in_progress.store(false);
                print_user_prompt(params.person);
                continue;
            }

            enqueue_tts(make_tts_req("Слушаю"));
            update_console_title(current_bot_name, "Listening");
            fflush(stdout);
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);
            print_user_prompt(params.person);
            continue;
        }

        // ---- regenerate ----
        // Откатываем embd_inp до n_past_prev (до user turn), восстанавливаем
        // text_heard = text_heard_prev. Ставим g_regen_pending — первый
        // токен следующего хода будет взят из smpl_regen (temp+0.5f),
        // чтобы ответ отличался от предыдущего.
        if (user_command == "regenerate" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string dummy;
            allow_xtts_file(dummy, 0);
            g_cancel_tts_requests.store(true);
            clear_tts_queue();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            new_command_allowed = 0;
            last_command_time = std::chrono::steady_clock::now();

            bool rollback_success = false;
            if (!past_prev_arr.empty()) {
                n_past_prev = past_prev_arr.back();
                past_prev_arr.pop_back();
                int rollback_num = static_cast<int>(embd_inp.size()) - n_past_prev;
                if (rollback_num > 0 && rollback_num <= static_cast<int>(embd_inp.size())) {
                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    if (!path_session.empty()) {
                        if (session_tokens.size() > embd_inp.size())
                            session_tokens.resize(embd_inp.size());
                    }
                    {
                        std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                        llama_context* active = g_ctx_llama_atomic.load();
                        if (active) {
                            llama_memory_seq_rm(llama_get_memory(active), 0,
                                static_cast<int>(embd_inp.size()), -1);
                        }
                    }
                    text_heard = text_heard_prev;
                    trim(text_heard);
                    rollback_success = true;
                    // Флаг: следующий ход — regen. В 23.16 первый токен
                    // будет взят из smpl_regen (temp+0.5f), чтобы получить
                    // ДРУГОЙ ответ на тот же user turn.
                    g_regen_pending.store(true);
                }
            }

            g_cancel_tts_requests.store(false);
            { std::string dummy2; allow_xtts_file(dummy2, 1); }
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            g_command_in_progress.store(false);

            if (!rollback_success || text_heard.empty()) {
                show_user_replica(text_heard, "regen", user_input_from_keyboard);
                print_replica(C_BOT, current_bot_name,
                    "Нечего переделывать — история усечена", "");
                audio.clear();
                print_user_prompt(params.person);
                continue;
            }

            show_user_replica(text_heard, "regen", user_input_from_keyboard);
            display_text_for_ui = text_heard;
            force_speak = false;
            // Проваливаемся в 23.14 (генерация).
        }

        // ---- repeat ----
        if (user_command == "repeat" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            std::string last_text;
            {
                std::lock_guard<std::mutex> lock(g_last_tts_mutex);
                last_text = g_last_tts_text;
            }
            show_user_replica(text_heard, "repeat", user_input_from_keyboard);
            if (!last_text.empty()) {
                print_replica(C_BOT, current_bot_name, last_text, "tts");
                enqueue_tts(make_tts_req(last_text));
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
            print_user_prompt(params.person);
            continue;
        }

        // ---- delete ----
        if (user_command == "delete" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            show_user_replica(text_heard, "del", user_input_from_keyboard);
            if (!past_prev_arr.empty()) {
                n_past_prev = past_prev_arr.back();
                past_prev_arr.pop_back();
                int rollback_num = static_cast<int>(embd_inp.size()) - n_past_prev;
                if (rollback_num > 0 && rollback_num <= static_cast<int>(embd_inp.size())) {
                    embd_inp.erase(embd_inp.end() - rollback_num, embd_inp.end());
                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    if (!path_session.empty()) {
                        if (session_tokens.size() > embd_inp.size())
                            session_tokens.resize(embd_inp.size());
                    }
                    {
                        std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                        llama_context* active = g_ctx_llama_atomic.load();
                        if (active) {
                            llama_memory_seq_rm(llama_get_memory(active), 0,
                                static_cast<int>(embd_inp.size()), -1);
                        }
                    }
                    last_command_time = std::chrono::steady_clock::now();
                    new_command_allowed = 0;
                    enqueue_tts(make_tts_req("Удалено"));
                }
            }
            else {
                enqueue_tts(make_tts_req("Нечего удалять"));
            }
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            fflush(stdout);
            g_command_in_progress.store(false);
            print_user_prompt(params.person);
            continue;
        }

        // ---- reset ----
        // Полная переинициализация контекста: освобождаем ctx_llama,
        // создаём заново, декодируем current_prompt. Дорого, но надёжно.
        if (user_command == "reset" && (is_high_priority || new_command_allowed)) {
            g_command_in_progress.store(true);
            g_reset_in_progress.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::string dummy;
            allow_xtts_file(dummy, 0);
            g_cancel_tts_requests.store(true);
            clear_tts_queue();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            show_user_replica(text_heard, "reset", user_input_from_keyboard);
            if (g_bot_state.load() == BotState::GENERATING) {
                g_interrupt_reason.store(InterruptReason::MANUAL_STOP);
                g_interrupt_processed.store(true);
                g_bot_state.store(BotState::IDLE);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            {
                std::unique_lock<std::shared_mutex> wlock(g_llama_rw_mutex);

                llama_batch_free(batch);
                g_ctx_llama_atomic.store(nullptr);
                if (ctx_llama) {
                    llama_free(ctx_llama);
                    ctx_llama = nullptr;
                }
                ctx_llama = llama_init_from_model(model_llama, lcparams);
                if (!ctx_llama) {
                    fprintf(stderr, "%s : ERROR: Failed to reinitialize "
                            "llama context on reset\n", __func__);
                    g_ctx_llama_atomic.store(nullptr);
                    g_reset_in_progress.store(false);
                    g_reset_cv.notify_all();
                    g_command_in_progress.store(false);
                    g_shutting_down.store(true);
                    is_running = false;
                    break;
                }
                g_ctx_llama_atomic.store(ctx_llama);
                batch = llama_batch_init(params.batch_size, 0, 1);

                embd_inp.clear();
                auto append_reset = [&](const std::string& part, bool parse_special) {
                    if (part.empty()) return;
                    std::vector<llama_token> toks =
                        llama_tokenize_ex(ctx_llama, part, /*add_special=*/false, parse_special);
                    embd_inp.insert(embd_inp.end(), toks.begin(), toks.end());
                };
                append_reset(preset_d.sys_prefix, true);
                append_reset(current_prompt, false);
                append_reset(preset_d.sys_suffix, true);

                ensure_bos_in_prompt(embd_inp, ctx_llama, vocab_llama,
                                     add_bos_token, preset_d);

                if (embd_inp.empty()) {
                    fprintf(stderr, "%s : ERROR: Prompt size invalid\n", __func__);
                    g_reset_in_progress.store(false);
                    g_reset_cv.notify_all();
                    g_command_in_progress.store(false);
                    g_shutting_down.store(true);
                    is_running = false;
                    break;
                }
                size_t total = embd_inp.size();
                size_t offset = 0;
                bool decode_failed = false;
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
                        decode_failed = true;
                        break;
                    }
                    offset += chunk;
                }
                if (decode_failed) {
                    g_reset_in_progress.store(false);
                    g_reset_cv.notify_all();
                    g_command_in_progress.store(false);
                    g_shutting_down.store(true);
                    is_running = false;
                    break;
                }
            }
            n_past = static_cast<int>(embd_inp.size());
            n_session_consumed = n_past;
            recalc_n_keep(static_cast<int>(embd_inp.size()));
            session_tokens = embd_inp;
            text_heard = "";
            text_heard_trimmed = "";

            past_prev_arr.clear();
            text_heard_prev.clear();
            g_transcription_count.store(0);

            g_tts_muted.store(false);
            g_cancel_tts_requests.store(false);

            enqueue_tts(make_tts_req("Контекст сброшен"));

            { std::string dummy2; allow_xtts_file(dummy2, 1); }
            g_reset_in_progress.store(false);
            g_reset_cv.notify_all();
            audio.clear();
            g_bot_state.store(BotState::IDLE);
            g_interrupt_reason.store(InterruptReason::NONE);
            g_interrupt_processed.store(false);
            fflush(stdout);
            g_command_in_progress.store(false);
            print_user_prompt(params.person);
            continue;
        }

        // ====================================================================
        // 23.14. НАЧАЛО ГЕНЕРАЦИИ LLaMA
        // ====================================================================
        // Сюда попадаем либо после обычного user turn, либо после regen
        // (force_speak == false), либо после auto-continue (force_speak == true).
        //
        // Инварианты:
        //   • При !force_speak: text_heard_prev = text_heard, в past_prev_arr
        //     пушится текущий n_past (для будущего delete/regen).
        //   • При force_speak: user turn НЕ добавляется (23.15), embd остаётся
        //     пустым, генерируем продолжение.
        int tokens_in_reply = 0;
        llama_start_generation_time = 0.0;
        llama_start_time = get_current_time_ms();

        if (g_tts_muted.load()) {
            g_tts_muted.store(false);
            g_cancel_tts_requests.store(false);
            std::string dummy;
            allow_xtts_file(dummy, 1);
        }

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

            if (user_command != "regenerate") {
                show_user_replica(text_heard, "", user_input_from_keyboard);
            }
        }
        force_speak = false;

        update_console_title(current_bot_name, "Generating...");
        g_bot_state.store(BotState::GENERATING);

        // ====================================================================
        // 23.15. ФОРМИРОВАНИЕ ПРОМПТА
        // ====================================================================
        // При !force_speak: user_prefix + sanitize_user_text(text_heard)
        //   + user_suffix + bot_prefix.
        // При force_speak: embd пуст (генерируем продолжение).
        //
        // sanitize_user_text удаляет:
        //   • <|...|> — спецтокены;
        //   • "Имя:" в начале строк — защита от вставки диалога.
        {
            embd.clear();

            if (!force_speak) {
                std::string safe_user_text = sanitize_user_text(text_heard);
                // Передаём модели актуальные локальные дату и время на каждом ходе.
                {
                    const std::time_t t_now = std::time(nullptr);
                    std::tm tm_local_now{};
                    bool time_ok = false;

                #ifdef _WIN32
                    time_ok = (localtime_s(&tm_local_now, &t_now) == 0);
                #else
                    time_ok = (localtime_r(&t_now, &tm_local_now) != nullptr);
                #endif

                    if (time_ok) {
                        static const char* months_ru[] = {
                            "января", "февраля", "марта", "апреля", "мая", "июня",
                            "июля", "августа", "сентября", "октября", "ноября", "декабря"
                        };

                        static const char* weekdays_ru[] = {
                            "воскресенье", "понедельник", "вторник", "среда",
                            "четверг", "пятница", "суббота"
                        };

                        char datetime_buffer[256];

                        std::snprintf(
                            datetime_buffer,
                            sizeof(datetime_buffer),
                            "%d %s %d года, %s, %02d:%02d",
                            tm_local_now.tm_mday,
                            months_ru[tm_local_now.tm_mon],
                            tm_local_now.tm_year + 1900,
                            weekdays_ru[tm_local_now.tm_wday],
                            tm_local_now.tm_hour,
                            tm_local_now.tm_min
                        );

                        safe_user_text =
                            std::string(
                                "[Системная справка — актуальные локальные дата и время: "
                            ) + datetime_buffer +
                            ". Это достоверные текущие значения. Если вопрос касается "
                            "времени или даты, используй их, не угадывай. "
                            "Отвечай естественно и учитывай контекст предыдущего разговора. "
                            "Не произноси эту справку как часть ответа.]\n"
                            "Реплика пользователя: " + safe_user_text;
                    }
                }

                auto append_part = [&](const std::string& part, bool parse_special) {
                    if (part.empty()) return;
                    std::vector<llama_token> part_tokens =
                        llama_tokenize_ex(ctx_llama, part, /*add_special=*/false, parse_special);
                    embd.insert(embd.end(), part_tokens.begin(), part_tokens.end());
                };

                append_part(preset_d.user_prefix, true);
                append_part(safe_user_text, false);
                append_part(preset_d.user_suffix, true);
                append_part(preset_d.bot_prefix, true);

                input_tokens_count = static_cast<int>(embd.size());
            }
            else {
                input_tokens_count = 0;
            }
        }

        { std::string dummy; allow_xtts_file(dummy, 1); }

        float temp_next = params.temp;
        // Regen: флаг g_regen_pending устанавливается в 23.13.
        // Первый токен будет взят из smpl_regen (temp = params.temp + 0.5f),
        // затем — из стандартного smpl. Флаг сбрасывается в 23.17.8.2
        // после первого токена. При greedy (smpl_regen == nullptr)
        // флаг сбрасываем здесь — иначе он «зависнет» навсегда.
        bool regen_first_token = g_regen_pending.load();
        if (regen_first_token && smpl_regen == nullptr) {
            g_regen_pending.store(false);
            regen_first_token = false;
        }
        if (regen_first_token && params.verbose) {
            fprintf(stderr, "[Regen] первый токен из smpl_regen (temp=%.2f)\n",
                    params.temp + 0.5f);
        }
        bool done = false;
        std::string text_to_speak;
        std::string full_response_text;
        int new_tokens = 0;
        std::string tts_smart_buffer;
        bool was_interrupted = false;
        int loop_detection_counter = 0;
        int tokens_since_last_loop = 0;
        int seqrep_counter = 0;
        int loop_check_counter = 0;
        bool bot_prefix_printed = false;
        int empty_iterations = 0;
        std::string last_output_buffer;
        std::string last_output_needle;
        std::string token_accumulator;

        g_interrupt_processed.store(false);

        llama_sampler_reset(smpl);
        llama_sampler_reset(smpl_high_temp);
        if (smpl_regen) llama_sampler_reset(smpl_regen);

        // ====================================================================
        // 23.17. ОСНОВНОЙ ЦИКЛ ГЕНЕРАЦИИ ТОКЕНОВ
        // ====================================================================
        while (true) {
            // --- 23.17.1. ЦЕНТРАЛИЗОВАННАЯ ОБРАБОТКА ПРЕРЫВАНИЯ ---
            // Единственная точка, где чистится TTS-очередь и семафор.
            // При barge-in сохраняем снимок для auto-continue — но ТОЛЬКО
            // если embd_inp.size() == n_past (синхронизировано). Иначе
            // auto-continue отключаем, чтобы не откатиться в KV.
            {
                InterruptReason reason = g_interrupt_reason.load();
                if (reason != InterruptReason::NONE) {
                    g_cancel_tts_requests.store(true);
                    clear_tts_queue();
                    {
                        std::string dummy;
                        allow_xtts_file(dummy, 0);
                    }

                    done = true;
                    was_interrupted = true;
                    text_to_speak.clear();
                    tts_smart_buffer.clear();

                    if (bot_prefix_printed) {
                        if (reason == InterruptReason::VAD_SPEECH) {
                            append_status_to_current_line("int", C_STOP);
                        } else {
                            append_status_to_current_line("stop", C_STOP);
                        }
                        bot_prefix_printed = false;
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_saved_generation_mutex);
                        if (reason != InterruptReason::MANUAL_STOP) {
                            const int embd_size = static_cast<int>(embd_inp.size());
                            if (embd_size == n_past) {
                                // Синхронизировано — auto-continue безопасен.
                                g_saved_n_past = n_past;
                                g_saved_embd_inp = embd_inp;
                                g_saved_timestamp = get_current_time_ms();
                                g_saved_was_interrupted = true;
                            } else {
                                // Рассинхронизация: embd_inp отстаёт от n_past
                                // (токены в embd ещё не влиты). Auto-continue
                                // отключаем, чтобы не откатиться назад в KV
                                // и не получить inconsistent positions.
                                g_saved_was_interrupted = false;
                                if (params.verbose) {
                                    fprintf(stderr,
                                        "[Barge-in] embd_inp (%d) != n_past (%d), "
                                        "auto-continue отключён\n",
                                        embd_size, n_past);
                                }
                            }
                        } else {
                            g_saved_was_interrupted = false;
                        }
                    }

                    g_bot_state.store(BotState::IDLE);
                    break;
                }
            }

            // --- 23.17.2. Проверка: ctx_llama не null ---
            {
                std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                if (!g_ctx_llama_atomic.load()) {
                    fprintf(stderr, "\n[ERROR] ctx_llama is null during generation\n");
                    done = true; was_interrupted = true;
                    break;
                }
            }

            if (g_reset_in_progress.load()) {
                done = true; was_interrupted = true; break;
            }

            // --- 23.17.3. Проверка горячей клавиши ---
            {
                std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex);
                if (!g_hotkey_pressed.empty() && g_hotkey_pressed != "Alt") {
                    g_interrupt_reason.store(InterruptReason::HOTKEY_STOP);
                    g_interrupt_processed.store(true);
                    g_hotkey_pressed = "";
                }
            }

            // --- 23.17.4. Проверка лимита токенов ---
            if (new_tokens > params.n_predict) {
                if (params.verbose) {
                    fprintf(stderr, "\n[Generation] Лимит токенов (%d)\n",
                            params.n_predict);
                }
                done = true;
                break;
            }

            // ================================================================
            // 23.17.5. ОБРАБОТКА ПЕРЕПОЛНЕНИЯ КОНТЕКСТА (K-shift / iSWA)
            // ================================================================
            // Если n_past + embd.size() > n_ctx:
            //   • iSWA-модель — hard truncate до n_keep.
            //   • Обычная — удаляем 1/8 хвоста (K_SHIFT_DIVISOR), сдвигаем
            //     KV, режем embd_inp. При can_shift=false — fallback до n_keep.
            if (embd.size() > 0) {
                if (n_past + static_cast<int>(embd.size()) > n_ctx) {
                    std::unique_lock<std::shared_mutex> wlock(g_llama_rw_mutex);
                    llama_context* active_ctx = g_ctx_llama_atomic.load();
                    if (!active_ctx) { done = true; break; }

                    llama_memory_t mem = llama_get_memory(active_ctx);

                    static const bool is_iswa_model =
                        (llama_model_n_swa(model_llama) > 0);

                    bool kv_updated = false;

                    if (is_iswa_model) {
                        if (params.verbose) {
                            fprintf(stderr,
                                "\n[iSWA] Hard truncate: n_past=%d, n_keep=%d, "
                                "ctx=%d, embd=%zu\n",
                                n_past, n_keep, n_ctx, embd.size());
                        }
                        llama_memory_seq_rm(mem, 0, n_keep, -1);
                        if (embd_inp.size() > static_cast<size_t>(n_keep)) {
                            embd_inp.resize(n_keep);
                        }
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        if (session_tokens.size() > embd_inp.size())
                            session_tokens.resize(embd_inp.size());

                        token_accumulator.clear();
                        last_output_buffer.clear();
                        last_output_needle.clear();
                        embd.clear();
                        llama_sampler_reset(smpl);
                        llama_sampler_reset(smpl_high_temp);
                        if (smpl_regen) llama_sampler_reset(smpl_regen);

                        for (int& p : past_prev_arr) {
                            if (p > n_keep) p = n_keep;
                        }
                        text_heard_prev.clear();

                        done = true;
                        was_interrupted = false;
                        if (bot_prefix_printed) {
                            append_status_to_current_line("ctx-full", C_STATUS_STOP);
                        }
                        bot_prefix_printed = false;
                        g_bot_state.store(BotState::IDLE);
                        break;
                    }

                    const int n_left_local = std::max(0, n_past - n_keep);
                    int n_discard_local = n_left_local / K_SHIFT_DIVISOR;
                    n_discard_local = std::clamp(n_discard_local, 0,
                                                 std::max(0, n_left_local - 1));

                    // Защита от разрезания хвоста.
                    if (n_keep + n_discard_local > n_past) {
                        n_discard_local = std::max(0, n_past - n_keep - 1);
                    }

                    if (!k_shift_enabled || !llama_memory_can_shift(mem)) {
                        if (params.verbose) {
                            fprintf(stderr, "\n[KShift] can_shift=false "
                                    "(k_shift_enabled=%d, runtime=%d), "
                                    "fallback to n_keep\n",
                                    (int)k_shift_enabled,
                                    (int)llama_memory_can_shift(mem));
                        }
                        embd_inp.resize(std::min(static_cast<size_t>(n_keep),
                                                 embd_inp.size()));
                        llama_memory_seq_rm(mem, 0, n_keep, -1);
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        if (session_tokens.size() > embd_inp.size())
                            session_tokens.resize(embd_inp.size());
                        token_accumulator.clear();
                        last_output_buffer.clear();
                        last_output_needle.clear();
                        for (int& p : past_prev_arr) {
                            if (p > n_keep) p = n_keep;
                        }
                        text_heard_prev.clear();
                        kv_updated = true;
                    }

                    if (!kv_updated && n_discard_local > 0 &&
                        n_keep + n_discard_local <= static_cast<int>(embd_inp.size()) &&
                        n_keep + n_discard_local <= n_past) {

                        bool rm_ok = llama_memory_seq_rm(mem, 0,
                                                         n_keep,
                                                         n_keep + n_discard_local);
                        if (rm_ok) {
                            llama_memory_seq_add(mem, 0,
                                                 n_keep + n_discard_local,
                                                 n_past, -n_discard_local);

                            embd_inp.erase(embd_inp.begin() + n_keep,
                                           embd_inp.begin() + n_keep + n_discard_local);

                            if (session_tokens.size() > embd_inp.size())
                                session_tokens.resize(embd_inp.size());

                            llama_sampler_reset(smpl);
                            llama_sampler_reset(smpl_high_temp);
                            if (smpl_regen) llama_sampler_reset(smpl_regen);

                            for (int& p : past_prev_arr) {
                                if (p > n_keep) {
                                    p -= n_discard_local;
                                    if (p < n_keep) p = n_keep;
                                }
                            }
                            text_heard_prev.clear();

                            kv_updated = true;
                            if (params.verbose) {
                                fprintf(stderr, "\n[KShift] Удал. %d ток, ост: %zu "
                                        "(session_tokens=%zu)\n",
                                        n_discard_local, embd_inp.size(),
                                        session_tokens.size());
                            }
                        }
                    }

                    if (!kv_updated) {
                        if (params.verbose) {
                            fprintf(stderr, "\n[KShift] Fallback: resize to n_keep=%d\n",
                                    n_keep);
                        }
                        embd_inp.resize(std::min(static_cast<size_t>(n_keep),
                                                 embd_inp.size()));
                        llama_memory_seq_rm(mem, 0, n_keep, -1);
                        for (int& p : past_prev_arr) {
                            if (p > n_keep) p = n_keep;
                        }
                        text_heard_prev.clear();
                        if (session_tokens.size() > embd_inp.size())
                            session_tokens.resize(embd_inp.size());
                    }

                    n_past = static_cast<int>(embd_inp.size());
                    n_session_consumed = n_past;
                    need_to_save_session = true;

                    const llama_vocab* vocab_local = llama_model_get_vocab(model_llama);
                    if (vocab_local) {
                        const llama_token bos_token = llama_vocab_bos(vocab_local);
                        if (!embd_inp.empty() && embd_inp[0] != bos_token) {
                            embd_inp.insert(embd_inp.begin(), bos_token);
                            n_past = static_cast<int>(embd_inp.size());
                            n_session_consumed = n_past;
                        }
                    }
                }
            }

            new_tokens++;

            // --- 23.17.6. Сохранение сессии ---
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = static_cast<int>(session_tokens.size());
            }

            // ================================================================
            // 23.17.7. ДЕКОДИРОВАНИЕ
            // ================================================================
            bool decode_failed = false;
            bool decode_skipped = false;
            {
                std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                llama_context* active_ctx = g_ctx_llama_atomic.load();
                if (!active_ctx) { done = true; break; }

                if (embd.empty()) {
                    decode_skipped = true;
                } else {
                    int total_to_decode = static_cast<int>(embd.size());
                    int decoded = 0;
                    bool chunk_failed = false;
                    while (decoded < total_to_decode) {
                        int chunk = std::min(params.batch_size,
                                             total_to_decode - decoded);
                        batch.n_tokens = chunk;
                        for (int i = 0; i < chunk; ++i) {
                            batch.token[i] = embd[decoded + i];
                            batch.pos[i] = n_past + decoded + i;
                            batch.n_seq_id[i] = 1;
                            batch.seq_id[i][0] = 0;
                            batch.logits[i] = (i == chunk - 1) ? 1 : 0;
                        }
                        if (llama_decode(active_ctx, batch)) {
                            fprintf(stderr, "%s : failed to decode chunk "
                                    "(decoded=%d, chunk=%d)\n",
                                    __func__, decoded, chunk);
                            chunk_failed = true;
                            break;
                        }
                        decoded += chunk;
                    }

                    if (chunk_failed) {
                        llama_memory_seq_rm(llama_get_memory(active_ctx), 0,
                                            n_past, -1);
                        embd.clear();
                        n_past = static_cast<int>(embd_inp.size());
                        n_session_consumed = n_past;
                        decode_failed = true;
                    }
                }
            }

            if (decode_failed) {
                empty_iterations++;
                if (empty_iterations > MAX_EMPTY_ITERATIONS) {
                    if (params.verbose) {
                        fprintf(stderr, "\n[Generation] Слишком много сбоев "
                                "decode (%d), выход\n", empty_iterations);
                    }
                    done = true;
                    break;
                }
                continue;
            }

            if (decode_skipped) {
                empty_iterations = 0;
            } else {
                empty_iterations = 0;
                embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                n_past = static_cast<int>(embd_inp.size());
                embd.clear();

                static const bool is_iswa_model_sync =
                    (llama_model_n_swa(model_llama) > 0);
                std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                llama_context* active_ctx = g_ctx_llama_atomic.load();
                if (active_ctx) {
                    llama_pos seq_max =
                        llama_memory_seq_pos_max(llama_get_memory(active_ctx), 0);
                    if (seq_max < 0) {
                        if (n_past < n_keep) n_past = n_keep;
                        if (n_session_consumed < n_past)
                            n_session_consumed = n_past;
                    } else if (!is_iswa_model_sync) {
                        int expected_n_past = static_cast<int>(seq_max) + 1;
                        if (n_past != expected_n_past) {
                            if (params.verbose) {
                                fprintf(stderr,
                                    "\n[Sync] n_past расхождение: %d -> %d "
                                    "(seq_max=%d)\n",
                                    n_past, expected_n_past, seq_max);
                            }
                            if (expected_n_past < n_past) {
                                // KV отстаёт от n_past: откат запрещён,
                                // иначе llama_decode откажется (Y = X + 1).
                                // Прерываем ход, KV не трогаем.
                                fprintf(stderr,
                                    "[Sync] ERROR: KV отстаёт от n_past. "
                                    "Откат запрещён.\n");
                                // Сообщаем 23.18, что ход прерван: иначе
                                // was_interrupted_final == false и TTS-очередь
                                // не будет очищена, text_to_speak не сброшен.
                                g_interrupt_reason.store(InterruptReason::HOTKEY_STOP);
                                g_interrupt_processed.store(true);
                                // Синхронизируем сессию с embd_inp: KV не принял
                                // эти токены, поэтому session_tokens не должен
                                // «убегать» вперёд. Следующая загрузка сессии
                                // восстановит ровно то, что есть в KV.
                                n_session_consumed = static_cast<int>(embd_inp.size());
                                if (session_tokens.size() > embd_inp.size()) {
                                    session_tokens.resize(embd_inp.size());
                                }
                                done = true;
                                was_interrupted = true;
                                break;
                            }
                            if (embd_inp.size() >
                                static_cast<size_t>(expected_n_past)) {
                                embd_inp.resize(expected_n_past);
                            }
                            n_past = expected_n_past;
                            n_session_consumed = n_past;
                        }
                    }
                }
            }

            if (done) break;

            std::string out_token_str = "";
            char out_token_symbol = 0;
            if (llama_start_generation_time == 0.0)
                llama_start_generation_time = get_current_time_ms();

            // ================================================================
            // 23.17.8. ОБРАБОТКА ВЫХОДНОГО ТОКЕНА
            // ================================================================
            {
                std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
                llama_context* active_ctx = g_ctx_llama_atomic.load();
                if (!active_ctx) { done = true; break; }

                // --- 23.17.8.1. Сохранение сессии ---
                if (!path_session.empty() && need_to_save_session) {
                    need_to_save_session = false;
                    llama_state_save_file(active_ctx, path_session.c_str(),
                        session_tokens.data(), session_tokens.size());
                }

                // --- 23.17.8.2. Выборка токена ---
                // Приоритет:
                //   1) regen_first_token — smpl_regen (temp+0.5f), один раз.
                //   2) temp != temp_next — smpl_high_temp (temp=2.0), один раз.
                //   3) иначе — стандартный smpl.
                llama_token id = 0;
                if (regen_first_token && smpl_regen != nullptr) {
                    regen_first_token = false;
                    g_regen_pending.store(false);
                    llama_sampler_reset(smpl_regen);
                    id = llama_sampler_sample(smpl_regen, active_ctx, -1);
                } else if (temp != temp_next) {
                    llama_sampler_reset(smpl_high_temp);
                    id = llama_sampler_sample(smpl_high_temp, active_ctx, -1);
                    temp = temp_next = params.temp;
                } else {
                    id = llama_sampler_sample(smpl, active_ctx, -1);
                }

                // --- 23.17.8.3. Проверка стоп-строк ---
                {
                    std::string piece = llama_token_to_piece(active_ctx, id);
                    if (token_accumulator.size() > 256) {
                        token_accumulator = token_accumulator.substr(
                            token_accumulator.size() - 256);
                    }
                    token_accumulator += piece;

                    bool stop_found = false;

                    if (llama_vocab_is_eog(vocab_llama, id)) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] EOG токен %d '%s'\n",
                                    id, piece.c_str());
                        }
                    }

                    if (!stop_found && llama_vocab_is_control(vocab_llama, id)) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] Control токен %d '%s'\n",
                                    id, piece.c_str());
                        }
                    }

                    if (!stop_found && piece.find("<|eo_id|>") != std::string::npos) {
                        stop_found = true;
                        if (params.verbose) {
                            fprintf(stderr, "[Generation] Неправильный EOG: '%s'\n",
                                    piece.c_str());
                        }
                    }

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

                // --- 23.17.8.4. Обработка выходного токена ---
                if (id != llama_vocab_eos(vocab_llama)) {
                    embd.push_back(id);
                    out_token_str = llama_token_to_piece(active_ctx, id);

                    // Вырезаем плейсхолдеры {0}..{5}, если модель их сгенерировала.
                    // Подстановки больше нет: дата и имена уже в промпте.
                    for (const char * ph : {"{0}", "{1}", "{2}", "{3}", "{4}", "{5}"}) {
                        size_t p = out_token_str.find(ph);
                        if (p != std::string::npos) {
                            out_token_str.erase(p, std::strlen(ph));
                        }
                    }

                    // --- 23.17.8.5. Мультиперсонажность ---
                    std::string visible_token = out_token_str;

                    if (params.multi_chars && !visible_token.empty()) {
                        std::string tail = full_response_text;
                        if (tail.size() > 200) {
                            tail = tail.substr(tail.size() - 200);
                        }
                        tail += visible_token;

                        static const std::regex re_name(
                            R"(\n([^\n:]{2,30}):)",
                            std::regex::optimize);
                        std::smatch m;
                        if (std::regex_search(tail, m, re_name) && m.size() >= 2) {
                            std::string candidate = m[1].str();
                            trim(candidate);
                            if (candidate.size() >= 2 && candidate.size() <= 30 &&
                                candidate != params.person &&
                                candidate != current_bot_name) {
                                if (current_voice != candidate) {
                                    if (g_verbose_mode.load()) {
                                        fprintf(stderr, "[MultiChars] Голос "
                                                "переключён: '%s' → '%s'\n",
                                                current_voice.c_str(),
                                                candidate.c_str());
                                    }
                                    current_voice = candidate;
                                }
                            }
                        }
                    }

                    // --- 23.17.8.6. ЕДИНЫЙ ВЫВОД: консоль + TTS ---
                    if (!visible_token.empty()) {
                        if (!bot_prefix_printed) {
                            print_bot_prefix(current_bot_name);
                            bot_prefix_printed = true;
                        }
                        printf("%s", visible_token.c_str());
                        fflush(stdout);
                        tokens_in_reply++;
                    }

                    text_to_speak += visible_token;
                    full_response_text += visible_token;

                    // --- 23.17.8.7. Проверка антипромптов ---
                    {
                        bool antiprompt_hit = false;
                        std::string antiprompt_matched = "";
                        std::string last_output_buffer_ap = text_to_speak;
                        if (last_output_buffer_ap.size() > 256) {
                            last_output_buffer_ap = last_output_buffer_ap.substr(
                                last_output_buffer_ap.size() - 256);
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
                            if (params.min_tokens > 0 &&
                                tokens_in_reply < params.min_tokens) {
                                temp_next = params.temp + 0.3f;
                            } else {
                                size_t pos_in_speak =
                                    text_to_speak.find(antiprompt_matched);
                                if (pos_in_speak != std::string::npos) {
                                    text_to_speak =
                                        text_to_speak.substr(0, pos_in_speak);
                                    trim_spaces_only(text_to_speak);
                                }
                                done = true;
                                break;
                            }
                        }
                    }

                    // --- 23.17.8.8. Отправка в TTS по предложениям ---
                    tts_smart_buffer += text_to_speak;
                    text_to_speak.clear();
                    bool should_send = false;
                    size_t buf_len = tts_smart_buffer.size();
                    if (buf_len >= 1) {
                        char last_char = tts_smart_buffer[buf_len - 1];
                        bool punctuation_ok = false;
                        if (last_char == '!' || last_char == '?') {
                            punctuation_ok = true;
                        }
                        else if (last_char == '.' || last_char == '\n') {
                            if (last_char == '\n') {
                                punctuation_ok = true;
                            } else if (buf_len >= 2) {
                                char prev = tts_smart_buffer[buf_len - 2];
                                if (!std::isdigit(static_cast<unsigned char>(prev))) {
                                    punctuation_ok = true;
                                }
                            } else {
                                punctuation_ok = true;
                            }
                        }
                        if (punctuation_ok) should_send = true;

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
                            std::isspace(static_cast<unsigned char>(c2))) {
                            bool prev_is_digit = false;
                            if (buf_len >= 3) {
                                prev_is_digit = std::isdigit(
                                    static_cast<unsigned char>(
                                        tts_smart_buffer[buf_len - 3]));
                            }
                            if (!(c1 == '.' && prev_is_digit)) {
                                should_send = true;
                            }
                        }
                    }

                    if (!should_send && params.split_after > 0 &&
                        static_cast<int>(tts_smart_buffer.size()) >= params.split_after &&
                        g_interrupt_reason.load() == InterruptReason::NONE) {
                        int split_open_p = 0, split_open_b = 0, split_open_c = 0;
                        for (char c : tts_smart_buffer) {
                            if (c == '(') split_open_p++;
                            else if (c == ')') split_open_p = std::max(0, split_open_p - 1);
                            else if (c == '[') split_open_b++;
                            else if (c == ']') split_open_b = std::max(0, split_open_b - 1);
                            else if (c == '{') split_open_c++;
                            else if (c == '}') split_open_c = std::max(0, split_open_c - 1);
                        }
                        if (split_open_p == 0 && split_open_b == 0 && split_open_c == 0) {
                            should_send = true;
                        }
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
                            std::string to_send_part =
                                tts_smart_buffer.substr(0, cut_pos + 1);
                            tts_smart_buffer = tts_smart_buffer.substr(cut_pos + 1);
                            to_send_part = strip_dialog_prefixes(to_send_part,
                                params.person, current_bot_name, chat_symb);
                            if (!preset_d.stop_sequence.empty())
                                to_send_part = string_replace_all(to_send_part,
                                    preset_d.stop_sequence, "");
                            if (!preset_d.bot_suffix.empty())
                                to_send_part = string_replace_all(to_send_part,
                                    preset_d.bot_suffix, "");

                            bool was_space_terminated = false;
                            if (!to_send_part.empty()) {
                                char lastc = to_send_part.back();
                                was_space_terminated = (lastc == ' ' || lastc == '\t' ||
                                                        lastc == '\n' || lastc == '\r');
                            }
                            trim_spaces_only(to_send_part);
                            if (!to_send_part.empty() && !was_space_terminated) {
                                char lastc = to_send_part.back();
                                if (lastc != '.' && lastc != '!' &&
                                    lastc != '?' && lastc != ':' &&
                                    lastc != ';' && lastc != ',') {
                                    to_send_part += ' ';
                                }
                            }

                            int open_p = 0, open_b = 0, open_c = 0;
                            for (char c : to_send_part) {
                                if (c == '(') open_p++; else if (c == ')') open_p--;
                                else if (c == '[') open_b++; else if (c == ']') open_b--;
                                else if (c == '{') open_c++; else if (c == '}') open_c--;
                            }
                            if (open_p > 0 || open_b > 0 || open_c > 0) {
                                should_send = false;
                            }

                            if (!to_send_part.empty() && should_send) {
                                wait_for_tts_idle();
                                enqueue_tts(make_tts_req(to_send_part));
                                update_console_title(current_bot_name, "Speaking...");
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

                        bool was_space_terminated = false;
                        if (!to_send.empty()) {
                            char lastc = to_send.back();
                            was_space_terminated = (lastc == ' ' || lastc == '\t' ||
                                                    lastc == '\n' || lastc == '\r');
                        }
                        trim_spaces_only(to_send);
                        if (!to_send.empty() && !was_space_terminated) {
                            char lastc = to_send.back();
                            if (lastc != '.' && lastc != '!' &&
                                lastc != '?' && lastc != ':' &&
                                lastc != ';' && lastc != ',') {
                                to_send += ' ';
                            }
                        }

                        int open_p = 0, open_b = 0, open_c = 0;
                        for (char c : to_send) {
                            if (c == '(') open_p++; else if (c == ')') open_p--;
                            else if (c == '[') open_b++; else if (c == ']') open_b--;
                            else if (c == '{') open_c++; else if (c == '}') open_c--;
                        }
                        if (open_p > 0 || open_b > 0 || open_c > 0) {
                            should_send = false;
                        }

                        if (!to_send.empty() && should_send) {
                            wait_for_tts_idle();
                            enqueue_tts(make_tts_req(to_send));
                            update_console_title(current_bot_name, "Speaking...");
                        }
                    }

                    // --- 23.17.8.9. Seqrep — защита от повторов ---
                    if (params.seqrep && !out_token_str.empty()) {
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
                                last_output_buffer.find(last_output_needle)
                                    != std::string::npos) {
                                if (params.verbose)
                                    fprintf(stderr, "[Seqrep] Повтор '%s'\n",
                                            last_output_needle.c_str());
                                seqrep_counter++;
                                if (seqrep_counter >= 3) {
                                    if (params.verbose)
                                        fprintf(stderr,
                                            "[Seqrep] Стоп после %d\n",
                                            seqrep_counter);
                                    done = true;
                                    break;
                                }
                                int symbols_to_delete = utf8_length(last_output_needle);
                                std::vector<llama_token> tokens_to_del =
                                    llama_tokenize_ex(active_ctx,
                                        last_output_needle, false, false);
                                int rollback_num = static_cast<int>(tokens_to_del.size());
                                if (rollback_num > 0 &&
                                    rollback_num <= static_cast<int>(embd_inp.size())) {
                                    embd_inp.erase(embd_inp.end() - rollback_num,
                                                   embd_inp.end());
                                    n_past = static_cast<int>(embd_inp.size());
                                    n_session_consumed = n_past;
                                    llama_memory_seq_rm(
                                        llama_get_memory(active_ctx), 0,
                                        static_cast<int>(embd_inp.size()), -1);
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

                    // --- 23.17.8.10. Loop detection ---
                    loop_check_counter++;
                    bool run_loop_check =
                        (loop_check_counter % LOOP_CHECK_INTERVAL == 0);

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

                        const size_t max_len = std::min<size_t>(12, N / 3);
                        for (size_t len = 6; len <= max_len; ++len) {
                            for (size_t start = 0;
                                 start + len * 3 <= N && start <= N / 2; ++start) {
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
                                fprintf(stderr,
                                    "[Generation] Повтор '%s' (%zu)\n",
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

                    // --- 23.17.8.11. Паттерны циклов ---
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

                    // --- 23.17.8.12. Повтор одного символа ---
                    if (out_token_str.length() > 10) {
                        bool all_same = true;
                        for (size_t i = 1; i < out_token_str.length(); ++i) {
                            if (out_token_str[i] != out_token_str[0]) {
                                all_same = false; break;
                            }
                        }
                        if (all_same) {
                            if (params.verbose)
                                fprintf(stderr,
                                    "[Generation] Повтор символов '%s'\n",
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
        // 23.18. ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ
        // ====================================================================
        // Назначение: финальная отправка остатка в TTS, обработка прерывания,
        //   сохранение сессии, статистика, печать приглашения.
        // Инварианты:
        //   • При was_interrupted_final — text_to_speak = "", audio.clear().
        //   • При !was_interrupted_final — enqueue_tts остатка, append_status("tts").
        //   • g_interrupt_reason/processed сбрасываются.
        //   • g_bot_state = IDLE.
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
                if (g_interrupt_reason.load() == InterruptReason::NONE &&
                    !g_tts_muted.load()) {
                    wait_for_tts_idle();
                    enqueue_tts(make_tts_req(text_to_speak));
                }
            }

            if (bot_prefix_printed) {
                append_status_to_current_line("tts", C_TTS);
            } else {
                std::lock_guard<std::recursive_mutex> lock(g_console_mutex);
                printf("\n");
                fflush(stdout);
            }
            bot_prefix_printed = false;
        } else {
            bot_prefix_printed = false;
        }

        if (was_interrupted_final) {
            g_cancel_tts_requests.store(false);
            if (!g_tts_muted.load()) {
                std::string dummy2;
                allow_xtts_file(dummy2, 1);
            }
        }
        if (!was_interrupted_final) audio.clear();

        // --- 23.19. Сохранение сессии после хода ---
        if (!path_session.empty() && !was_interrupted_final) {
            std::shared_lock<std::shared_mutex> rlock(g_llama_rw_mutex);
            llama_context* active_ctx = g_ctx_llama_atomic.load();
            if (active_ctx) {
                if (session_tokens.size() > embd_inp.size()) {
                    session_tokens.resize(embd_inp.size());
                }
                llama_state_save_file(active_ctx, path_session.c_str(),
                    session_tokens.data(), session_tokens.size());
            }
        }

        // --- 23.20. Статистика (verbose) ---
        llama_end_time = get_current_time_ms();
        if (params.verbose) {
            llama_time_input = llama_start_generation_time - llama_start_time;
            llama_time_output = llama_end_time - llama_start_generation_time;
            llama_time_total = llama_end_time - llama_start_time;
            printf("\n[Context: %d/%d. Tokens: %d in + %d out. "
                   "Input %.3f s + output %.3f s = total: %.3f s]\n",
                n_past, n_ctx, input_tokens_count, new_tokens,
                llama_time_input, llama_time_output, llama_time_total);
            double input_speed = (llama_time_input > 0.001) ?
                static_cast<double>(input_tokens_count) / llama_time_input : 0.0;
            double output_speed = (llama_time_output > 0.001) ?
                static_cast<double>(new_tokens) / llama_time_output : 0.0;
            printf("[Speed: input %.2f t/s + output %.2f t/s]\n",
                input_speed, output_speed);
        }

        g_interrupt_reason.store(InterruptReason::NONE);
        g_interrupt_processed.store(false);
        g_bot_state.store(BotState::IDLE);
        llama_start_generation_time = 0.0;
        { std::lock_guard<std::mutex> lock(g_hotkey_pressed_mutex); g_hotkey_pressed = ""; }

        update_console_title(current_bot_name, "Listening");

        // --- 23.21. Печать нового приглашения пользователя ---
        if (!was_interrupted_final &&
            g_bot_state.load() == BotState::IDLE &&
            !g_shutting_down.load()) {
            print_user_prompt(params.person);
        } else if (was_interrupted_final) {
            print_user_prompt(params.person);
        }

        fflush(stdout);
    }
    // ========================================================================
    // КОНЕЦ ОСНОВНОГО ЦИКЛА while (is_running)
    // ========================================================================

    // ========================================================================
    // 24. ЗАВЕРШЕНИЕ РАБОТЫ
    // ========================================================================
    // Назначение: корректно остановить все фоновые потоки, освободить
    //   ресурсы, напечатать статистику.
    //
    // Порядок:
    //   1) g_shutting_down = true — все циклы увидят флаг.
    //   2) audio_thread.join() — дождаться завершения Whisper.
    //   3) tts_worker.join() — дождаться завершения HTTP.
    //   4) audio.pause() / clear().
    //   5) input_thread.join() — readline не прерывается, но флаг
    //      keyboard_input_running = false.
    //   6) shortcut_thread.join().
    //   7) whisper_print_timings / llama_perf_context_print.
    //   8) Освобождение: ctx_llama, smpl, smpl_high_temp, smpl_regen,
    //      batch, model_llama, backend.
    //
    // Инварианты:
    //   • Порядок остановки важен: audio_thread первым — иначе TTS-воркер
    //     может получить запрос после clear_tts_queue.
    //   • smpl_regen может быть nullptr (при temp <= 0) — проверять.
    //   • g_ctx_llama_atomic.store(nullptr) — до llama_free, чтобы
    //     другие потоки не получили dangling pointer.
    if (params.verbose) printf("Cleaning up TTS threads...\n");
    g_shutting_down.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // --- 24.1. Остановка audio-потока ---
    if (audio_thread.joinable()) {
        for (int i = 0; i < 60 && g_audio_thread_running.load(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (g_audio_thread_running.load() && params.verbose) {
            fprintf(stderr, "[AudioInput] Поток ещё работает, ждём transcribe...\n");
        }
        audio_thread.join();
        if (params.verbose) printf("[AudioInput] Thread stopped\n");
    }

    // --- 24.2. Остановка TTS-воркера ---
    g_tts_worker_running.store(false);
    g_tts_queue_cv.notify_all();
    if (tts_worker.joinable()) tts_worker.join();

    // --- 24.3. Остановка аудио и очистка очереди ---
    audio.pause();
    audio.clear();

    // --- 24.4. Остановка клавиатурного потока ---
    keyboard_input_running.store(false);
    if (input_thread.joinable()) input_thread.join();

    // --- 24.5. Остановка потока горячих клавиш ---
    g_shortcut_thread_running.store(false);
    if (shortcut_thread.joinable()) shortcut_thread.join();

    audio.pause();

    // --- 24.6. Статистика Whisper и LLaMA ---
    whisper_print_timings(ctx_wsp);
    if (ctx_llama) llama_perf_context_print(ctx_llama);

    // --- 24.7. Освобождение ресурсов ---
    // Сначала обнуляем атомарный указатель — другие потоки уже остановлены,
    // но это гарантия от dangling pointer.
    g_ctx_llama_atomic.store(nullptr);

    whisper_free(ctx_wsp);

    if (smpl) {
        llama_perf_sampler_print(smpl);
        llama_sampler_free(smpl);
    }
    if (smpl_high_temp) llama_sampler_free(smpl_high_temp);
    if (smpl_regen) llama_sampler_free(smpl_regen);

    llama_batch_free(batch);
    llama_free(ctx_llama);
    llama_model_free(model_llama);
    llama_backend_free();

    if (params.verbose) printf("Goodbye!\n");
    return 0;
}


// ============================================================================
// 25. ТОЧКИ ВХОДА
// ============================================================================
// Назначение:
//   Windows: wmain, принимает UTF-16LE argv и конвертирует в UTF-8
//            (кириллица в путях к моделям).
//   POSIX:   обычный main.
//
// Инварианты:
//   • curl_global_init обязателен до первого использования libcurl.
//   • console::init(true, true) включает UTF-8 ввод/вывод и «продвинутый»
//     режим отображения.
//   • atexit() гарантирует очистку консоли и curl.
//   • TALK_LLAMA_VERBOSE=1 (или true) — включает verbose без аргументов.
// Используется: ОС.

#if _WIN32
int wmain(int argc, const wchar_t** argv_UTF16LE) {
    console::init(true, true);
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl" << std::endl;
        console::cleanup();
        return 1;
    }
    atexit([] {
        console::cleanup();
        curl_global_cleanup();
    });

    std::vector<std::string> buffer(argc);
    std::vector<char*> argv_UTF8(argc);
    for (int i = 0; i < argc; ++i) {
        buffer[i] = console::UTF16toUTF8(argv_UTF16LE[i]);
        argv_UTF8[i] = &buffer[i][0];
    }
    return run(argc, argv_UTF8.data());
}
#else
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
    atexit([] {
        console::cleanup();
        curl_global_cleanup();
    });
    return run(argc, argv_UTF8);
}
#endif