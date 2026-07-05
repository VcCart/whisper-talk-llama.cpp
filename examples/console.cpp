#include "console.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <climits>
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <termios.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

namespace console {

    //
    // Console state
    //

    static bool      advanced_display = false;
    static bool      simple_io        = true;
    static display_t current_display  = reset;

    static FILE*     out              = stdout;

#if defined (_WIN32)
    static void*     hConsole;
#else
    static FILE*     tty              = nullptr;
    static termios   initial_state;
#endif

#if _WIN32
// Convert UTF-8 to UTF-16
// Windows only
    std::wstring UTF8toUTF16(const std::string& utf8Str) {
        if (utf8Str.empty()) return {std::wstring()};

        int requiredSize = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, NULL, 0);
        if (requiredSize == 0) {
            // Handle error here
            return {std::wstring()};
        }

        std::wstring utf16Str(requiredSize, 0);
        if (MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, &utf16Str[0], requiredSize) == 0) {
            // Handle error here
            return {std::wstring()};
        }

        // Remove the additional null byte from the end
        utf16Str.resize(requiredSize - 1);
        return utf16Str;
    }
#endif

#if _WIN32
// Convert UTF-16 to UTF-8
// Windows only
    std::string UTF16toUTF8(const std::wstring & utf16Str) {
        if (utf16Str.empty()) return {std::string()};

        int requiredSize = WideCharToMultiByte(CP_UTF8, 0, utf16Str.c_str(), -1, NULL, 0, NULL, NULL);
        if (requiredSize == 0) {
            // Handle error here
            return {std::string()};
        }

        std::string utf8Str(requiredSize, 0);
        if (WideCharToMultiByte(CP_UTF8, 0, utf16Str.c_str(), -1, &utf8Str[0], requiredSize, NULL, NULL) == 0) {
            // Handle error here
            return {std::string()};
        }

        // Remove the additional null byte from the end
        utf8Str.resize(requiredSize - 1);
        return utf8Str;
    }
#endif

    //
    // Init and cleanup
    //

    void init(bool use_simple_io, bool use_advanced_display) {
        advanced_display = use_advanced_display;
        simple_io = use_simple_io;
#if defined(_WIN32)
        // Windows-specific console initialization
        DWORD dwMode = 0;
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
            if (hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(hConsole, &dwMode))) {
                hConsole = nullptr;
                simple_io = true;
            }
        }
        if (hConsole) {
            // Check conditions combined to reduce nesting
            if (advanced_display && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
                !SetConsoleMode(hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                advanced_display = false;
            }
            // Set console output codepage to UTF8
            SetConsoleOutputCP(CP_UTF8);
        }
        HANDLE hConIn = GetStdHandle(STD_INPUT_HANDLE);
        if (hConIn != INVALID_HANDLE_VALUE && GetConsoleMode(hConIn, &dwMode)) {
            // Set console input codepage to UTF16
            _setmode(_fileno(stdin), _O_WTEXT);

            // Set ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
            if (simple_io) {
                dwMode |= ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT;
            } else {
                dwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
            }
            if (!SetConsoleMode(hConIn, dwMode)) {
                simple_io = true;
            }
        }
#else
        // POSIX-specific console initialization
        if (!simple_io) {
            struct termios new_termios;
            tcgetattr(STDIN_FILENO, &initial_state);
            new_termios = initial_state;
            new_termios.c_lflag &= ~(ICANON | ECHO);
            new_termios.c_cc[VMIN] = 1;
            new_termios.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

            tty = fopen("/dev/tty", "w+");
            if (tty != nullptr) {
                out = tty;
            }
        }

        setlocale(LC_ALL, "");
#endif
    }

    void cleanup() {
        // Reset console display
        set_display(reset);

#if !defined(_WIN32)
        // Restore settings on POSIX systems
        if (!simple_io) {
            if (tty != nullptr) {
                out = stdout;
                fclose(tty);
                tty = nullptr;
            }
            tcsetattr(STDIN_FILENO, TCSANOW, &initial_state);
        }
#endif
    }

    //
    // Display and IO
    //

    // Keep track of current display and only emit ANSI code if it changes
    void set_display(display_t display) {
        if (advanced_display && current_display != display) {
            fflush(stdout);
            switch(display) {
                case reset:
                    fprintf(out, ANSI_COLOR_RESET);
                    break;
                case prompt:
                    fprintf(out, ANSI_COLOR_YELLOW);
                    break;
                case user_input:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                    break;
                case error:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_RED);
            }
            current_display = display;
            fflush(out);
        }
    }

    static char32_t getchar32() {
#if defined(_WIN32)
        HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
        wchar_t high_surrogate = 0;

        while (true) {
            INPUT_RECORD record;
            DWORD count;
            if (!ReadConsoleInputW(hConsole, &record, 1, &count) || count == 0) {
                return WEOF;
            }

            if (record.EventType == KEY_EVENT && record.Event.KeyEvent.bKeyDown) {
                wchar_t wc = record.Event.KeyEvent.uChar.UnicodeChar;
                if (wc == 0) {
                    const DWORD ctrl_mask = LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED;
                    const bool ctrl_pressed = (record.Event.KeyEvent.dwControlKeyState & ctrl_mask) != 0;
                    switch (record.Event.KeyEvent.wVirtualKeyCode) {
                        case VK_LEFT:   return ctrl_pressed ? 0xE006 : 0xE000;
                        case VK_RIGHT:  return ctrl_pressed ? 0xE007 : 0xE001;
                        case VK_UP:     return 0xE002;
                        case VK_DOWN:   return 0xE003;
                        case VK_HOME:   return 0xE004;
                        case VK_END:    return 0xE005;
                        case VK_DELETE: return 0xE008;
                        default:        continue;
                    }
                }

                if ((wc >= 0xD800) && (wc <= 0xDBFF)) {
                    high_surrogate = wc;
                    continue;
                }
                if ((wc >= 0xDC00) && (wc <= 0xDFFF)) {
                    if (high_surrogate != 0) {
                        return ((high_surrogate - 0xD800) << 10) + (wc - 0xDC00) + 0x10000;
                    }
                }

                high_surrogate = 0;
                return static_cast<char32_t>(wc);
            }
        }
#else
        wchar_t wc = getwchar();
        if (static_cast<wint_t>(wc) == WEOF) {
            return WEOF;
        }

#if WCHAR_MAX == 0xFFFF
        if ((wc >= 0xD800) && (wc <= 0xDBFF)) {
            wchar_t low_surrogate = getwchar();
            if ((low_surrogate >= 0xDC00) && (low_surrogate <= 0xDFFF)) {
                return (static_cast<char32_t>(wc & 0x03FF) << 10) + (low_surrogate & 0x03FF) + 0x10000;
            }
        }
        if ((wc >= 0xD800) && (wc <= 0xDFFF)) {
            return 0xFFFD;
        }
#endif

        return static_cast<char32_t>(wc);
#endif
    }

    static void pop_cursor() {
#if defined(_WIN32)
        if (hConsole != NULL) {
            CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
            GetConsoleScreenBufferInfo(hConsole, &bufferInfo);

            COORD newCursorPosition = bufferInfo.dwCursorPosition;
            if (newCursorPosition.X == 0) {
                newCursorPosition.X = bufferInfo.dwSize.X - 1;
                newCursorPosition.Y -= 1;
            } else {
                newCursorPosition.X -= 1;
            }

            SetConsoleCursorPosition(hConsole, newCursorPosition);
            return;
        }
#endif
        putc('\b', out);
    }

    static int estimateWidth(char32_t codepoint) {
#if defined(_WIN32)
        (void)codepoint;
        return 1;
#else
        return wcwidth(codepoint);
#endif
    }

    static int put_codepoint(const char* utf8_codepoint, size_t length, int expectedWidth) {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
        if (!GetConsoleScreenBufferInfo(hConsole, &bufferInfo)) {
            // go with the default
            return expectedWidth;
        }
        COORD initialPosition = bufferInfo.dwCursorPosition;
        DWORD nNumberOfChars = length;
        WriteConsole(hConsole, utf8_codepoint, nNumberOfChars, &nNumberOfChars, NULL);

        CONSOLE_SCREEN_BUFFER_INFO newBufferInfo;
        GetConsoleScreenBufferInfo(hConsole, &newBufferInfo);

        // Figure out our real position if we're in the last column
        if (utf8_codepoint[0] != 0x09 && initialPosition.X == newBufferInfo.dwSize.X - 1) {
            DWORD nNumberOfChars;
            WriteConsole(hConsole, &" \b", 2, &nNumberOfChars, NULL);
            GetConsoleScreenBufferInfo(hConsole, &newBufferInfo);
        }

        int width = newBufferInfo.dwCursorPosition.X - initialPosition.X;
        if (width < 0) {
            width += newBufferInfo.dwSize.X;
        }
        return width;
#else
        // We can trust expected Width if we've got one
        if (expectedWidth >= 0 || tty == nullptr) {
            fwrite(utf8_codepoint, length, 1, out);
            return expectedWidth;
        }

        fputs("\033[6n", tty);
        int x1, y1, x2, y2;
        int results = 0;
        results = fscanf(tty, "\033[%d;%dR", &y1, &x1);

        fwrite(utf8_codepoint, length, 1, tty);

        fputs("\033[6n", tty);
        results += fscanf(tty, "\033[%d;%dR", &y2, &x2);

        if (results != 4) {
            return expectedWidth;
        }

        int width = x2 - x1;
        if (width < 0) {
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            width += w.ws_col;
        }
        return width;
#endif
    }

    static void replace_last(char ch) {
#if defined(_WIN32)
        pop_cursor();
        put_codepoint(&ch, 1, 1);
#else
        fprintf(out, "\b%c", ch);
#endif
    }

    static void append_utf8(char32_t ch, std::string & out) {
        if (ch <= 0x7F) {
            out.push_back(static_cast<unsigned char>(ch));
        } else if (ch <= 0x7FF) {
            out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0xFFFF) {
            out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0x10FFFF) {
            out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else {
            // Invalid Unicode code point
        }
    }

    static void pop_back_utf8_char(std::string & line) {
        if (line.empty()) {
            return;
        }

        size_t pos = line.length() - 1;

        for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
            if ((line[pos] & 0xC0) != 0x80) {
                break;
            }
        }
        line.erase(pos);
    }

    // ============================================================
    // UTF-8 ��������������� ������� (���������)
    // ============================================================

    static char32_t decode_utf8(const std::string & input, size_t pos, size_t & advance) {
        unsigned char c = static_cast<unsigned char>(input[pos]);
        if ((c & 0x80u) == 0u) {
            advance = 1;
            return c;
        }
        if ((c & 0xE0u) == 0xC0u && pos + 1 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            if ((c1 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 2;
            return ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(input[pos + 1]) & 0x3Fu);
        }
        if ((c & 0xF0u) == 0xE0u && pos + 2 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            unsigned char c2 = static_cast<unsigned char>(input[pos + 2]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 3;
            return ((c & 0x0Fu) << 12) |
                   ((static_cast<unsigned char>(input[pos + 1]) & 0x3Fu) << 6) |
                   (static_cast<unsigned char>(input[pos + 2]) & 0x3Fu);
        }
        if ((c & 0xF8u) == 0xF0u && pos + 3 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            unsigned char c2 = static_cast<unsigned char>(input[pos + 2]);
            unsigned char c3 = static_cast<unsigned char>(input[pos + 3]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u || (c3 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 4;
            return ((c & 0x07u) << 18) |
                   ((static_cast<unsigned char>(input[pos + 1]) & 0x3Fu) << 12) |
                   ((static_cast<unsigned char>(input[pos + 2]) & 0x3Fu) << 6) |
                   (static_cast<unsigned char>(input[pos + 3]) & 0x3Fu);
        }

        advance = 1;
        return 0xFFFD;
    }

    static size_t next_utf8_char_pos(const std::string & line, size_t pos) {
        if (pos >= line.length()) return line.length();
        pos++;
        while (pos < line.length() && (line[pos] & 0xC0) == 0x80) {
            pos++;
        }
        return pos;
    }

    static size_t prev_utf8_char_pos(const std::string & line, size_t pos) {
        if (pos == 0) return 0;
        pos--;
        while (pos > 0 && (line[pos] & 0xC0) == 0x80) {
            pos--;
        }
        return pos;
    }

    // ============================================================
    // ����������� �������
    // ============================================================

    static void move_cursor(int delta) {
        if (delta == 0) return;
#if defined(_WIN32)
        if (hConsole != NULL) {
            CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
            GetConsoleScreenBufferInfo(hConsole, &bufferInfo);
            COORD newCursorPosition = bufferInfo.dwCursorPosition;
            int width = bufferInfo.dwSize.X;
            int newX = newCursorPosition.X + delta;
            int newY = newCursorPosition.Y;

            while (newX >= width) {
                newX -= width;
                newY++;
            }
            while (newX < 0) {
                newX += width;
                newY--;
            }

            newCursorPosition.X = newX;
            newCursorPosition.Y = newY;
            SetConsoleCursorPosition(hConsole, newCursorPosition);
        }
#else
        if (delta < 0) {
            for (int i = 0; i < -delta; i++) fprintf(out, "\b");
        } else {
            for (int i = 0; i < delta; i++) fprintf(out, "\033[C");
        }
#endif
    }

    // ============================================================
    // �������
    // ============================================================

    struct history_t {
        std::vector<std::string> entries;
        size_t viewing_idx = SIZE_MAX;
        std::string backup_line;
        
        void add(std::string_view line) {
            if (line.empty()) return;
            if (entries.empty() || entries.back() != line) {
                entries.emplace_back(line);
            }
            end_viewing();
        }
        
        bool prev(std::string & cur_line) {
            if (entries.empty() || viewing_idx == SIZE_MAX) return false;
            if (viewing_idx > 0) viewing_idx--;
            cur_line = entries[viewing_idx];
            return true;
        }
        
        bool next(std::string & cur_line) {
            if (entries.empty() || viewing_idx == SIZE_MAX) return false;
            viewing_idx++;
            if (viewing_idx >= entries.size()) {
                cur_line = backup_line;
                end_viewing();
            } else {
                cur_line = entries[viewing_idx];
            }
            return true;
        }
        
        void begin_viewing(const std::string & line) {
            backup_line = line;
            viewing_idx = entries.size();
        }
        
        void end_viewing() {
            viewing_idx = SIZE_MAX;
            backup_line.clear();
        }
        
        bool is_viewing() const {
            return viewing_idx != SIZE_MAX;
        }
    } history;

    // ============================================================
    // readline_advanced (� �������� � ����������)
    // ============================================================

    static bool readline_advanced(std::string & line, bool multiline_input) {
        if (out != stdout) {
            fflush(stdout);
        }

        line.clear();
        std::vector<int> widths;
        bool is_special_char = false;
        bool end_of_stream = false;
        size_t byte_pos = 0;
        size_t char_pos = 0;

        char32_t input_char;
        while (true) {
            fflush(out);
            input_char = getchar32();

            if (input_char == '\r' || input_char == '\n') {
                break;
            }

            // ������� �� �������� �����/����
            if (input_char == 0xE002 || input_char == 0xE003) {
                if (input_char == 0xE002) {
                    if (!history.is_viewing()) {
                        history.begin_viewing(line);
                    }
                    std::string new_line;
                    if (history.prev(new_line)) {
                        line = new_line;
                    }
                } else {
                    if (history.is_viewing()) {
                        std::string new_line;
                        if (history.next(new_line)) {
                            line = new_line;
                        }
                    }
                }
                printf("\r\033[K");
                printf("%s", line.c_str());
                fflush(stdout);
                continue;
            }

            if (input_char == (char32_t) WEOF || input_char == 0x04) {
                end_of_stream = true;
                break;
            }

            if (is_special_char) {
                set_display(user_input);
                replace_last(line.back());
                is_special_char = false;
            }

            if (input_char == '\033') {
                char32_t code = getchar32();
                if (code == '[') {
                    while (true) {
                        code = getchar32();
                        if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~' || code == (char32_t) WEOF) {
                            break;
                        }
                    }
                }
                continue;
            }

            // ������� Left / Right
            if (input_char == 0xE000 || input_char == 0xE001) {
                if (input_char == 0xE000 && char_pos > 0) {
                    size_t prev_pos = prev_utf8_char_pos(line, byte_pos);
                    int width = 0;
                    for (size_t i = prev_pos; i < byte_pos; ) {
                        size_t advance = 0;
                        char32_t cp = decode_utf8(line, i, advance);
                        (void)cp;
                        width += estimateWidth(cp);
                        i += advance;
                    }
                    move_cursor(-width);
                    byte_pos = prev_pos;
                    char_pos--;
                } else if (input_char == 0xE001 && char_pos < widths.size()) {
                    size_t next_pos = next_utf8_char_pos(line, byte_pos);
                    int width = 0;
                    for (size_t i = byte_pos; i < next_pos; ) {
                        size_t advance = 0;
                        char32_t cp = decode_utf8(line, i, advance);
                        (void)cp;
                        width += estimateWidth(cp);
                        i += advance;
                    }
                    move_cursor(width);
                    byte_pos = next_pos;
                    char_pos++;
                }
                continue;
            }

            // Home / End
            if (input_char == 0xE004 || input_char == 0xE005) {
                if (input_char == 0xE004) {
                    int back_width = 0;
                    for (size_t i = 0; i < char_pos; i++) {
                        back_width += widths[i];
                    }
                    move_cursor(-back_width);
                    char_pos = 0;
                    byte_pos = 0;
                } else {
                    int forward_width = 0;
                    for (size_t i = char_pos; i < widths.size(); i++) {
                        forward_width += widths[i];
                    }
                    move_cursor(forward_width);
                    char_pos = widths.size();
                    byte_pos = line.length();
                }
                continue;
            }

            // Delete
            if (input_char == 0xE008) {
                if (char_pos < widths.size()) {
                    size_t next_pos = next_utf8_char_pos(line, byte_pos);
                    int w = widths[char_pos];
                    size_t char_len = next_pos - byte_pos;
                    line.erase(byte_pos, char_len);
                    widths.erase(widths.begin() + char_pos);
                    
                    size_t p = byte_pos;
                    for (size_t i = char_pos; i < widths.size(); i++) {
                        size_t following = next_utf8_char_pos(line, p);
                        put_codepoint(line.c_str() + p, following - p, widths[i]);
                        p = following;
                    }
                    move_cursor(-w);
                }
                continue;
            }

            if (input_char == 0x08 || input_char == 0x7F) {
                if (!widths.empty() && char_pos > 0) {
                    size_t prev_pos = prev_utf8_char_pos(line, byte_pos);
                    int w = widths[char_pos - 1];
                    size_t char_len = byte_pos - prev_pos;
                    line.erase(prev_pos, char_len);
                    widths.erase(widths.begin() + char_pos - 1);
                    
                    size_t p = prev_pos;
                    for (size_t i = char_pos - 1; i < widths.size(); i++) {
                        size_t following = next_utf8_char_pos(line, p);
                        put_codepoint(line.c_str() + p, following - p, widths[i]);
                        p = following;
                    }
                    move_cursor(-w);
                    byte_pos = prev_pos;
                    char_pos--;
                }
            } else {
                std::string new_char_str;
                append_utf8(input_char, new_char_str);
                int w = estimateWidth(input_char);

                if (char_pos == widths.size()) {
                    line += new_char_str;
                    int real_w = put_codepoint(new_char_str.c_str(), new_char_str.length(), w);
                    if (real_w < 0) real_w = 0;
                    widths.push_back(real_w);
                    byte_pos += new_char_str.length();
                    char_pos++;
                } else {
                    line.insert(byte_pos, new_char_str);
                    int real_w = put_codepoint(new_char_str.c_str(), new_char_str.length(), w);
                    if (real_w < 0) real_w = 0;
                    widths.insert(widths.begin() + char_pos, real_w);
                    byte_pos += new_char_str.length();
                    char_pos++;
                }
            }

            if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
                set_display(prompt);
                replace_last(line.back());
                is_special_char = true;
            }
        }

        bool has_more = multiline_input;
        if (is_special_char) {
            replace_last(' ');
            pop_cursor();

            char last = line.back();
            line.pop_back();
            if (last == '\\') {
                line += '\n';
                fputc('\n', out);
                has_more = !has_more;
            } else {
                if (line.length() == 1 && line.back() == ' ') {
                    line.clear();
                    pop_cursor();
                }
                has_more = false;
            }
        } else {
            if (end_of_stream) {
                has_more = false;
            } else {
                line += '\n';
                fputc('\n', out);
            }
        }

        if (!end_of_stream && !line.empty()) {
            std::string_view hline = line;
            if (!line.empty() && line.back() == '\n') {
                hline.remove_suffix(1);
            }
            history.add(hline);
        }

        fflush(out);
        return has_more;
    }

    static bool readline_simple(std::string & line, bool multiline_input) {
#if defined(_WIN32)
        std::wstring wline;
        if (!std::getline(std::wcin, wline)) {
            line.clear();
            GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
            return false;
        }

        int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), NULL, 0, NULL, NULL);
        line.resize(size_needed);
        WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), &line[0], size_needed, NULL, NULL);
#else
        if (!std::getline(std::cin, line)) {
            line.clear();
            return false;
        }
#endif
        if (!line.empty()) {
            char last = line.back();
            if (last == '/') {
                line.pop_back();
                return false;
            }
            if (last == '\\') {
                line.pop_back();
                multiline_input = !multiline_input;
            }
        }
        line += '\n';
        return multiline_input;
    }

    bool readline(std::string & line, bool multiline_input) {
        set_display(user_input);

        if (simple_io) {
            return readline_simple(line, multiline_input);
        }
        return readline_advanced(line, multiline_input);
    }

}