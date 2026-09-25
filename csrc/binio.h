// File I/O for checkpoints and blueprints, streamed from and into the C++ tables (no Python objects):
//
//   * open_file        - UTF-8 path (what pybind11 hands over), wide-char API on Windows
//   * BinWriter        - buffered little-endian writer into <path>.tmp, renamed over <path> when
//                        finished (a crash mid-write never destroys the previous file); running
//                        checksum over every byte written
//   * BinReader        - buffered reader with the same checksum, throws on a short file
//   * JsonReader       - streaming tokenizer for the JSON checkpoints / blueprints Python writes
//                        (numbers through std::from_chars: correctly rounded, the same doubles as
//                        Python's float())
//   * py_float_repr,   - floats and strings spelled exactly as json.dump spells them, so a JSON
//     py_json_string     file written here has the bytes Python would write for the same values
#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace negp {

static_assert(sizeof(double) == 8 && sizeof(uint64_t) == 8, "64-bit doubles expected");

inline std::filesystem::path fs_path(const std::string& utf8) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
#else
    return std::filesystem::u8path(utf8);
#endif
}

inline FILE* open_file(const std::string& utf8, const char* mode) {
#if defined(_WIN32)
    std::wstring wmode(mode, mode + std::strlen(mode));
    return _wfopen(fs_path(utf8).c_str(), wmode.c_str());
#else
    return std::fopen(utf8.c_str(), mode);
#endif
}

// word-at-a-time checksum of a byte stream (independent of how the stream is chunked: the words are
// the little-endian 8-byte groups of the whole stream)
class StreamHash {
public:
    void update(const unsigned char* p, size_t n) {
        size_t i = 0;
        while (i < n && fill_ != 0) byte(p[i++]);  // complete a partial word
        for (; n - i >= 8; i += 8) {
            std::memcpy(&word_, p + i, 8);
            mix();
        }
        while (i < n) byte(p[i++]);
    }
    uint64_t value() const {
        uint64_t h = h_;
        if (fill_) { h ^= word_; h *= 0x9E3779B97F4A7C15ULL; h ^= h >> 31; }
        h ^= count_ + fill_;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33;
        return h;
    }

private:
    void byte(unsigned char c) {
        word_ |= (uint64_t)c << (8 * fill_);
        if (++fill_ == 8) mix();
    }
    void mix() {
        h_ ^= word_;
        h_ *= 0x9E3779B97F4A7C15ULL;
        h_ ^= h_ >> 31;
        word_ = 0;
        fill_ = 0;
        count_ += 8;
    }
    uint64_t h_ = 0x243F6A8885A308D3ULL, word_ = 0, count_ = 0;
    int fill_ = 0;
};

class BinWriter {
public:
    explicit BinWriter(const std::string& path) : path_(path), tmp_(path + ".tmp"), buf_(1 << 20) {
        f_ = open_file(tmp_, "wb");
        if (!f_) throw std::runtime_error("cannot write " + tmp_);
    }
    ~BinWriter() {
        if (f_) {  // not finished: an exception is on its way, drop the partial file
            std::fclose(f_);
            std::error_code ec;
            std::filesystem::remove(fs_path(tmp_), ec);
        }
    }
    BinWriter(const BinWriter&) = delete;
    BinWriter& operator=(const BinWriter&) = delete;

    void bytes(const void* p, size_t n) {
        const unsigned char* c = static_cast<const unsigned char*>(p);
        hash_.update(c, n);
        written_ += n;
        while (n) {
            size_t k = std::min(n, buf_.size() - pos_);
            std::memcpy(buf_.data() + pos_, c, k);
            pos_ += k;
            c += k;
            n -= k;
            if (pos_ == buf_.size()) flush();
        }
    }
    template <class T>
    void pod(const T& v) { bytes(&v, sizeof(T)); }  // little-endian hosts only (x86-64, ARM64)
    void u8(uint8_t v) { pod(v); }
    void u16(uint16_t v) { pod(v); }
    void u32(uint32_t v) { pod(v); }
    void i32(int32_t v) { pod(v); }
    void u64(uint64_t v) { pod(v); }
    void i64(int64_t v) { pod(v); }
    void f64(double v) { pod(v); }
    template <class T>
    void array(const std::vector<T>& v) { if (!v.empty()) bytes(v.data(), v.size() * sizeof(T)); }
    void str16(const char* s, size_t n) {
        if (n > 0xFFFF) throw std::runtime_error("string too long for the binary format");
        u16((uint16_t)n);
        bytes(s, n);
    }
    void str16(const std::string& s) { str16(s.data(), s.size()); }
    uint64_t checksum() const { return hash_.value(); }
    uint64_t written() const { return written_; }

    // flush, close and move the file into place
    void finish() {
        flush();
        if (std::fflush(f_) != 0 || std::fclose(f_) != 0) {
            f_ = nullptr;
            throw std::runtime_error("write error on " + tmp_);
        }
        f_ = nullptr;
        std::error_code ec;
        std::filesystem::rename(fs_path(tmp_), fs_path(path_), ec);  // replaces an existing file
        if (ec) throw std::runtime_error("cannot move " + tmp_ + " to " + path_ + ": " + ec.message());
    }

private:
    void flush() {
        if (pos_ && std::fwrite(buf_.data(), 1, pos_, f_) != pos_) throw std::runtime_error("write error on " + tmp_);
        pos_ = 0;
    }
    std::string path_, tmp_;
    FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t pos_ = 0;
    uint64_t written_ = 0;
    StreamHash hash_;
};

class BinReader {
public:
    explicit BinReader(const std::string& path) : path_(path), buf_(1 << 20) {
        std::error_code ec;
        size_ = (uint64_t)std::filesystem::file_size(fs_path(path), ec);
        if (ec) throw std::runtime_error("cannot read " + path + ": " + ec.message());
        f_ = open_file(path, "rb");
        if (!f_) throw std::runtime_error("cannot read " + path);
    }
    ~BinReader() { if (f_) std::fclose(f_); }
    BinReader(const BinReader&) = delete;
    BinReader& operator=(const BinReader&) = delete;

    void bytes(void* p, size_t n) {
        unsigned char* c = static_cast<unsigned char*>(p);
        size_t want = n;
        while (n) {
            if (pos_ == len_) refill();
            size_t k = std::min(n, len_ - pos_);
            std::memcpy(c, buf_.data() + pos_, k);
            pos_ += k;
            c += k;
            n -= k;
        }
        hash_.update(static_cast<unsigned char*>(p), want);
        read_ += want;
    }
    template <class T>
    T pod() { T v; bytes(&v, sizeof(T)); return v; }
    uint8_t u8() { return pod<uint8_t>(); }
    uint16_t u16() { return pod<uint16_t>(); }
    uint32_t u32() { return pod<uint32_t>(); }
    int32_t i32() { return pod<int32_t>(); }
    uint64_t u64() { return pod<uint64_t>(); }
    int64_t i64() { return pod<int64_t>(); }
    double f64() { return pod<double>(); }
    std::string str16() {
        std::string s;
        str16(s);
        return s;
    }
    void str16(std::string& s) {  // into an existing string (keeps its capacity)
        uint16_t n = u16();
        s.resize(n);
        if (n) bytes(&s[0], n);
    }
    // n values into v (replacing its contents); refuses sizes the file cannot hold
    template <class T>
    void array(std::vector<T>& v, uint64_t n) {
        if (n > remaining() / sizeof(T)) throw std::runtime_error("corrupt file (array longer than the file): " + path_);
        v.resize((size_t)n);
        if (n) bytes(v.data(), (size_t)n * sizeof(T));
    }
    // bytes left after the read position (the file size is taken once, at open)
    uint64_t remaining() const { return size_ - read_; }
    uint64_t checksum() const { return hash_.value(); }
    uint64_t read_bytes() const { return read_; }
    // the stored checksum of everything read so far must follow
    void expect_checksum(const char* what) {
        const uint64_t want = checksum();
        const uint64_t got = pod<uint64_t>();
        if (got != want) throw std::runtime_error(std::string("corrupt file (") + what + " checksum): " + path_);
    }
    const std::string& path() const { return path_; }

private:
    void refill() {
        len_ = std::fread(buf_.data(), 1, buf_.size(), f_);
        pos_ = 0;
        if (len_ == 0) throw std::runtime_error("unexpected end of file: " + path_);
    }
    std::string path_;
    FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t pos_ = 0, len_ = 0;
    uint64_t read_ = 0, size_ = 0;
    StreamHash hash_;
};

// first bytes of a file ("" if it cannot be read): tells binary from JSON
inline std::string file_magic(const std::string& path, size_t n = 8) {
    FILE* f = open_file(path, "rb");
    if (!f) return "";
    std::string s(n, '\0');
    size_t k = std::fread(&s[0], 1, n, f);
    std::fclose(f);
    s.resize(k);
    return s;
}

// ------------------------------------------------------------------ JSON
// Streaming tokenizer for the JSON files Python's json module writes (and any valid JSON): the
// caller walks the structure it expects and skips the rest with skip_value().
class JsonReader {
public:
    explicit JsonReader(const std::string& path) : path_(path), buf_(1 << 20) {
        f_ = open_file(path, "rb");
        if (!f_) throw std::runtime_error("cannot read " + path);
    }
    ~JsonReader() { if (f_) std::fclose(f_); }
    JsonReader(const JsonReader&) = delete;
    JsonReader& operator=(const JsonReader&) = delete;

    // next non-space character without consuming it (0 at the end of the file)
    char peek() {
        skip_ws();
        return at_end() ? '\0' : buf_[pos_];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        pos_++;
    }
    bool consume(char c) {
        if (peek() != c) return false;
        pos_++;
        return true;
    }
    // members of an object / items of an array: call after '{' / '[' and after each item
    bool next_member(bool& first, char close) {
        if (first) {
            first = false;
            if (consume(close)) return false;
            return true;
        }
        if (consume(',')) return true;
        expect(close);
        return false;
    }
    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            char c = raw();
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            char e = raw();
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {  // surrogate pair
                        if (raw() != '\\' || raw() != 'u') fail("bad surrogate pair");
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(cp, out);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    // a number (JSON, plus Python's NaN / Infinity / -Infinity) as a double: std::from_chars is
    // correctly rounded, so this is the double Python's float() gives for the same text
    double number() {
        const std::string s = number_token();
        if (s == "NaN") return std::numeric_limits<double>::quiet_NaN();
        if (s == "Infinity") return std::numeric_limits<double>::infinity();
        if (s == "-Infinity") return -std::numeric_limits<double>::infinity();
        double v = 0.0;
        auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc() || r.ptr != s.data() + s.size()) fail("bad number '" + s + "'");
        return v;
    }
    int64_t integer() {
        const std::string s = number_token();
        int64_t v = 0;
        auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc() || r.ptr != s.data() + s.size()) fail("expected an integer, got '" + s + "'");
        return v;
    }
    bool boolean() {
        char c = peek();
        if (c == 't') { literal("true"); return true; }
        if (c == 'f') { literal("false"); return false; }
        fail("expected true or false");
        return false;
    }
    void skip_value() {
        char c = peek();
        if (c == '{') {
            pos_++;
            bool first = true;
            while (next_member(first, '}')) { string(); expect(':'); skip_value(); }
        } else if (c == '[') {
            pos_++;
            bool first = true;
            while (next_member(first, ']')) skip_value();
        } else if (c == '"') {
            string();
        } else if (c == 't') {
            literal("true");
        } else if (c == 'f') {
            literal("false");
        } else if (c == 'n') {
            literal("null");
        } else {
            number();
        }
    }
    [[noreturn]] void fail(const std::string& what) {
        throw std::runtime_error("JSON " + what + " at byte " + std::to_string(offset_ + pos_) + " of " + path_);
    }

private:
    std::string number_token() {
        skip_ws();
        std::string s;
        while (!at_end()) {
            char c = buf_[pos_];
            const bool part = (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' ||
                              c == 'N' || c == 'a' || c == 'I' || c == 'n' || c == 'f' || c == 'i' || c == 't' || c == 'y';
            if (!part) break;
            if (s.size() >= 64) fail("number too long");
            s += c;
            pos_++;
        }
        if (s.empty()) fail("expected a number");
        return s;
    }
    bool at_end() {
        if (pos_ < len_) return false;
        refill();
        return len_ == 0;
    }
    void refill() {
        offset_ += len_;
        len_ = std::fread(buf_.data(), 1, buf_.size(), f_);
        pos_ = 0;
    }
    void skip_ws() {
        for (;;) {
            if (at_end()) return;
            char c = buf_[pos_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') pos_++;
            else return;
        }
    }
    char raw() {
        if (at_end()) fail("unexpected end");
        return buf_[pos_++];
    }
    uint32_t hex4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = raw();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    static void utf8(uint32_t cp, std::string& out) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }
    void literal(const char* word) {
        skip_ws();
        for (const char* p = word; *p; p++) if (raw() != *p) fail(std::string("expected ") + word);
    }
    std::string path_;
    FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t pos_ = 0, len_ = 0;
    uint64_t offset_ = 0;
};

// A double spelled the way json.dump spells a float: NaN / Infinity / -Infinity, else float.__repr__
// (CPython float_repr_style 'short' = PyOS_double_to_string(x, 'r', 0, Py_DTSF_ADD_DOT_0)): the
// shortest digits that read back to the same double (std::to_chars), in fixed notation when
// 1e-4 <= |x| < 1e16 (".0" added to integral values), else d.ddde-XX (signed exponent, at least two
// digits).  So a file written here has the bytes json.dump writes for the same values.
inline void py_float_repr(double v, std::string& out) {
    if (v != v) { out += "NaN"; return; }
    if (v == std::numeric_limits<double>::infinity()) { out += "Infinity"; return; }
    if (v == -std::numeric_limits<double>::infinity()) { out += "-Infinity"; return; }
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // [-]d[.ddd]e(+|-)XX
    const char* p = buf;
    if (*p == '-') { out += '-'; p++; }
    char digits[32];
    int nd = 0;
    for (; p < r.ptr && *p != 'e'; p++) if (*p != '.') digits[nd++] = *p;
    int e10 = 0;
    std::from_chars(p + 1 + (p[1] == '+' ? 1 : 0), r.ptr, e10);
    const int decpt = e10 + 1;  // value = 0.d1d2...dn * 10^decpt, as _Py_dg_dtoa reports it
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) { out += '.'; out.append(digits + 1, (size_t)(nd - 1)); }
        const int e = decpt - 1;
        out += e < 0 ? "e-" : "e+";
        const int ae = e < 0 ? -e : e;
        if (ae < 10) out += '0';
        out += std::to_string(ae);
    } else if (decpt <= 0) {
        out += "0.";
        out.append((size_t)(-decpt), '0');
        out.append(digits, (size_t)nd);
    } else if (decpt < nd) {
        out.append(digits, (size_t)decpt);
        out += '.';
        out.append(digits + decpt, (size_t)(nd - decpt));
    } else {
        out.append(digits, (size_t)nd);
        out.append((size_t)(decpt - nd), '0');
        out += ".0";
    }
}

// A UTF-8 string as json.dump writes a str (ensure_ascii=True): \" \\ \n \r \t \b \f, other
// characters outside ' '..'~' as \uXXXX (UTF-16 surrogate pairs above U+FFFF).
inline void py_json_string(const char* s, size_t n, std::string& out) {
    static const char* hex = "0123456789abcdef";
    auto u4 = [&](uint32_t c) {
        out += "\\u";
        for (int sh = 12; sh >= 0; sh -= 4) out += hex[(c >> sh) & 0xF];
    };
    out += '"';
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c >= ' ' && c <= '~' && c != '\\' && c != '"') { out += (char)c; i++; continue; }
        if (c < 0x80) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default: u4(c);
            }
            i++;
            continue;
        }
        // a multi-byte UTF-8 sequence (the strings come from Python str, so they are valid)
        int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
        uint32_t cp = c & (len == 4 ? 0x07 : len == 3 ? 0x0F : 0x1F);
        for (int k = 1; k < len && i + k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        i += (size_t)len;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            u4(0xD800 + (cp >> 10));
            u4(0xDC00 + (cp & 0x3FF));
        } else {
            u4(cp);
        }
    }
    out += '"';
}

// buffered text output for JSON exports (same .tmp + rename as BinWriter)
class TextWriter {
public:
    explicit TextWriter(const std::string& path) : w_(path) {}
    std::string& buf() { return s_; }
    void maybe_flush() { if (s_.size() > (1 << 20)) flush(); }
    void flush() { w_.bytes(s_.data(), s_.size()); s_.clear(); }
    void finish() { flush(); w_.finish(); }

private:
    BinWriter w_;
    std::string s_;
};

}  // namespace negp
