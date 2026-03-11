// sherpa-onnx/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

#include <fstream>
#include <locale>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"  // NOLINT
#include "phonemize.hpp"    // NOLINT
#include <algorithm>
#include <cctype>

#include "sherpa-onnx/csrc/file-utils.h"

// Cyrillic uppercase (U+0410-U+042F) -> lowercase (+0x20)
static char32_t ToLowerCyrillic(char32_t c) {
  if (c >= 0x0410 && c <= 0x042F) {
    return c + 0x20;
  }
  // Є U+0404 -> є U+0454, І U+0406 -> і U+0456, Ї U+0407 -> ї U+0457,
  // Ё U+0401 -> ё U+0451, Й U+0419 -> й U+0439 (same +0x20)
  if (c == 0x0404) return 0x0454;  // Є -> є
  if (c == 0x0406) return 0x0456;  // І -> і
  if (c == 0x0407) return 0x0457;  // Ї -> ї
  if (c == 0x0401) return 0x0451;  // Ё -> ё
  return c;
}

// Normalize Ukrainian for Piper phoneme_type=text models.
// 1) Compose decomposed і+̈ -> ї so we have a single char to process.
// 2) For ї: use precomposed if in token2id, else expand to і+̈ (decomposed).
//    Supports both model formats (precomposed vs decomposed training).
static void NormalizeUkrainianForPiper(
    std::u32string *s,
    const std::unordered_map<char32_t, int32_t> &token2id) {
  std::u32string composed;
  composed.reserve(s->size());
  for (size_t i = 0; i < s->size(); ++i) {
    char32_t c = (*s)[i];
    if (i + 1 < s->size()) {
      char32_t next = (*s)[i + 1];
      if (next == 0x0308) {  // combining diaeresis ̈
        if (c == 0x0456) {   // і + ̈ -> ї (U+0457)
          composed.push_back(0x0457);
          ++i;
          continue;
        }
      }
    }
    composed.push_back(c);
  }
  // ї: prefer decomposed (і+̈) when model has both tokens—some models train on
  // that form. Otherwise use precomposed if present.
  std::u32string out;
  out.reserve(composed.size() * 2);
  for (char32_t c : composed) {
    if (c == 0x0457) {  // ї
      if (token2id.count(0x0456) && token2id.count(0x0308)) {
        out.push_back(0x0456);
        out.push_back(0x0308);
      } else if (token2id.count(0x0457)) {
        out.push_back(0x0457);
      } else {
        out.push_back(0x0457);
      }
    } else {
      out.push_back(c);
    }
  }
  *s = std::move(out);
}

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// Encode a single char32_t to UTF-8 string. For debugging only
static std::string ToString(char32_t cp) {
  std::string result;

  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0x10FFFF) {
    result += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    SHERPA_ONNX_LOGE("Invalid Unicode code point: %d",
                     static_cast<int32_t>(cp));
  }

  return result;
}

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes) {
  static std::mutex espeak_mutex;

  std::lock_guard<std::mutex> lock(espeak_mutex);

  // keep multi threads from calling into piper::phonemize_eSpeak
  piper::phonemize_eSpeak(text, config, *phonemes);
}

static std::unordered_map<char32_t, int32_t> ReadTokens(std::istream &is) {
  std::unordered_map<char32_t, int32_t> token2id;

  std::string line;

  std::string sym;
  std::u32string s;
  int32_t id = 0;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error when reading tokens: %s", line.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    s = Utf8ToUtf32(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }

      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    char32_t c = s[0];

    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      SHERPA_ONNX_EXIT(-1);
    }

    token2id.insert({c, id});
  }

  return token2id;
}

// see the function "phonemes_to_ids" from
// https://github.com/rhasspy/piper/blob/master/notebooks/piper_inference_(ONNX).ipynb
static std::vector<int64_t> PiperPhonemesToIdsVits(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes) {
  // see
  // https://github.com/rhasspy/piper-phonemize/blob/master/src/phoneme_ids.hpp#L17
  int32_t pad = token2id.at(U'_');
  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  std::vector<int64_t> ans;
  ans.reserve(phonemes.size());

  ans.push_back(bos);
  for (auto p : phonemes) {
    if (token2id.count(p)) {
      ans.push_back(token2id.at(p));
      ans.push_back(pad);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }
  ans.push_back(eos);

  return ans;
}

// For Piper models with phoneme_type "text" (e.g., Ukrainian): use character
// codepoints as phonemes. Each character maps to itself; tokens.txt contains
// Cyrillic letters. See Piper JSON config "phoneme_type": "text".
static std::vector<int64_t> TextToIdsVitsCodepoints(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::u32string &text) {
  int32_t pad = token2id.at(U'_');
  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  std::vector<int64_t> ans;
  ans.reserve(text.size() * 2 + 3);
  ans.push_back(bos);

  for (char32_t c : text) {
    if (token2id.count(c)) {
      ans.push_back(token2id.at(c));
      ans.push_back(pad);
    }
    // Skip unknown characters (e.g., unsupported punctuation)
  }
  ans.push_back(eos);
  return ans;
}

static std::vector<TokenIDs> PiperPhonemesToIdsVitsCodepoints(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::string &text) {
  std::u32string s = Utf8ToUtf32(text);

  NormalizeUkrainianForPiper(&s, token2id);

  // Lowercase for case-insensitive token lookup (Ukrainian tokens are
  // lowercase). Use Cyrillic-aware lowercasing.
  for (char32_t &c : s) {
    if (c >= 0x41 && c <= 0x5A) {
      c = c + 0x20;  // ASCII A-Z -> a-z
    } else {
      c = ToLowerCyrillic(c);
    }
  }

  std::vector<TokenIDs> ans;
  std::u32string current_sentence;

  auto flush_sentence = [&]() {
    if (!current_sentence.empty()) {
      ans.emplace_back(
          TextToIdsVitsCodepoints(token2id, current_sentence));
      current_sentence.clear();
    }
  };

  for (size_t i = 0; i < s.size(); ++i) {
    char32_t c = s[i];
    if (c == U'.' || c == U'!' || c == U'?' || c == U';' || c == U':') {
      flush_sentence();
    } else if (token2id.count(c) || c == U' ' || c == U',' || c == U'-' ||
               c == U'\'') {
      current_sentence.push_back(c);
    } else {
      SHERPA_ONNX_LOGE(
          "Skip unknown char in codepoint phonemization: U+%04X ('%s')",
          static_cast<uint32_t>(c), ToString(c).c_str());
    }
  }
  flush_sentence();

  if (ans.empty()) {
    ans.emplace_back(TextToIdsVitsCodepoints(token2id, std::u32string()));
  }
  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsVitsCodepoints(
    const std::string &text) const {
  return PiperPhonemesToIdsVitsCodepoints(token2id_, text);
}

static std::vector<std::vector<int64_t>> PiperPhonemesToIdsMatcha(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, bool use_eos_bos,
    int32_t max_token_len = 400) {
  // We set max_token_len to 400 here to fix
  // https://github.com/k2-fsa/sherpa-onnx/issues/2666
  std::vector<std::vector<int64_t>> ans;
  std::vector<int64_t> current;

  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  if (use_eos_bos) {
    current.push_back(bos);
  }

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      current.push_back(token2id.at(p));
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }

    if (current.size() > max_token_len + 1) {
      if (use_eos_bos) {
        current.push_back(eos);
      }

      ans.push_back(std::move(current));

      if (use_eos_bos) {
        current.push_back(bos);
      }
    }
  }  // for (auto p : phonemes)

  if (!current.empty()) {
    if (use_eos_bos) {
      if (current.size() > 1) {
        current.push_back(eos);

        ans.push_back(std::move(current));
      }
    } else {
      ans.push_back(std::move(current));
    }
  }

  return ans;
}

static std::vector<std::vector<int64_t>> PiperPhonemesToIdsKokoroOrKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, int32_t max_len) {
  std::vector<std::vector<int64_t>> ans;

  std::vector<int64_t> current;
  current.reserve(phonemes.size());

  current.push_back(0);

  for (auto p : phonemes) {
    // SHERPA_ONNX_LOGE("%d %s", static_cast<int32_t>(p), ToString(p).c_str());
    if (token2id.count(p)) {
      if (current.size() > max_len - 1) {
        current.push_back(0);
        ans.push_back(std::move(current));

        current.reserve(phonemes.size());
        current.push_back(0);
      }

      current.push_back(token2id.at(p));
      if (p == '.') {
        current.push_back(token2id.at(' '));
      }
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }

  current.push_back(0);
  ans.push_back(std::move(current));
  return ans;
}

static std::vector<int64_t> CoquiPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes,
    const OfflineTtsVitsModelMetaData &vits_meta_data) {
  // see
  // https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py#L87
  int32_t use_eos_bos = vits_meta_data.use_eos_bos;
  int32_t bos_id = vits_meta_data.bos_id;
  int32_t eos_id = vits_meta_data.eos_id;
  int32_t blank_id = vits_meta_data.blank_id;
  int32_t add_blank = vits_meta_data.add_blank;
  int32_t comma_id = token2id.at(',');

  std::vector<int64_t> ans;
  if (add_blank) {
    ans.reserve(phonemes.size() * 2 + 3);
  } else {
    ans.reserve(phonemes.size() + 2);
  }

  if (use_eos_bos) {
    ans.push_back(bos_id);
  }

  if (add_blank) {
    ans.push_back(blank_id);

    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
        ans.push_back(blank_id);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  } else {
    // not adding blank
    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  }

  // add a comma at the end of a sentence so that we can have a longer pause.
  ans.push_back(comma_id);

  if (use_eos_bos) {
    ans.push_back(eos_id);
  }

  return ans;
}

void InitEspeak(const std::string &data_dir) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [data_dir]() {
#if __ANDROID_API__ >= 9 || defined(__OHOS__)
    if (data_dir[0] != '/') {
      SHERPA_ONNX_LOGE(
          "You need to follow our examples to copy the espeak-ng-data "
          "directory from the assets folder to an external storage directory.");

      SHERPA_ONNX_LOGE(
          "Hint: Please see\n"
          "https://github.com/k2-fsa/sherpa-onnx/blob/master/android/"
          "SherpaOnnxTtsEngine/app/src/main/java/com/k2fsa/sherpa/onnx/tts/"
          "engine/TtsEngine.kt#L188\n"
          "The function copyDataDir()\n");
    }
#endif

    int32_t result =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.c_str(), 0);
    if (result != 22050) {
      SHERPA_ONNX_LOGE(
          "Failed to initialize espeak-ng with data dir: %s. Return code is: "
          "%d",
          data_dir.c_str(), result);
      SHERPA_ONNX_EXIT(-1);
    }
  });
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data,
    const std::string &phoneme_type)
    : vits_meta_data_(vits_meta_data),
      use_codepoint_phonemes_(phoneme_type == "text") {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  if (!use_codepoint_phonemes_) {
    InitEspeak(data_dir);
  }
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data,
    const std::string &phoneme_type)
    : vits_meta_data_(vits_meta_data),
      use_codepoint_phonemes_(phoneme_type == "text") {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istringstream is(std::string(buf.data(), buf.size()));
    token2id_ = ReadTokens(is);
  }

  if (!use_codepoint_phonemes_) {
    // We should copy the directory of espeak-ng-data from the asset to
    // some internal or external storage and then pass the directory to
    // data_dir.
    InitEspeak(data_dir);
  }
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data)
    : kitten_meta_data_(kitten_meta_data), is_kitten_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istringstream is(std::string(buf.data(), buf.size()));
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istringstream is(std::string(buf.data(), buf.size()));
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data)
    : kitten_meta_data_(kitten_meta_data), is_kitten_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istringstream is(std::string(buf.data(), buf.size()));
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &voice /*= ""*/) const {
  if (is_matcha_) {
    return ConvertTextToTokenIdsMatcha(text, voice);
  } else if (is_kokoro_) {
    return ConvertTextToTokenIdsKokoroOrKitten(
        token2id_, kokoro_meta_data_.max_token_len, text, voice);
  } else if (is_kitten_) {
    return ConvertTextToTokenIdsKokoroOrKitten(
        token2id_, kitten_meta_data_.max_token_len, text, voice);
  } else {
    return ConvertTextToTokenIdsVits(text, voice);
  }
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsMatcha(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  for (const auto &p : phonemes) {
    auto phoneme_ids =
        PiperPhonemesToIdsMatcha(token2id_, p, matcha_meta_data_.use_eos_bos);

    for (auto &ids : phoneme_ids) {
      ans.emplace_back(std::move(ids));
    }
  }

  return ans;
}

std::vector<TokenIDs> ConvertTextToTokenIdsKokoroOrKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    int32_t max_token_len, const std::string &text,
    const std::string &voice /*= ""*/) {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  for (const auto &p : phonemes) {
    auto phoneme_ids =
        PiperPhonemesToIdsKokoroOrKitten(token2id, p, max_token_len);

    for (auto &ids : phoneme_ids) {
      ans.emplace_back(std::move(ids));
    }
  }

  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsVits(
    const std::string &text, const std::string &voice /*= ""*/) const {
  if (use_codepoint_phonemes_) {
    return ConvertTextToTokenIdsVitsCodepoints(text);
  }

  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(text, config, &phonemes);

  std::vector<TokenIDs> ans;

  std::vector<int64_t> phoneme_ids;

  if (vits_meta_data_.is_piper || vits_meta_data_.is_icefall) {
    for (const auto &p : phonemes) {
      phoneme_ids = PiperPhonemesToIdsVits(token2id_, p);
      ans.emplace_back(std::move(phoneme_ids));
    }
  } else if (vits_meta_data_.is_coqui) {
    for (const auto &p : phonemes) {
      phoneme_ids = CoquiPhonemesToIds(token2id_, p, vits_meta_data_);
      ans.emplace_back(std::move(phoneme_ids));
    }

  } else {
    SHERPA_ONNX_LOGE("Unsupported model");
    SHERPA_ONNX_EXIT(-1);
  }

  return ans;
}

#if __ANDROID_API__ >= 9
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kokoro_meta_data);
#endif

#if __OHOS__
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kokoro_meta_data);
#endif

}  // namespace sherpa_onnx
