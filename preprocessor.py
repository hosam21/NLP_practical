import re
import string
import emoji
import logging
from typing import Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmojiHandler:
    def __init__(self):
        # Define emoji sentiment mappings
        self.positive_emojis = {
            '😊': 'رمز تعبيري إيجابي',
            '😄': 'رمز تعبيري إيجابي',
            '😃': 'رمز تعبيري إيجابي',
            '😀': 'رمز تعبيري إيجابي',
            '😁': 'رمز تعبيري إيجابي',
            '😆': 'رمز تعبيري إيجابي',
            '😍': 'رمز تعبيري إيجابي',
            '🥰': 'رمز تعبيري إيجابي',
            '😘': 'رمز تعبيري إيجابي',
            '❤️': 'رمز تعبيري إيجابي',
            '👍': 'رمز تعبيري إيجابي',
            '🙌': 'رمز تعبيري إيجابي',
            '🎉': 'رمز تعبيري إيجابي',
            '✨': 'رمز تعبيري إيجابي',
            '🌟': 'رمز تعبيري إيجابي'
        }

        self.negative_emojis = {
            '😞': 'رمز تعبيري سلبي',
            '😔': 'رمز تعبيري سلبي',
            '😟': 'رمز تعبيري سلبي',
            '😕': 'رمز تعبيري سلبي',
            '😣': 'رمز تعبيري سلبي',
            '😖': 'رمز تعبيري سلبي',
            '😫': 'رمز تعبيري سلبي',
            '😩': 'رمز تعبيري سلبي',
            '😢': 'رمز تعبيري سلبي',
            '😭': 'رمز تعبيري سلبي',
            '😤': 'رمز تعبيري سلبي',
            '😠': 'رمز تعبيري سلبي',
            '😡': 'رمز تعبيري سلبي',
            '👎': 'رمز تعبيري سلبي',
            '💔': 'رمز تعبيري سلبي'
        }

        self.sarcastic_emojis = {
            '😏': 'رمز تعبيري ساخر',
            '😒': 'رمز تعبيري ساخر',
            '🙄': 'رمز تعبيري ساخر',
            '😑': 'رمز تعبيري ساخر',
            '😐': 'رمز تعبيري ساخر',
            '🤔': 'رمز تعبيري ساخر',
            '🤨': 'رمز تعبيري ساخر',
            '😶': 'رمز تعبيري ساخر',
            '😏': 'رمز تعبيري ساخر',
            '😌': 'رمز تعبيري ساخر'
        }

        # Combine all emoji mappings
        self.emoji_mappings = {**self.positive_emojis, **self.negative_emojis, **self.sarcastic_emojis}
        logger.info(f"Initialized EmojiHandler with {len(self.emoji_mappings)} emoji mappings")

    def process_emojis(self, text):
        """Convert emojis to meaningful text representations"""
        # First, get all emojis in the text
        emojis = emoji.emoji_list(text)

        if emojis:
            logger.debug(f"Found {len(emojis)} emojis in text: {text}")

        # Sort emojis by their position in reverse order to avoid position shifting
        emojis.sort(key=lambda x: x['match_start'], reverse=True)

        # Replace each emoji with its sentiment representation
        for emoji_info in emojis:
            emoji_char = emoji_info['emoji']
            if emoji_char in self.emoji_mappings:
                replacement = f" {self.emoji_mappings[emoji_char]} "
                text = text[:emoji_info['match_start']] + replacement + text[emoji_info['match_end']:]
                logger.debug(f"Replaced emoji {emoji_char} with {self.emoji_mappings[emoji_char]}")

        return text

class ArabicTextPreprocessor:
    def __init__(self, stopwords_path: str, sentiment_words_path: str):
        # Initialize Arabic stopwords
        with open(stopwords_path, encoding="utf8") as f:
            self.stop_words: Set[str] = {w.strip() for w in f if w.strip()}
        
        # Load sentiment words
        sentiment_words = set()
        with open(sentiment_words_path, encoding="utf8") as lex:
            for line in lex:
                parts = line.strip().split()
                if parts:
                    sentiment_words.add(parts[0])
        
        # Define sarcasm cues
        sarcasm_cues = {
            "لا"و"هههه", "عيب", "يا سلام", "حقًا", "طبعًا", "طبعاً", "أيوه", "أها"
        }
        
        # Build filtered stopwords
        self.stop_words = self.stop_words - sentiment_words - sarcasm_cues
        
        # Define Arabic and English punctuation
        self.punctuations = string.punctuation + '،؛؟'
        
        # Define Arabic diacritics pattern
        self.arabic_diacritics = re.compile("""
                                         ّ    | # Tashdid
                                         َ    | # Fatha
                                         ً    | # Tanwin Fath
                                         ُ    | # Damma
                                         ٌ    | # Tanwin Damm
                                         ِ    | # Kasra
                                         ٍ    | # Tanwin Kasr
                                         ْ    | # Sukun
                                         ـ     # Tatwil/Kashida
                                         """, re.VERBOSE)

        # Define common Arabic noise patterns
        self.noise_patterns = {
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]': '',  # Keep only Arabic characters
            r'\s+': ' ',  # Replace multiple spaces with single space
            r'[^\w\s]': '',  # Remove special characters
            r'[a-zA-Z]': '',  # Remove English characters
        }

        # Initialize emoji handler
        self.emoji_handler = EmojiHandler()
        logger.info("Initialized ArabicTextPreprocessor")

    def remove_duplicate_phrases(self, text: str, max_phrase_length: int = 3) -> str:
        """Remove duplicate phrases of length 2 or more words while preserving word order."""
        if not text:
            return text

        words = text.split()
        if len(words) < 2:
            return text

        for phrase_length in range(max_phrase_length, 1, -1):
            i = 0
            while i < len(words) - phrase_length:
                current_phrase = ' '.join(words[i:i + phrase_length])
                j = i + phrase_length
                while j < len(words) - phrase_length + 1:
                    next_phrase = ' '.join(words[j:j + phrase_length])
                    if current_phrase == next_phrase:
                        words = words[:j] + words[j + phrase_length:]
                    else:
                        j += 1
                i += 1

        return ' '.join(words)

    def remove_duplicate_words(self, text: str) -> str:
        """Remove duplicate words while preserving word order."""
        if not text:
            return text

        words = text.split()
        if not words:
            return text

        result = []
        prev_word = None

        for word in words:
            if word != prev_word:
                result.append(word)
                prev_word = word

        return ' '.join(result)

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text"""
        text = re.sub(self.arabic_diacritics, '', text)
        return text

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    def remove_mentions(self, text: str) -> str:
        """Remove Twitter mentions"""
        return re.sub(r'@\w+', '', text)

    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags"""
        return re.sub(r'#\w+', '', text)

    def remove_numbers(self, text: str) -> str:
        """Remove numbers"""
        return re.sub(r'\d+', '', text)

    def remove_punctuations(self, text: str) -> str:
        """Remove punctuations"""
        translator = str.maketrans('', '', self.punctuations)
        return text.translate(translator)

    def remove_stopwords(self, text: str) -> str:
        """Remove Arabic stopwords"""
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])

    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text by standardizing characters"""
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)
        return text

    def remove_noise(self, text: str) -> str:
        """Remove noise patterns from text"""
        for pattern, replacement in self.noise_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to the text"""
        if not isinstance(text, str):
            return ""

        # Process emojis first
        original_text = text
        text = self.emoji_handler.process_emojis(text)

        if text != original_text:
            logger.info(f"Emoji processing changed text from '{original_text}' to '{text}'")

        # Apply other preprocessing steps
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuations(text)
        text = self.normalize_arabic_text(text)
        text = self.remove_diacritics(text)
        text = self.remove_noise(text)
        text = self.remove_stopwords(text)
        text = self.remove_duplicate_words(text)
        text = self.remove_duplicate_phrases(text)
        text = ' '.join(text.split())
        return text 
