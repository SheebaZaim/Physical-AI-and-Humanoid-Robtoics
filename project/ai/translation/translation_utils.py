class LanguageCodes:
    """Utility class for managing language codes and names."""

    @staticmethod
    def get_supported_languages():
        """Return a dictionary of supported languages with codes as keys."""
        return {
            "en": "English",
            "ur": "Urdu",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "pt": "Portuguese",
            "it": "Italian",
            "nl": "Dutch",
            "sv": "Swedish",
            "tr": "Turkish",
        }

    @staticmethod
    def get_language_name(code: str) -> str:
        """Get the full name of a language from its code."""
        languages = LanguageCodes.get_supported_languages()
        return languages.get(code.lower(), code)

    @staticmethod
    def get_language_code(name: str) -> str:
        """Get the code of a language from its name."""
        languages = LanguageCodes.get_supported_languages()
        for code, lang_name in languages.items():
            if lang_name.lower() == name.lower():
                return code
        return name  # Return original if not found

    @staticmethod
    def is_right_to_left(code: str) -> bool:
        """Check if a language is written right-to-left."""
        rtl_languages = ["ar", "ur", "he", "fa", "dv", "ku", "ug", "yi", "sd", "prs", "ps", "ug", "sd", "ks"]
        return code.lower() in rtl_languages