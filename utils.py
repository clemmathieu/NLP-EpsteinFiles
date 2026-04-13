
#words that spaCy incorretly marked as PERSON
NON_PERSON_TOKENS = frozenset({
    # Social media & tech apps
    "twitter", "facebook", "instagram", "google", "youtube", "snapchat",
    "whatsapp", "linkedin", "tiktok", "apple", "microsoft", "amazon",
    "netflix", "uber", "lyft", "paypal", "venmo", "blackberry", "nokia",
    "telegram", "signal", "skype", "zoom", "slack",
    # News & media
    "cnn", "bbc", "nbc", "abc", "cbs", "msnbc", "fox", "fox news",
    "new york times", "washington post", "reuters", "ap",
    "associated press", "daily mail", "new york post",
    # Government agencies
    "fbi", "cia", "nsa", "doj", "sec", "dea", "nypd", "interpol",
    "u.s.", "u.k.", "usa", "america",
    # Honorifics / titles extracted on their own
    "mr", "ms", "mrs", "dr", "sir", "lord", "lady", "hon", "esq",
})


#categories taken from Epstein case facts
OFFENSE_CATEGORIES = [
    "Sexual exploitation or trafficking",
    "Financial fraud or money laundering",
    "Obstruction or witness tampering",
    "Bribery or corruption",
    "Coercion or blackmail",
    "Network facilitation or coordination",
]