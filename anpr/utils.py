import re

# Regex for Indian license plate format
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

def normalize_plate(text: str) -> str:
    clean = re.sub(r'[^A-Z0-9\n]', '', text)
    return clean.replace("\n", "")
