# Importuj wszystkie klasy nodów z Twojego nowego, zbiorczego pliku
from .nodes import *

# Definicja centralnej konfiguracji nodów
# Klucz: Wewnętrzna nazwa nodu (unikalna)
# Wartość: Słownik zawierający:
#   - "class": Odwołanie do klasy nodu
#   - "name": Nazwa wyświetlana w interfejsie ComfyUI
NODE_CONFIG = {
     "PathTool": {"class": PathTool, "name": "Path Tool"},
     "ColorMatchFalloff": {"class": ColorMatchFalloff, "name": "Color Match Falloff"},
}

# UWAGA: Upewnij się, że nazwy klas (np. HyvidSwitcher, MidiAnalyzer) 
# są dokładnie takie same jak w pliku nodes.py!

# Funkcja generująca mapowania dla ComfyUI na podstawie NODE_CONFIG
def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        # Użyj podanej nazwy "name" lub, jeśli jej nie ma, nazwy klasy
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

# Generowanie mapowań, które ComfyUI rozumie
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

# Ustawienie katalogu web i eksport zmiennych
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("### Loading: Patatajec-ComfyUI (Success) ###")

