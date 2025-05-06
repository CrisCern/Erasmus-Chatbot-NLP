import json
import time
from gliner import GLiNER
from tqdm import tqdm


file_input = "../erasmus_WA.json"
file_output = "erasmus_WA_ner_universal_FINAL.json"


model = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")


etichette_personalizzate = [
    "università",
    "sigla accademica",
    "livello linguistico",
    "lingua",
    "luogo",
    "organizzazione",
    "concetto Erasmus",
    "documento Erasmus",
    "programma di studio",
    "valuta",
    "data",
    "contatto"
]


falsi_positivi = {"informazioni", "rischi", "scegliere", "bisogno", "tranquilla", "cioè", "ahhh"}


documenti_erasmus = {"application form", "LA", "transcript of records", "grant agreement"}


lingue = {"inglese", "spagnolo", "francese", "tedesco", "italiano", "portoghese", "olandese", "svedese"}
livelli_linguistici = {"B1", "B2", "C1", "C2"}


with open(file_input, "r", encoding="utf-8") as file:
    messaggi = json.load(file)


risultati = []


print("\n Analizzando TUTTI i messaggi con il modello Universal NER_messaggi_singoli ITA...")
for msg in tqdm(messaggi, desc="Processando messaggi", unit="msg"):
    entita = model.predict_entities(msg["message"], etichette_personalizzate)

    refined_entities = []
    for e in entita:
        testo = e["text"]
        tipo = e["label"]

        if testo in falsi_positivi:
            continue

        if testo in documenti_erasmus:
            tipo = "documento Erasmus"

        if testo in lingue:
            tipo = "lingua"
        elif testo in livelli_linguistici:
            tipo = "livello linguistico"

        refined_entities.append({"testo": testo, "tipo": tipo})

    risultati.append({"messaggio": msg["message"], "entità": refined_entities})
    time.sleep(0.005)


with open(file_output, "w", encoding="utf-8") as f:
    json.dump(risultati, f, indent=4, ensure_ascii=False)

print(f"\n Analisi COMPLETA e corretta! File salvato in: {file_output}")
