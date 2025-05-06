import re
import json
import os
import csv

def parse_whatsapp_chat(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    messages = []
    skipped = 0
    pattern = r"\[(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)"

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                date, time, sender, message = match.groups()

                # Pulizia caratteri invisibili
                message = message.replace("\u200e", "").replace("\u202f", "").strip()
                sender = sender.replace("\u200e", "").replace("\u202f", "").strip()

                # Filtra messaggi di sistema
                if any(phrase in message for phrase in [
                    "ha usato il link d'invito per entrare nel gruppo",
                    "ha creato questo gruppo",
                    "I messaggi e le chiamate sono crittografati",
                    "Hai usato il link d'invito per entrare nel gruppo.",
                    "Questo messaggio è stato eliminato.",
                    "Lorenzo Dulcis ha rimosso"
                ]):
                    skipped += 1
                    continue

                messages.append({
                    "date": date,
                    "time": time,
                    "sender": sender,
                    "message": message
                })

    print(f"📄 Messaggi estratti: {len(messages)}")
    print(f"⚠️  Messaggi filtrati (di sistema): {skipped}")
    return messages


def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"✅ JSON salvato in: {output_file}")


if __name__ == "__main__":
    messages = parse_whatsapp_chat("erasmus_WA.txt")
    save_to_json(messages, "erasmus_WA.json")

