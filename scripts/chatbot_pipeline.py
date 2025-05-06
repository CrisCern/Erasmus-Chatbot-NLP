
import pandas as pd
from datetime import timedelta
from bertopic import BERTopic
import os
from sentence_transformers import SentenceTransformer
import time
from gliner import GLiNER
from tqdm import tqdm
from tqdm.auto import tqdm
tqdm.pandas()
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

#PER RUNNARE QUESTO FILE è NECESSARIO AVERE PRIMA:
#- IL FILE erasmus_WA.json

#E PREFERIBILE AVERE PRIMA:
#- Il FILE etichette.csv

def merge_contesto(df, group_interval_minutes=60):
    """
    Raggruppa in finestre di durata `group_interval_minutes` i messaggi in df,
    unendo testo e mittenti. Restituisce un nuovo DataFrame con colonne:
      - start_time, end_time
      - partecipanti (stringa separata da ', ')
      - documento     (tutti i messaggi concatenati)
      - num_messaggi  (numero di messaggi nel gruppo)
    """
    # Assicuriamoci di non modificare l'originale
    df = df.copy()

    # Se il tuo JSON ha colonne "date" e "time", altrimenti adatta il timestamp:
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")

    # Garantiamo che il messaggio sia stringa
    df["message"] = df["message"].fillna("").astype(str)

    # Ordiniamo per timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    grouped_docs = []
    # Inizializza la finestra temporale
    current_start = df["timestamp"].iloc[0]
    current_end   = current_start + timedelta(minutes=group_interval_minutes)
    current_msgs  = []
    current_senders = set()

    for _, row in df.iterrows():
        ts = row["timestamp"]
        if ts <= current_end:
            # ancora nella stessa finestra
            current_msgs.append(row["message"])
            current_senders.add(row["sender"])
        else:
            # finestra terminata: salva il gruppo
            grouped_docs.append({
                "start_time":     current_start,
                "end_time":       current_end,
                "partecipanti":   ", ".join(sorted(current_senders)),
                "documento":      " ".join(current_msgs),
                "num_messaggi":   len(current_msgs)
            })
            # riinizializza la finestra
            current_start = ts
            current_end   = ts + timedelta(minutes=group_interval_minutes)
            current_msgs  = [row["message"]]
            current_senders = {row["sender"]}

    # salva l’ultimo gruppo
    grouped_docs.append({
        "start_time":     current_start,
        "end_time":       current_end,
        "partecipanti":   ", ".join(sorted(current_senders)),
        "documento":      " ".join(current_msgs),
        "num_messaggi":   len(current_msgs)
    })

    return pd.DataFrame(grouped_docs)


def step2_topic_modeling(df):
    grafico_html = "grafico_topic_contextualizzati.html"
    modello_salvato = "bertopic_model"
    df = df.copy()

    docs = df["documento"].astype(str).tolist()

    try:
        topic_model = BERTopic.load(modello_salvato)
        print("✅ Modello BERTopic caricato.")
    except Exception as e:
        print(f"❌ Errore durante il caricamento del modello: {e}")
        print("⚠️ Tentativo di creazione di un nuovo modello...")
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=True)
        topic_model.fit_transform(docs)
        topic_model.save(modello_salvato) # save the new model
        print(f"✅ Nuovo modello BERTopic creato e salvato in: {modello_salvato}")

    topics, probs = topic_model.fit_transform(docs)
    df["topic"] = topics
    fig = topic_model.visualize_barchart(top_n_topics=10)
    fig.write_html(grafico_html)
    print(f"📊 Grafico HTML salvato: {grafico_html}")
    return df



def step3_etichetta_o_unisci(df):
    file_etichettatura = "etichette.csv"
    df = df.copy()
    if os.path.exists(file_etichettatura):
        print("🔁 File etichettato già esistente. Verrà usato per unione automatica.")
        df_labels = pd.read_csv(file_etichettatura)
        df_topic_label = pd.merge(df, df_labels[["topic", "topic_label"]], on="topic", how="left")
        return df_topic_label
    else:
        print("📝 Nessun file etichettato trovato. Inizio etichettatura manuale...")
        df = df.copy()
        docs = df["documento"].astype(str).tolist()

        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=True)
        topic_model.fit_transform(docs)

        topic_info = topic_model.get_topic_info()
        keywords_per_topic = topic_model.get_topics()

        etichettatura = []
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue
            keywords = ", ".join([word for word, _ in keywords_per_topic[topic_id]])
            print(f"\n[Topic {topic_id}] Parole chiave: {keywords}")
            label = input("Etichetta assegnata (es. 'Documenti Erasmus', 'Non rilevante', ecc.): ")
            etichettatura.append({
                "topic": topic_id,
                "keywords": keywords,
                "topic_label": label
            })

        df_etichette = pd.DataFrame(etichettatura)
        df_etichette.to_csv(file_etichettatura, index=False)
        print(f"\n✅ Etichette salvate in: {file_etichettatura}")

        df_topic_label = pd.merge(df, df_etichette[["topic", "topic_label"]], on="topic", how="left")
        return df_topic_label



def step4_NER(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica l'analisi NER_messaggi_singoli al DataFrame df (colonna 'documento')
    e restituisce un nuovo DataFrame con colonna aggiuntiva 'entities',
    lista di dizionari {'text': ..., 'label': ...}.
    """
    # Evitiamo di modificare l'originale
    df_ner = df.copy()
    # Inizializziamo la colonna
    df_ner['entities'] = None

    # Carica il modello Universal NER_messaggi_singoli in italiano una sola volta
    model = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")

    # Definizione delle etichette personalizzate e dei dizionari di mapping
    ETICHETTE_PERSONALIZZATE = [
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

    FALSI_POSITIVI = {"informazioni", "rischi", "scegliere", "bisogno", "tranquilla", "cioè", "ahhh"}

    DOCUMENTI_ERASMUS = {"application form", "LA", "transcript of records", "grant agreement"}

    LINGUE = {"inglese", "spagnolo", "francese", "tedesco", "italiano", "portoghese", "olandese", "svedese"}
    LIVELLI_LINGUISTICI = {"B1", "B2", "C1", "C2"}

    for idx, row in tqdm(df_ner.iterrows(), total=len(df_ner), desc="NER_messaggi_singoli su DataFrame"):
        text = str(row.get('documento', ''))
        # Predizione entità dal modello
        raw_entities = model.predict_entities(text, ETICHETTE_PERSONALIZZATE)

        labels = []
        for ent in raw_entities:
            t = ent['text']
            label = ent['label']
            # Filtri e rimappature
            if t in FALSI_POSITIVI:
                continue
            if t in DOCUMENTI_ERASMUS:
                label = 'documento Erasmus'
            if t in LINGUE:
                label = 'lingua'
            elif t in LIVELLI_LINGUISTICI:
                label = 'livello linguistico'
            labels.append({'text': t, 'label': label})

        unique_labels = []
        for label_dict in labels:
            if label_dict not in unique_labels:
                unique_labels.append(label_dict)

        df_ner.at[idx, 'entities'] = unique_labels  # Assigning the unique list

        # Piccola pausa per gestire rate limits o CPU
        time.sleep(0.001)

    return df_ner


def step5_clean_ner(df: pd.DataFrame):
    """
    Pulisce la colonna 'entità' di df lasciando solo il campo 'testo' per ciascuna entità,
    elimina eventuali duplicati e salva il DataFrame risultante in output_csv.

    Parametri
    ---------
    df : pd.DataFrame
        DataFrame contenente almeno la colonna 'entità' con liste di dict {'testo', 'tipo'}.
    output_csv : str
        Percorso del file CSV in cui salvare il risultato.
    """
    # Non toccare l'originale
    df_clean = df.copy()

    def extract_texts(ner_list):
        if not isinstance(ner_list, list):
            return []
        # Prende solo il campo 'testo' e deduplica
        texts = [ent.get("label") for ent in ner_list if isinstance(ent, dict) and "label" in ent]
        # Usa dict.fromkeys per mantenere l’ordine originale
        return list(dict.fromkeys(texts))

    # Applichiamo la pulizia
    df_clean["entitità"] = df_clean["entities"].apply(extract_texts)

    # Salviamo il risultato
    return df_clean



def step6_sentiment_emotion_analysis(df):
    df = df.copy()

    df = df[df["documento"].str.strip().astype(bool)].reset_index(drop=True) #rimozione di stringhe vuote


    # SENTIMENT

    model_name = "MilaNLProc/feel-it-italian-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Creazione pipeline
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model=model,
                                  tokenizer=tokenizer,
                                  batch_size= 25,
                                  truncation = True,
                                  max_length=512)

    df["sentiment"] = df["documento"].progress_apply(lambda x: sentiment_pipeline(x)[0]["label"])
    df["sentiment_score"] = df["documento"].progress_apply(lambda x: sentiment_pipeline(x)[0]["score"])

    # EMOTIONS

    emotion_pipeline = pipeline(
        "text-classification",
        model="MilaNLProc/feel-it-italian-emotion",
        top_k=1  # restituisce solo l'emozione dominante
    )


    # Applica il modello a ciascun documento
    df[["emotion", "emotion_score"]] = df["documento"].progress_apply(
        lambda x: pd.Series({
            "emotion": emotion_pipeline(str(x)[:512])[0][0]['label'],
            "emotion_score": emotion_pipeline(str(x)[:512])[0][0]["score"]
        })
    )

    return df


if __name__ == "__main__":
    df = pd.read_json("../erasmus_WA.json")

    df_grouped = merge_contesto(df, group_interval_minutes=60)
    df_topic = step2_topic_modeling(df_grouped)
    df_topic_label = step3_etichetta_o_unisci(df_topic)
    df_NER = step4_NER(df_topic_label)
    df_clean = step5_clean_ner(df_NER)
    df_complete = step6_sentiment_emotion_analysis(df_clean)
    df_complete.to_csv("documenti_completi.csv", index=False)
    print("DATI SALVATI IN FILE CSV")


