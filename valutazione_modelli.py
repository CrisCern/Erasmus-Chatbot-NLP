import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
nltk.download('punkt')

def evaluate_chatbot(rag_system, benchmark_csv_path):
    # Carica il benchmark dal file CSV
    benchmark_df = pd.read_csv(benchmark_csv_path)

    generated_answers = []
    references_list = []  # Lista dei token reference per il calcolo del BLEU
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    # Inizializza il calcolatore ROUGE e la funzione di smoothing per BLEU
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    for index, row in benchmark_df.iterrows():
        query = row["query"]
        ref_answer = row["reference"]

        # Chiedi al sistema RAG di generare la risposta per la query
        result = rag_system.ask_question(query)
        try:
            generated = result["llm"]['replies'][0]
        except (KeyError, IndexError):
            generated = ""

        generated_answers.append(generated)

        # Tokenizza le risposte per il calcolo del BLEU
        gen_tokens = nltk.word_tokenize(generated.lower())
        ref_tokens = nltk.word_tokenize(ref_answer.lower())
        references_list.append([ref_tokens])

        # Calcola i punteggi ROUGE per la singola query
        scores = scorer.score(ref_answer, generated)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

        print("Query:      ", query)
        print("Reference:  ", ref_answer)
        print("Generated:  ", generated)
        print("-" * 60)

    # Calcola il corpus BLEU score per l'intero benchmark
    corpus_bleu_score = corpus_bleu(
        references_list,
        [nltk.word_tokenize(ans.lower()) for ans in generated_answers],
        #smoothing_function=smoothie
    )

    # Calcola le medie dei punteggi ROUGE
    average_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    average_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    average_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

    print("==== Risultati della valutazione ====")
    print("Corpus BLEU Score: {:.4f}".format(corpus_bleu_score))
    print("Average ROUGE-1:   {:.4f}".format(average_rouge1))
    print("Average ROUGE-2:   {:.4f}".format(average_rouge2))
    print("Average ROUGE-L:   {:.4f}".format(average_rougeL))
    print("=====================================")


if __name__ == "__main__":
    from RAG_classi import WhatsAppRAG

    # Impostazione dei percorsi dei file:
    csv_path = "documenti_completi.csv"  # File con i documenti (conversazioni WhatsApp)
    benchmark_csv_path = "benchmark.csv"  # File CSV di benchmark (domande e risposte di riferimento)

    # Inizializzazione del RAG
    rag_system = WhatsAppRAG(csv_path=csv_path)
    rag_system.load_and_prepare_data()
    rag_system.create_documents()
    rag_system.build_indexing_pipeline()
    rag_system.build_retrieval_pipeline()

    # Valutazione sul benchmark
    evaluate_chatbot(rag_system, benchmark_csv_path)
