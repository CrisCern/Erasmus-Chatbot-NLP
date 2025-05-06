import panel as pn
import param
pn.extension()

from RAG_classi import WhatsAppRAG
import warnings
warnings.filterwarnings("ignore")

csv_path = "documenti_completi.csv"
rag_system = WhatsAppRAG(csv_path=csv_path, device="cuda:0")
rag_system.load_and_prepare_data()
rag_system.create_documents()
rag_system.build_indexing_pipeline()
rag_system.build_retrieval_pipeline()

class ChatInterface(param.Parameterized):
    """
    Interfaccia chat costruita con Panel che usa il sistema RAG per rispondere alle domande.
    La lista 'panels' raccoglie, in sequenza, le righe della conversazione.
    """
    panels = param.List(default=[])
    rag = rag_system

    def convchain(self, query):
        if not query:
            return
        # Pulisce il widget di ingresso (l'ingresso globale 'inp' verrà definito sotto)
        inp.value = ""
        # Richiede al sistema RAG di rispondere alla query
        result = self.rag.ask_question(query)
        try:
            answer = result["llm"]['replies'][0]
        except (KeyError, IndexError):
            answer = "Nessuna risposta trovata."
        # Aggiunge alla cronologia la riga per l'utente
        self.panels.append(pn.Row("User:", pn.pane.Markdown(query, width=450)))
        # Aggiunge la risposta del chatbot, con uno sfondo leggermente diverso
        self.panels.append(pn.Row("ChatBot:", pn.pane.Markdown(answer, width=450, styles={'background-color': '#F6F6F6'})))
        # Restituisce un WidgetBox contenente la conversazione corrente con possibilità di scroll
        return pn.WidgetBox(*self.panels, scroll=True)

    def clr_history(self):
        """
        Pulisce la cronologia della conversazione.
        """
        self.panels = []
        return pn.WidgetBox(*self.panels)

# Crea un widget di ingresso per l'inserimento della query
inp = pn.widgets.TextInput(placeholder="Inserisci la tua domanda qui...", width=450)

# Crea un pulsante per inviare la domanda
send_button = pn.widgets.Button(name="Invia", button_type="primary")

# Crea un'istanza dell'interfaccia
chat_interface_instance = ChatInterface()

def on_click(event):
    # Quando l'utente clicca sul pulsante "Invia", la funzione convchain viene eseguita
    chat_interface_instance.convchain(inp.value)

send_button.on_click(on_click)

# Costruzione dell'interfaccia di conversazione
conversation_box = pn.Column(
    pn.Row(inp, send_button),
    pn.layout.Divider(),
    # Qui il contenuto della conversazione viene aggiornato dinamicamente
    pn.panel(lambda: pn.WidgetBox(*chat_interface_instance.panels, scroll=True), height=400),
    pn.layout.Divider(),
)

# Un semplice dashboard con titolo e tab per la conversazione
dashboard = pn.Column(
    pn.pane.Markdown("# ChatBot RAG Interface"),
    pn.Tabs(("Conversazione", conversation_box))
)

# Se esegui il file tramite un server Panel, usa .servable()
dashboard.servable()

# Per eseguire in locale (ad esempio, da terminale)
if __name__ == "__main__":
    dashboard.show()
