class _Response:
    def __init__(self, text: str):
        self.text = text


def configure(api_key: str):
    # Stub: no-op configuration
    return None


class GenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_content(self, parts, **kwargs):
        return _Response("Descrizione immagine stub per validare l'annotazione.")
