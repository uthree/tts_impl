import lightning as L

from tts_impl.net.tts.base import GanTextToSpeech

from .models import VitsGenerator


class Vits(L.LightningModule, GanTextToSpeech):
    def __init__(self):
        pass


__all__ = ["VitsGenerator", "Vits"]
