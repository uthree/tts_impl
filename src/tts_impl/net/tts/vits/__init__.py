from tts_impl.net.tts.base import GanTextToSpeech
from tts_impl.net.vocoder.hifigan.discriminator import HifiganDiscriminator
import lightning as L

from .models import VitsGenerator


class Vits(L.LightningModule, GanTextToSpeech):
    def __init__(self):
        pass
