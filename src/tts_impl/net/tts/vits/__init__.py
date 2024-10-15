from tts_impl.net.tts.base import GanTextToSpeech
from tts_impl.net.vocoder.hifigan.discriminator import HifiganDiscriminator

from .models import VitsGenerator


class Vits(GanTextToSpeech):
    def __init__(self):
        pass
