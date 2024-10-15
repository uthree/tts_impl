from tts_impl.net.tts.base import GanTextToSpeech

from .models import VitsGenerator
from tts_impl.net.vocoder.hifigan.discriminator import HifiganDiscriminator


class Vits(GanTextToSpeech):
    def __init__(self):
        pass