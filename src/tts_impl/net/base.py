from .tts.base import (AcousticFeatureEncoder, DurationDiscriminator,
                       DurationPredictor, GanTextToSpeech,
                       GanTextToSpeechGenerator, TextEncoder, TextToSpeech,
                       VariationalAcousticFeatureEncoder,
                       VariationalTextEncoder)
from .vc.base import (GanVoiceConersionGenerator, GanVoiceConversion,
                      VoiceConersionGenerator, VoiceConversion)
from .vocoder.base import (GanVocoder, GanVocoderDiscriminator,
                           GanVocoderGenerator, VocoderGenerator)
from .vocoder.discriminator import CombinedDiscriminator