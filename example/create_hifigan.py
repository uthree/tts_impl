import torch

from tts_impl.net.vocoder.hifigan import HifiganGenerator, HifiganDiscriminator
from tts_impl.net.vocoder.discriminator import MultiResolutionStftDiscriminator, CombinedDiscriminator, MultiPeriodDiscriminator

# Initialize HiFi-GAN Generator Model
generator = HifiganGenerator()

# Initialize Discriminator
discriminator = HifiganDiscriminator()

# Custom Discriminator
custom_discriminator = CombinedDiscriminator()
custom_discriminator.append(MultiPeriodDiscriminator([1, 2, 3, 5, 7, 11, 21, 27, 31]))
custom_discriminator.append(MultiResolutionStftDiscriminator([240, 120, 60]))

print(discriminator(torch.randn(1, 1, 4000)))