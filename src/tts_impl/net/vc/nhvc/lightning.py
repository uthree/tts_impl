from typing import Any

import torch
from lightning import LightningModule
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import HubertForCTC

from tts_impl.net.tts.vits.commons import slice_segments
from tts_impl.net.vocoder.hifigan import HifiganDiscriminator
from tts_impl.net.vocoder.hifigan.loss import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from tts_impl.transforms import LogMelSpectrogram
from tts_impl.utils.config import derive_config

from .generator import NhvcGenerator

_nhvc_discriminator_config = HifiganDiscriminator.Config()
_nhvc_discriminator_config.msd.scales = [1]
_nhvc_discriminator_config.mpd.periods = [2, 3, 5, 7, 11]
_nhvc_discriminator_config.mrsd.n_fft = [240, 400, 600]
_nhvc_discriminator_config.mrsd.hop_size = [50, 100, 200]


def rand_slice_segments(x, x_lengths=None, segment_size=8192):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def normalize(x: torch.Tensor):
    """Normalize tensor for tensorboard's image logging"""
    x = x.to(torch.float)
    mu = x.mean()
    x = x - mu
    x = x / torch.clamp_min(x.abs().max(), min=1e-8)
    return x


@derive_config
class NhvcLightningModule(LightningModule):
    def __init__(
        self,
        generator: NhvcGenerator.Config = NhvcGenerator.Config(),
        discriminator: HifiganDiscriminator.Config = _nhvc_discriminator_config,
        mel: LogMelSpectrogram.Config = LogMelSpectrogram.Config(),
        hubert_model_name: str = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4",
        n_speakers: int = 0,
        gin_channels: int = 128,
        weight_mel: float = 45.0,
        weight_feat: float = 1.0,
        weight_adv: float = 1.0,
        weight_phoneme: float = 10.0,
        weight_f0: float = 5.0,
        lr_decay: float = 0.999,
        betas: list[float] = None,
        segment_size: int = 8192,
        lr: float = 2e-4,
        metadata_path: str = "dataset_cache/metadata.json",
    ):
        if betas is None:
            betas = [0.8, 0.99]
        super().__init__()
        self.automatic_optimization = False

        # Update generator config with n_speakers and gin_channels if not already set
        if isinstance(generator, dict):
            generator_config = generator.copy()
            if "n_speakers" not in generator_config:
                generator_config["n_speakers"] = n_speakers
            if "gin_channels" not in generator_config:
                generator_config["gin_channels"] = gin_channels
            self.generator = NhvcGenerator(**generator_config)
        else:
            if not hasattr(generator, "n_speakers") or generator.n_speakers is None:
                generator.n_speakers = n_speakers
            if not hasattr(generator, "gin_channels") or generator.gin_channels is None:
                generator.gin_channels = gin_channels
            self.generator = NhvcGenerator(**generator)

        self.discriminator = HifiganDiscriminator(**discriminator)
        self.spectrogram = LogMelSpectrogram(**mel)

        # Load HuBERT model for distillation (CTC version for pseudo labels)
        self.hubert = HubertForCTC.from_pretrained(hubert_model_name)
        self.hubert.eval()  # Always in eval mode
        for param in self.hubert.parameters():
            param.requires_grad = False

        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.weight_mel = weight_mel
        self.weight_adv = weight_adv
        self.weight_feat = weight_feat
        self.weight_phoneme = weight_phoneme
        self.weight_f0 = weight_f0
        self.lr_decay = lr_decay
        self.lr = lr
        self.betas = betas
        self.segment_size = segment_size

        # Load speaker average pitch information from metadata
        self.speaker_avg_pitch_midi = None
        if metadata_path is not None:
            import json
            from pathlib import Path

            metadata_file = Path(metadata_path)
            if metadata_file.exists():
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)
                    if "speaker_avg_pitch" in metadata and "speakers" in metadata:
                        # Map speaker name to speaker ID (index)
                        speakers = metadata["speakers"]
                        speaker_avg_pitch = metadata["speaker_avg_pitch"]
                        self.speaker_avg_pitch_midi = {}
                        for idx, speaker_name in enumerate(speakers):
                            if speaker_name in speaker_avg_pitch:
                                self.speaker_avg_pitch_midi[idx] = speaker_avg_pitch[
                                    speaker_name
                                ]

        self.save_hyperparameters()

    def training_step(self, batch: dict):
        real = batch["waveform"]
        f0 = batch["f0"]
        sid = batch.get("speaker_id", None)

        # Compute mel spectrogram on-the-fly from waveform
        with torch.no_grad():
            acoustic_features = self.spectrogram(real.sum(1)).detach()
            # acoustic_features: [B, n_mels, T]

        # Generate pseudo labels from HuBERT (phoneme indices)
        with torch.no_grad():
            hubert_logits = self.hubert(real.squeeze(1), return_dict=True).logits
            # hubert_logits: [B, T, vocab_size]
            # Get pseudo labels by taking argmax
            hubert_pseudo_labels = hubert_logits.argmax(dim=-1)
            # hubert_pseudo_labels: [B, T]

        fake, slice_ids = self._adversarial_training_step(
            real, acoustic_features, f0, sid, hubert_pseudo_labels
        )
        self._discriminator_training_step(real, fake, slice_ids)

    def _generator_forward(
        self,
        x: torch.Tensor,
        sid: torch.Tensor | None,
        pitch_shift_semitones: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient separation between Encoder and Decoder.

        Encoder learns from:
        - Phoneme distillation loss (from HuBERT)
        - F0 estimation loss

        Decoder learns from:
        - Mel spectrogram reconstruction loss
        - Adversarial loss
        - Feature matching loss

        Args:
            x: Input features [B, C, T]
            sid: Speaker ID [B]
            pitch_shift_semitones: Pitch shift in semitones (MIDI scale)

        Returns:
            waveform: Generated waveform (detached from Encoder gradients)
            phoneme_logits: Phoneme classification logits for distillation loss
            f0_probs: F0 probability distribution for f0 loss
        """
        # Initialize hidden states
        h_enc = self.generator.encoder._initial_state(x.transpose(1, 2))
        h_dec = self.generator.decoder._initial_state(x.transpose(1, 2))

        # === ENCODER FORWARD (with gradients for phoneme & F0 losses) ===
        x_transposed = x.transpose(1, 2)  # [B, T, C]
        encoded, _ = self.generator.encoder._parallel_forward(x_transposed, h_enc)

        # Split encoder output
        phoneme_emb, f0_probs, noise_gate_logits = torch.split(
            encoded,
            [
                self.generator.encoder.d_phonemes,
                self.generator.encoder.n_f0_classes,
                self.generator.encoder.fft_bin,
            ],
            dim=-1,
        )

        # Get phoneme logits for distillation (keep gradients for Encoder)
        # phoneme_emb: [B, T, d_phonemes] -> [B, T, n_phonemes]
        phoneme_logits = self.generator.encoder.to_phoneme_prob(phoneme_emb)

        # Decode F0 from probabilities (keep gradients for Encoder)
        f0 = self.generator.encoder.decode_f0(f0_probs.transpose(1, 2))

        # Apply pitch shift if requested
        if pitch_shift_semitones != 0.0:
            # Shift F0 by the specified number of semitones
            # F0_shifted = F0_original * 2^(semitones/12)
            f0 = f0 * (2.0 ** (pitch_shift_semitones / 12.0))

        # Normalize phoneme embeddings
        phoneme_emb = F.instance_norm(phoneme_emb.mT).mT

        # === DETACH: Stop gradients from Decoder to Encoder ===
        phoneme_emb_detached = phoneme_emb.detach()

        # === DECODER FORWARD (isolated from Encoder gradients) ===
        # Get speaker embedding if available
        g = None
        if sid is not None and self.generator.n_speakers > 0:
            g = self.generator.speaker_embedding(sid).unsqueeze(1)

        # Decode: get vocoder parameters
        decoded, _ = self.generator.decoder._parallel_forward(
            phoneme_emb_detached, h_dec, g=g
        )

        # Split decoder output into env_per and env_noi
        decoded = decoded.transpose(1, 2)  # [B, fft_bin*2, T]
        env_per, env_noi = torch.split(decoded, self.generator.encoder.fft_bin, dim=1)
        env_per = torch.exp(env_per)
        env_noi = torch.exp(env_noi)

        # Synthesize waveform using HomomorphicVocoder
        waveform = self.generator.vocoder(f0=f0, env_per=env_per, env_noi=env_noi)
        waveform = waveform.unsqueeze(1)  # [B, 1, T*hop_length]

        return waveform, phoneme_logits, f0_probs.transpose(1, 2)

    def _adversarial_training_step(
        self,
        real: torch.Tensor,
        acoustic_features: torch.Tensor,
        f0: torch.Tensor,
        sid: torch.Tensor | None,
        hubert_pseudo_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        # Get optimizer
        opt_g, opt_d = self.optimizers()

        # Generate fake audio
        fake, phoneme_logits, f0_probs = self._generator_forward(acoustic_features, sid)

        # Compute mel spectrogram loss
        spec_real = self.spectrogram(real).detach()
        spec_fake = self.spectrogram(fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)

        # Slice waveforms for discriminator
        real_slice, slice_ids = rand_slice_segments(
            real, segment_size=self.segment_size
        )
        fake_slice = slice_segments(fake, slice_ids, segment_size=self.segment_size)

        # Adversarial and feature matching losses
        logits, fmap_fake = self.discriminator(fake_slice)
        _, fmap_real = self.discriminator(real_slice)
        loss_adv, loss_adv_list = generator_loss(logits)
        loss_feat = feature_loss(fmap_real, fmap_fake)

        # Phoneme distillation loss using pseudo labels from HuBERT
        # hubert_pseudo_labels: [B, T_hubert]
        # phoneme_logits: [B, T_phoneme, n_phonemes]

        # Downsample pseudo labels to match phoneme_logits time resolution if needed
        if hubert_pseudo_labels.shape[1] != phoneme_logits.shape[1]:
            # Use nearest neighbor interpolation for indices
            hubert_pseudo_labels = (
                F.interpolate(
                    hubert_pseudo_labels.unsqueeze(1).float(),
                    size=phoneme_logits.shape[1],
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
            # hubert_pseudo_labels: [B, T_phoneme]

        # Compute cross entropy loss
        # phoneme_logits: [B, T, n_phonemes] -> [B, n_phonemes, T] for cross_entropy
        # target: [B, T]
        loss_phoneme = F.cross_entropy(
            phoneme_logits.transpose(1, 2),  # [B, n_phonemes, T]
            hubert_pseudo_labels,  # [B, T]
            reduction="mean",
        )

        # F0 loss
        loss_f0 = self.generator.encoder.f0_loss(f0_probs, f0)

        # Total generator loss
        loss_g = (
            loss_mel * self.weight_mel
            + loss_feat * self.weight_feat
            + loss_adv * self.weight_adv
            + loss_phoneme * self.weight_phoneme
            + loss_f0 * self.weight_f0
        )

        # Update parameters
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, 1.0, "norm")
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Logs
        for i, l in enumerate(loss_adv_list):
            self.log(f"generator adversarial/{i}", l)
        self.log("train loss/generator total", loss_g)
        self.log("train loss/mel spectrogram", loss_mel)
        self.log("train loss/feature matching", loss_feat)
        self.log("train loss/generator adversarial", loss_adv)
        self.log("train loss/phoneme distillation", loss_phoneme)
        self.log("train loss/f0", loss_f0)
        self.log("G", loss_g, prog_bar=True, logger=False)

        return fake, slice_ids

    def _discriminator_training_step(
        self, real: torch.Tensor, fake: torch.Tensor, slice_ids
    ):
        opt_g, opt_d = self.optimizers()

        # Slice waveforms
        real_slice = slice_segments(real.detach(), slice_ids, self.segment_size)
        fake_slice = slice_segments(fake.detach(), slice_ids, self.segment_size)

        # Forward pass
        logits_fake, _ = self.discriminator(fake_slice)
        logits_real, _ = self.discriminator(real_slice)
        loss_d, loss_d_list_r, loss_d_list_f = discriminator_loss(
            logits_real, logits_fake
        )

        # Update parameters
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, 1.0, "norm")
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Logs
        for i, l in enumerate(loss_d_list_f):
            self.log(f"discriminator adversarial/fake {i}", l)
        for i, l in enumerate(loss_d_list_r):
            self.log(f"discriminator adversarial/real {i}", l)
        self.log("train loss/discriminator", loss_d)
        self.log("D", loss_d, prog_bar=True, logger=False)

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()
        self.log("scheduler/learning rate", sch_g.get_last_lr()[0])

    def _test_or_validate_batch(self, batch, bid):
        waveform = batch["waveform"]
        f0 = batch["f0"]
        sid = batch.get("speaker_id", None)

        # Compute mel spectrogram on-the-fly from waveform
        acoustic_features = self.spectrogram(waveform.sum(1)).detach()
        spec_real = self.spectrogram(waveform).detach()

        # Generate fake audio (reconstruction with original speaker)
        fake, _, f0_probs = self._generator_forward(acoustic_features, sid)

        spec_fake = self.spectrogram(fake)
        loss_mel = F.l1_loss(spec_fake, spec_real)
        loss_f0 = self.generator.encoder.f0_loss(f0_probs, f0)

        self.log("validation loss/mel spectrogram", loss_mel)
        self.log("validation loss/f0", loss_f0)

        # Log audio and spectrograms
        for i in range(min(4, fake.shape[0])):  # Log up to 4 samples
            f = fake[i].sum(dim=0, keepdim=True).detach().cpu()
            r = waveform[i].sum(dim=0, keepdim=True).detach().cpu()
            spec_fake_img = normalize(spec_fake[i, 0].detach().cpu().flip(0))
            spec_real_img = normalize(spec_real[i, 0].detach().cpu().flip(0))

            self.logger.experiment.add_audio(
                f"synthesized waveform/{bid}_{i}",
                f,
                self.current_epoch,
                sample_rate=self.generator.sample_rate,
            )
            self.logger.experiment.add_audio(
                f"reference waveform/{bid}_{i}",
                r,
                self.current_epoch,
                sample_rate=self.generator.sample_rate,
            )
            self.logger.experiment.add_image(
                f"synthesized mel spectrogram/{bid}_{i}",
                spec_fake_img,
                self.current_epoch,
                dataformats="HW",
            )
            self.logger.experiment.add_image(
                f"reference mel spectrogram/{bid}_{i}",
                spec_real_img,
                self.current_epoch,
                dataformats="HW",
            )

        # Voice conversion to different speakers (if multi-speaker model)
        if self.n_speakers > 0 and sid is not None:
            # Group samples by speaker ID
            speaker_samples = {}
            for i in range(waveform.shape[0]):
                spk = sid[i].item()
                if spk not in speaker_samples:
                    speaker_samples[spk] = []
                speaker_samples[spk].append(i)

            # Select up to 2 samples for conversion
            num_samples = min(2, fake.shape[0])
            for i in range(num_samples):
                source_sid = sid[i].item()

                # Convert to all other speakers (up to 3 different targets)
                available_target_sids = [
                    s for s in speaker_samples.keys() if s != source_sid
                ]
                target_sids = available_target_sids[
                    : min(3, len(available_target_sids))
                ]

                for target_sid in target_sids:
                    # Extract single sample for conversion
                    acoustic_single = acoustic_features[i : i + 1]
                    target_sid_tensor = torch.tensor(
                        [target_sid], device=acoustic_features.device
                    )

                    # Calculate pitch shift based on average pitch difference
                    pitch_shift = 0.0
                    if (
                        self.speaker_avg_pitch_midi is not None
                        and source_sid in self.speaker_avg_pitch_midi
                        and target_sid in self.speaker_avg_pitch_midi
                    ):
                        source_pitch = self.speaker_avg_pitch_midi[source_sid]
                        target_pitch = self.speaker_avg_pitch_midi[target_sid]
                        pitch_shift = target_pitch - source_pitch

                    # Convert to target speaker with pitch shift
                    converted, _, _ = self._generator_forward(
                        acoustic_single,
                        target_sid_tensor,
                        pitch_shift_semitones=pitch_shift,
                    )
                    spec_converted = self.spectrogram(converted)

                    c = converted[0].sum(dim=0, keepdim=True).detach().cpu()
                    spec_converted_img = normalize(
                        spec_converted[0, 0].detach().cpu().flip(0)
                    )

                    self.logger.experiment.add_audio(
                        f"converted waveform/{bid}_{i}_spk{source_sid}_to_spk{target_sid}",
                        c,
                        self.current_epoch,
                        sample_rate=self.generator.sample_rate,
                    )
                    self.logger.experiment.add_image(
                        f"converted mel spectrogram/{bid}_{i}_spk{source_sid}_to_spk{target_sid}",
                        spec_converted_img,
                        self.current_epoch,
                        dataformats="HW",
                    )

                    # Add target speaker's reconstruction for comparison
                    if (
                        target_sid in speaker_samples
                        and len(speaker_samples[target_sid]) > 0
                    ):
                        # Use the first available sample from target speaker
                        target_idx = speaker_samples[target_sid][0]
                        target_acoustic = acoustic_features[target_idx : target_idx + 1]
                        target_waveform = waveform[target_idx : target_idx + 1]

                        # Reconstruct target speaker's audio with their own speaker ID
                        target_reconstructed, _, _ = self._generator_forward(
                            target_acoustic,
                            torch.tensor([target_sid], device=acoustic_features.device),
                        )
                        spec_target_reconstructed = self.spectrogram(
                            target_reconstructed
                        )

                        tr = (
                            target_reconstructed[0]
                            .sum(dim=0, keepdim=True)
                            .detach()
                            .cpu()
                        )
                        tw = target_waveform[0].sum(dim=0, keepdim=True).detach().cpu()
                        spec_target_reconstructed_img = normalize(
                            spec_target_reconstructed[0, 0].detach().cpu().flip(0)
                        )
                        spec_target_real = self.spectrogram(target_waveform)
                        spec_target_real_img = normalize(
                            spec_target_real[0, 0].detach().cpu().flip(0)
                        )

                        # Log target speaker's reconstruction
                        self.logger.experiment.add_audio(
                            f"target_reconstructed waveform/{bid}_{i}_spk{target_sid}",
                            tr,
                            self.current_epoch,
                            sample_rate=self.generator.sample_rate,
                        )
                        self.logger.experiment.add_image(
                            f"target_reconstructed mel spectrogram/{bid}_{i}_spk{target_sid}",
                            spec_target_reconstructed_img,
                            self.current_epoch,
                            dataformats="HW",
                        )
                        # Log target speaker's real audio
                        self.logger.experiment.add_audio(
                            f"target_real waveform/{bid}_{i}_spk{target_sid}",
                            tw,
                            self.current_epoch,
                            sample_rate=self.generator.sample_rate,
                        )
                        self.logger.experiment.add_image(
                            f"target_real mel spectrogram/{bid}_{i}_spk{target_sid}",
                            spec_target_real_img,
                            self.current_epoch,
                            dataformats="HW",
                        )

        return loss_mel

    def validation_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

    def test_step(self, batch, id):
        return self._test_or_validate_batch(batch, id)

    def configure_optimizers(self):
        opt_g = optim.AdamW(
            self.generator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        sch_g = StepLR(opt_g, 1, self.lr_decay)
        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        sch_d = StepLR(opt_d, 1, self.lr_decay)
        return [opt_g, opt_d], [sch_g, sch_d]
