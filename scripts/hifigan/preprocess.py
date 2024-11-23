from tts_impl.utils.preprocess import Preprocessor, AudioCacheWriter, AudioDataCollector
import fire


def run_preprocess(target_dir: str):
    preprocess = Preprocessor()
    preprocess.with_collector(AudioDataCollector(target_dir, sample_rate=22050, length=65536))
    preprocess.with_writer(AudioCacheWriter())
    preprocess.run()


if __name__ == "__main__":
    fire.Fire(run_preprocess)