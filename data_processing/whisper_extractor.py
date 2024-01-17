import argparse
import os
import os.path as osp
import pickle
from tqdm import tqdm
import torch
import whisper
import whisperx
from whisperx.utils import get_writer
import warnings
warnings.filterwarnings("ignore")
os.environ['CURL_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def save_pickle(data, pickle_path):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
        
def load_srt(sub_path):
    raw_subs = pysrt.open(sub_path)
    subs = []
    for sub in raw_subs:
        raw_start = sub.start.to_time()
        start = (datetime.combine(date.min, raw_start) - datetime.min).total_seconds()
        raw_end = sub.end.to_time()
        end = (datetime.combine(date.min, raw_end) - datetime.min).total_seconds()
        text = sub.text
        subs.append([(start, end), text])
    return subs

def pickle_loader(filename):
    with open(filename, "rb") as file:
        return pickle.load(file, encoding="latin1")

def force_cudnn_initialization(device):
    s = 32
    dev = torch.device(device)
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Path to the directory with audio files",
    )
    parser.add_argument(
        "sub_file",
        type=str,
        help="sub file path",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the saving directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--whisper-size",
        type=str,
        default="tiny",
    )
    args = parser.parse_args()

    return args.audio_dir, args.sub_file, args.save_dir, args.device, args.whisper_size


if __name__ == "__main__":
    audio_dir, sub_file, save_dir, device, whisper_size = parse_arguments()
    sub_name = sub_file.split('/')[-1]
    output_file = osp.join(save_dir, sub_name)
    force_cudnn_initialization(device)

    # Load models
    transcribe_model = whisper.load_model(
        "tiny", device, download_root="./.cache/whisper"
    )
    alignment_model, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )
    record = []
    sub = pickle_loader(sub_file)
    for i in tqdm(range(len(sub))):
        # Get input and output filenames
        # audio_id = str(sub[i][1]) + '.wav'
        audio_id = str(i).zfill(5) + '.wav'
        audio_path = osp.join(audio_dir, audio_id)
        try:
            audio = whisperx.load_audio(audio_path)
            result = transcribe_model.transcribe(audio)
            sub_dic = result["segments"]
            sub_out = ''
            for j in range(len(sub_dic)):
                sub_out += sub_dic[j]['text']
            item = [sub[i][0], sub[i][1], sub[i][2], [sub_out]]
        except:
            item = [sub[i][0], sub[i][1], sub[i][2], [' ']]
        
        record.append(item)

    with open(output_file, "wb") as f:
        pickle.dump(record, f)
