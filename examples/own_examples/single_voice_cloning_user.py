from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent

import torch
import torchaudio
import base64
import os


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
        "You are an AI assistant designed to convert text into speech.\n"
        "<|scene_desc_start|>\n"
        "<|scene_desc_end|>"
    )

reference_text = "seis tres uno, seis dos, cinco tres; llámame."
reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "../voice_prompts/sara.wav")
)

messages = [
    Message(
        role='system', 
        content=[
             TextContent(text='Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.', type='text')], 
             recipient=None),
    Message(
            role="user",
            content=reference_text,
        ),
    Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
    Message(
        role="user",
        content="Hola, mi nombre es Laura Sánchez y soy ingeniera. Me encanta la montaña, pasear y comer bien.",
    )
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages, speaker=None),
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate) # type: ignore