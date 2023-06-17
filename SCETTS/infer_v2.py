# coding: utf-8
import os
import sys
from pathlib import Path
# import g2pk.english
# import g2pk.utils
# import g2pk.g2pk
import g2pk
from g2pk import G2p
g2p = G2p()
# print(dir(g2pk))
# print(g2p)
import re
import sys
from unicodedata import normalize
# import IPython

from TTS.utils.synthesizer import Synthesizer

def remove_duplicated_punctuations(text):
    text = re.sub(r"[.?!]+\?", "?", text)
    text = re.sub(r"[.?!]+!", "!", text)
    text = re.sub(r"[.?!]+\.", ".", text)
    return text

def split_text(text):
    text = remove_duplicated_punctuations(text)

    texts = []
    for subtext in re.findall(r'[^.!?\n]*[.!?\n]', text):
        texts.append(subtext.strip())

    return texts

def alphabet_text(text):
    text = re.sub(r"(a|A)", "에이", text)
    text = re.sub(r"(b|B)", "비", text)
    text = re.sub(r"(c|C)", "씨", text)
    text = re.sub(r"(d|D)", "디", text)
    text = re.sub(r"(e|E)", "이", text)
    text = re.sub(r"(f|F)", "에프", text)
    text = re.sub(r"(g|G)", "쥐", text)
    text = re.sub(r"(h|H)", "에이치", text)
    text = re.sub(r"(i|I)", "아이", text)
    text = re.sub(r"(j|J)", "제이", text)
    text = re.sub(r"(k|K)", "케이", text)
    text = re.sub(r"(l|L)", "엘", text)
    text = re.sub(r"(m|M)", "엠", text)
    text = re.sub(r"(n|N)", "엔", text)
    text = re.sub(r"(o|O)", "오", text)
    text = re.sub(r"(p|P)", "피", text)
    text = re.sub(r"(q|Q)", "큐", text)
    text = re.sub(r"(r|R)", "알", text)
    text = re.sub(r"(s|S)", "에스", text)
    text = re.sub(r"(t|T)", "티", text)
    text = re.sub(r"(u|U)", "유", text)
    text = re.sub(r"(v|V)", "브이", text)
    text = re.sub(r"(w|W)", "더블유", text)
    text = re.sub(r"(x|X)", "엑스", text)
    text = re.sub(r"(y|Y)", "와이", text)
    text = re.sub(r"(z|Z)", "지", text)

    return text

def punctuation_text(text):
    # 문장부호
    text = re.sub(r"!", "느낌표", text)
    text = re.sub(r"\?", "물음표", text)
    text = re.sub(r"\.", "마침표", text)

    return text

def jamo_text(text):
    # 기본 자모음
    text = re.sub(r"ㄱ", "기역", text)
    text = re.sub(r"ㄴ", "니은", text)
    text = re.sub(r"ㄷ", "디귿", text)
    text = re.sub(r"ㄹ", "리을", text)
    text = re.sub(r"ㅁ", "미음", text)
    text = re.sub(r"ㅂ", "비읍", text)
    text = re.sub(r"ㅅ", "시옷", text)
    text = re.sub(r"ㅇ", "이응", text)
    text = re.sub(r"ㅈ", "지읒", text)
    text = re.sub(r"ㅊ", "치읓", text)
    text = re.sub(r"ㅋ", "키읔", text)
    text = re.sub(r"ㅌ", "티읕", text)
    text = re.sub(r"ㅍ", "피읖", text)
    text = re.sub(r"ㅎ", "히읗", text)
    text = re.sub(r"ㄲ", "쌍기역", text)
    text = re.sub(r"ㄸ", "쌍디귿", text)
    text = re.sub(r"ㅃ", "쌍비읍", text)
    text = re.sub(r"ㅆ", "쌍시옷", text)
    text = re.sub(r"ㅉ", "쌍지읒", text)
    text = re.sub(r"ㄳ", "기역시옷", text)
    text = re.sub(r"ㄵ", "니은지읒", text)
    text = re.sub(r"ㄶ", "니은히읗", text)
    text = re.sub(r"ㄺ", "리을기역", text)
    text = re.sub(r"ㄻ", "리을미음", text)
    text = re.sub(r"ㄼ", "리을비읍", text)
    text = re.sub(r"ㄽ", "리을시옷", text)
    text = re.sub(r"ㄾ", "리을티읕", text)
    text = re.sub(r"ㄿ", "리을피읍", text)
    text = re.sub(r"ㅀ", "리을히읗", text)
    text = re.sub(r"ㅄ", "비읍시옷", text)
    text = re.sub(r"ㅏ", "아", text)
    text = re.sub(r"ㅑ", "야", text)
    text = re.sub(r"ㅓ", "어", text)
    text = re.sub(r"ㅕ", "여", text)
    text = re.sub(r"ㅗ", "오", text)
    text = re.sub(r"ㅛ", "요", text)
    text = re.sub(r"ㅜ", "우", text)
    text = re.sub(r"ㅠ", "유", text)
    text = re.sub(r"ㅡ", "으", text)
    text = re.sub(r"ㅣ", "이", text)
    text = re.sub(r"ㅐ", "애", text)
    text = re.sub(r"ㅒ", "얘", text)
    text = re.sub(r"ㅔ", "에", text)
    text = re.sub(r"ㅖ", "예", text)
    text = re.sub(r"ㅘ", "와", text)
    text = re.sub(r"ㅙ", "왜", text)
    text = re.sub(r"ㅚ", "외", text)
    text = re.sub(r"ㅝ", "워", text)
    text = re.sub(r"ㅞ", "웨", text)
    text = re.sub(r"ㅟ", "위", text)
    text = re.sub(r"ㅢ", "의", text)

    return text

def normalize_multiline_text(long_text):
    texts = split_text(long_text)
    normalized_texts = [normalize_text(text).strip() for text in texts]
    return [text for text in normalized_texts if len(text) > 0]

def synthesize(text):
    wavs = synthesizer.tts(text, None, None)
    return wavs

def normalize_text(text):
    text = text.strip()

    for c in ",;:":
        text = text.replace(c, ".")
    text = remove_duplicated_punctuations(text)

    text = jamo_text(text)

    text = g2p.idioms(text)
    text = g2pk.english.convert_eng(text, g2p.cmu)
    text = g2pk.utils.annotate(text, g2p.mecab)
    text = g2pk.numerals.convert_num(text)
    text = re.sub("/[PJEB]", "", text)

    text = alphabet_text(text)

    # remove unreadable characters
    text = normalize("NFD", text)
    text = "".join(c for c in text if c in symbols)
    text = normalize("NFC", text)

    text = text.strip()
    if len(text) == 0:
        return ""

    # only single punctuation
    if text in '.!?':
        return punctuation_text(text)

    # append punctuation if there is no punctuation at the end of the text
    if text[-1] not in '.!?':
        text += '.'

    return text

synthesizer = Synthesizer(
    "C:\\intelligence\\Flask\\models\\SCETTS\\glowtts-v2\\model_file.pth.tar",
    "C:\\intelligence\\Flask\\models\\SCETTS\\glowtts-v2\\config.json",
    None,
    "C:\\intelligence\\Flask\\models\\SCETTS\\hifigan-v2\\model_file.pth.tar",
    "C:\\intelligence\\Flask\\models\\SCETTS\\hifigan-v2\\config.json",
    None,
    None,
    False,
)
symbols = synthesizer.tts_config.characters.characters


from scipy.io import wavfile
import numpy as np
import re

# output_folder = 'C:\\SumModelproject\\SCETTS\\result'

def clean_filename(filename, replace=' '):
    # 공백, 특수 문자, 한글 등을 제거하고 영어로 된 파일 이름으로 변환합니다.
    cleaned = re.sub('[^A-Za-z0-9]+', replace, filename).strip()
    return cleaned

def text_name(filename):
    cleaned_filename = re.sub('[.\/:*?"<>|!]','',filename)
    return cleaned_filename.strip()

def file_name(path,filename):
    cleaned_filename = re.sub('[.\/:*?"<>|!]','',filename)
    if not cleaned_filename.endswith('.wav'):
        cleaned_filename += '.wav'

    output_path = os.path.join(path, cleaned_filename)
    return output_path

def text_connect(texts):
    # if not isinstance(texts, str):
    #     text = str(texts)
    i=0
    output_file_list=[]
    print("texts : ", texts)
    #wavfile.write 함수는 scipy 라이브러리에 있는 함수로 이미 파일이 존재할 경우에는 해당 파일을 덮어씌워 작성합니다.
    # 이는 기본적으로 파일 쓰기를 'w' 모드로 수행하기 때문에 자동으로 해당 파일을 덮어씌우는 동작을 수행합니다.
    for text in normalize_multiline_text(texts):
        # output_file = file_name(output_folder, text)
        # print(output_file)
        print("text : ",text)
        text = text_name(text)
        wav = synthesizer.tts(text, None, None)
        wav = np.array(wav, dtype=np.float32)
        output_file = f'static/audio_data/audio_output_{i}.wav'
        wavfile.write(output_file, 22050, wav)
        output_file_list.append(output_file)
        i+=1
        # IPython.display.display(IPython.display.Audio(wav, rate=22050))
    print(output_file_list)
    return output_file_list


