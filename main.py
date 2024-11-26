import re
import time

import gradio as gr
import pytube
import requests
import tiktoken
import torch
from deep_translator import GoogleTranslator
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import Ollama
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

languages = {
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic (Romanized)": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau (Arabic script)": "min_Arab",
    "Minangkabau (Latin script)": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei (Bengali script)": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn"
}

language_google = {
    'Akan': 'akan',
    'Amharic': 'amharic',
    'Modern Standard Arabic': 'arabic',
    'Bengali': 'bengali',
    'Bhojpuri': 'bhojpuri',
    'Bulgarian': 'bulgarian',
    'Catalan': 'catalan',
    'Chinese (Simplified)': 'chinese',
    'Croatian': 'croatian',
    'Czech': 'czech',
    'Danish': 'danish',
    'Dutch': 'dutch',
    'English': 'english',
    'Estonian': 'estonian',
    'Finnish': 'finnish',
    'French': 'french',
    'German': 'german',
    'Greek': 'greek',
    'Gujarati': 'gujarati',
    'Hausa': 'hausa',
    'Hebrew': 'hebrew',
    'Hindi': 'hindi',
    'Hungarian': 'hungarian',
    'Icelandic': 'icelandic',
    'Igbo': 'igbo',
    'Indonesian': 'indonesian',
    'Italian': 'italian',
    'Japanese': 'japanese',
    'Kannada': 'kannada',
    'Kinyarwanda': 'kinyarwanda',
    'Korean': 'korean',
    'Standard Latvian': 'latvian',
    'Lithuanian': 'lithuanian',
    'Standard Malay': 'malay',
    'Malayalam': 'malayalam',
    'Marathi': 'marathi',
    'Nepali': 'nepali',
    'Norwegian Nynorsk': 'norwegian',
    'Odia': 'odia',
    'Western Persian': 'persian',
    'Polish': 'polish',
    'Portuguese': 'portuguese',
    'Romanian': 'romanian',
    'Russian': 'russian',
    'Slovak': 'slovak',
    'Slovenian': 'slovenian',
    'Somali': 'somali',
    'Spanish': 'spanish',
    'Swahili': 'swahili',
    'Swedish': 'swedish',
    'Tamil': 'tamil',
    'Telugu': 'telugu',
    'Thai': 'thai',
    'Tigrinya': 'tigrinya',
    'Turkish': 'turkish',
    'Ukrainian': 'ukrainian',
    'Urdu': 'urdu',
    'Northern Uzbek': 'uzbek',
    'Vietnamese': 'vietnamese',
    'Welsh': 'welsh',
    'Xhosa': 'xhosa',
    'Yoruba': 'yoruba',
    'Zulu': 'zulu'
}

# global variables
overlap = 0
target_language = ""
target_lang_code = "eng_Latn"
summary_type = "long"
translation_engine = "Facebook Translator"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on " + device)

map_prompt_long = PromptTemplate.from_template(
    """
    You are given a portion of a document. Summarize the key points and main ideas from this text using clear, concise paragraphs. Focus on 
    capturing the core concepts and key discussions without adding personal opinions. Use simple language while maintaining the original meaning.

    "{text}"

    CONCISE SUMMARY:
    """
)

map_prompt_short = PromptTemplate.from_template(
    """
    You are summarizing a section of a document into a short and concise paragraph. Focus on the key points and most important information,
     and summarize the content in 2–3 sentences. Ensure the summary is clear, cohesive, and does not use lists or bullet points.

    Summarize the following text in a few brief sentences:
    "{text}"

    CONCISE SUMMARY:
    """
)
combine_prompt_long = PromptTemplate.from_template(
    """
    You are given a set of summarized texts. Combine these summaries into a comprehensive overview by synthesizing the main points. 
    Organize the ideas logically, grouping related points together. Use paragraphs to elaborate on important points, followed by 
    concise bullet points to highlight the most critical points. Avoid repetition. make sure the final summary is under 1000 words.

    "{text}"

    CONCISE SUMMARY:
    """
)
combine_prompt_short = PromptTemplate.from_template(
    """
    You have multiple short paragraph summaries of different sections of a document. Your task is to merge them into a single,
     brief paragraph that captures the core message of the entire document. Focus on the key points, and ensure the summary flows
      naturally as one cohesive paragraph without using lists or bullet points.

    Combine the following summaries into a single short paragraph:
    "{text}"

    CONCISE SUMMARY:
    """
)


# %%
def get_youtube_description(url: str):
    full_html = requests.get(url).text
    y = re.search(r'shortDescription":"', full_html)
    desc = ""
    count = y.start() + 19  # adding the length of the 'shortDescription":"
    while True:
        # get the letter at current index in text
        letter = full_html[count]
        if letter == "\"":
            if full_html[count - 1] == "\\":
                # this is case where the letter before is a backslash, meaning it is not real end of description
                desc += letter
                count += 1
            else:
                break
        else:
            desc += letter
            count += 1
    return desc


def get_youtube_info(url: str):
    yt = pytube.YouTube(url)
    title = yt.title
    if title is None:
        title = "None"
    desc = get_youtube_description(url)
    if desc is None:
        desc = "None"
    return title, desc


def get_youtube_transcript_loader_langchain(url: str):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )
    return loader.load()


def wrap_docs_to_string(docs):
    return " ".join([doc.page_content for doc in docs]).strip()


def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)


def get_youtube_transcription(url: str):
    global overlap
    text = wrap_docs_to_string(get_youtube_transcript_loader_langchain(url))
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    overlap = 0.1 * count
    return text, count


def get_transcription_summary(url: str, temperature: float, chunk_size: int, overlap_size: int):
    start_time = time.time()
    docs = get_youtube_transcript_loader_langchain(url)
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap)
    split_docs = text_splitter.split_documents(docs)
    llama_model = "llama3"
    if summary_type == "long":
        map_prompt = map_prompt_long
        combine_prompt = combine_prompt_long
        llama_model = "llama3"
    elif summary_type == "short":
        map_prompt = map_prompt_short
        combine_prompt = combine_prompt_short
        llama_model = "llama3.2"

    llm = Ollama(
        model=llama_model,
        base_url="http://localhost:11434",
        temperature=temperature
    )

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt,
                                 combine_prompt=combine_prompt,
                                 # these variables are the default values and can be modified/omitted
                                 combine_document_variable_name="text",
                                 map_reduce_document_variable_name="text")
    output = chain.invoke(split_docs)
    print(output['output_text'])
    print("Summary takes: --- %s seconds ---" % (time.time() - start_time))
    return output['output_text']


def format_text(text):
    # Split the text by '**', which indicates a new line or subtitle
    lines = text.split('**')

    formatted_lines = []
    for line in lines:
        stripped_line = line.strip()
        if "***" in stripped_line:
            break
        # Check if the line starts with bullet points or numbers
        if stripped_line.startswith(('-', '*', '1.', '2.', '3.')):
            formatted_lines.append(stripped_line)
        elif stripped_line:  # For subtitles or regular text
            formatted_lines.append("\n" + stripped_line + "\n")

    return '\n'.join(formatted_lines)


def get_translation_and_summary(urll: str, temperaturee: float, chunk_sizee: int):
    article = f"{get_transcription_summary(urll, temperaturee, chunk_sizee, 0)}"
    if target_lang_code == "eng_Latn":
        return article

    print(translation_engine)

    if translation_engine == 'Facebook Translator':
        start_time = time.time()
        # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn", device=device)
        tokenizer = NllbTokenizerFast.from_pretrained(
            "facebook/nllb-200-3.3B", src_lang="eng_Latn", device=device)
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
        inputs = tokenizer(article, return_tensors="pt")

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.encode(target_lang_code)[1], max_length=1024)

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(f"before: {result}")
        result = format_text(result)
        print(result)
        print("translation time takes:---> %s seconds ---" % (time.time() - start_time))
        return result
    else:
        start_time = time.time()
        target = language_google[target_language]
        translated_summary = GoogleTranslator(source='auto', target=target.lower()).translate(article)
        print(translated_summary)
        print("translation time takes:---> %s seconds ---" % (time.time() - start_time))
        return translated_summary


def set_target_language(target_lang):
    global target_lang_code
    global target_language
    target_language = target_lang
    target_lang_code = languages[target_lang]


def change_summary_type(type):
    global summary_type
    summary_type = type


def change_engine(engine):
    global translation_engine
    translation_engine = engine


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# YouTube Summarizer with Llama 3 """)

    with gr.Row(equal_height=True) as r0:
        with gr.Column(scale=4) as r0c1:
            url = gr.Textbox(label='YouTube URL', placeholder="Enter a youtube video link here")
        with gr.Column(scale=1) as r0c2:
            bttn_clear = gr.ClearButton(interactive=True, variant='stop')

    with gr.Row(variant='panel') as r1:
        with gr.Column(scale=2) as r1c1:
            title = gr.Textbox(label='Title', lines=2, max_lines=10, show_copy_button=True)
        with gr.Column(scale=3, ) as r1c2:
            desc = gr.Textbox(label='Description', max_lines=10, autoscroll=False, show_copy_button=True)

    with gr.Row(equal_height=True) as r2:
        bttn_info_get = gr.Button('Get Info', variant='primary')
        bttn_info_get.click(
            fn=get_youtube_info,
            inputs=url,
            outputs=[title, desc],
            api_name="get_youtube_info")
        bttn_trns_get = gr.Button("Get Transcription", variant='primary')
        bttn_summ_get = gr.Button("Summarize", variant='primary')

    with gr.Row(equal_height=True) as r3:
        tkncount = gr.Number(label='Token Count (est)')
        chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=4000)
        summary_type_dropdown = gr.Dropdown(choices=["short", "long"], label='Summary Type', value=summary_type)
        summary_type_dropdown.input(fn=change_summary_type, inputs=summary_type_dropdown)
        dropdown = gr.Dropdown(
            choices=list(languages.keys()),  # List of languages
            label="Select language",  # Label for the dropdown
            value=list(languages.keys())[list(languages.values()).index(target_lang_code)]
        )
        dropdown.change(set_target_language, inputs=dropdown)
        select_engine_dropdown = gr.Dropdown(
            choices=['Google Translator', 'Facebook Translator'],
            label='Select Engine',
            info="Use Facebook Engine for a low resourced language like Odia",
            value="Facebook Translator"
        )
        select_engine_dropdown.input(fn=change_engine, inputs=select_engine_dropdown)
    with gr.Row() as r4:
        with gr.Column() as r4c1:
            trns_raw = gr.Textbox(label='Transcript', show_copy_button=True)
        with gr.Column() as r4c2:
            trns_sum = gr.Textbox(label="Summary", show_copy_button=True)

    bttn_trns_get.click(fn=get_youtube_transcription,
                        inputs=url,
                        outputs=[trns_raw, tkncount]
                        )

    temperature = gr.Number(value=0, visible=False)
    bttn_summ_get.click(fn=get_translation_and_summary,
                        inputs=[url, temperature, chunk],
                        outputs=trns_sum)

    bttn_clear.add([url, title, desc, trns_raw, trns_sum, tkncount])

if __name__ == "__main__":
    demo.launch(share=True)
