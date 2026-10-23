"""
Sindh Smart Farmer AI - PyTorch Training Dataset & Tokenizer
Contains agricultural disease cases, agronomy guidance, and multilingual vocabularies (Urdu, Sindhi, English).
"""

import json
import torch
from torch.utils.data import Dataset
import re

# Disease classes
DISEASE_CLASSES = [
    "Cotton Leaf Curl Virus (CLCuV)",
    "Rice Blast (Pyricularia oryzae)",
    "Sugarcane Red Rot (Colletotrichum falcatum)",
    "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
    "Chilli Leaf Curl & Anthracnose",
    "Tomato Early & Late Blight",
    "Mango Blossom Malformation",
    "Healthy Crop / Good Agronomy"
]

# Intent classes
INTENT_CLASSES = [
    "diagnosis",
    "treatment",
    "prevention",
    "fertilizer",
    "irrigation",
    "general_advice"
]

# Crop classes
CROP_CLASSES = [
    "Cotton",
    "Rice",
    "Sugarcane",
    "Wheat",
    "Chilli",
    "Tomato",
    "Mango",
    "General"
]

# Multilingual Training Knowledge Corpus
TRAINING_SAMPLES = [
    # Cotton Leaf Curl Virus
    {
        "text": "cotton leaves curling upward vein thickening enations under leaf stunted growth Sanghar Matiari whitefly",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "diagnosis",
        "crop": "Cotton",
        "lang": "en"
    },
    {
        "text": "کپاس کے پتے اوپر کی طرف مڑ رہے ہیں رگیں موٹی ہو گئی ہیں سفید مکھی کا حملہ ہے قد رک گیا ہے سانگھڑ مٹیاری",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "diagnosis",
        "crop": "Cotton",
        "lang": "ur"
    },
    {
        "text": "ڪپھ جا پن مٿي چڙهي ويا آهن رڳون ٿلهيون ٿي ويون آهن اڇي مکي جو حملو آهي واڌ ويجهه رڪجي وئي آهي سانگهڙ نوابشاهه",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "diagnosis",
        "crop": "Cotton",
        "lang": "sd"
    },
    {
        "text": "how to treat cotton leaf curl virus control whitefly imidacloprid pyriproxyfen yellow sticky traps",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "treatment",
        "crop": "Cotton",
        "lang": "en"
    },
    {
        "text": "کپاس کے پتا مروڑ وائرس کا علاج سفید مکھی پر قابو پانے کے لیے امیڈاکلوپرڈ یا پیری پروکسیفن سپرے کریں",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "treatment",
        "crop": "Cotton",
        "lang": "ur"
    },
    {
        "text": "ڪپھ جي پن ورچڻ وائرس جو علاج اڇي مکي کي ختم ڪرڻ لاء پيليون ڦنديون ۽ سفارش ڪيل اسپري استعمال ڪريو",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "intent": "treatment",
        "crop": "Cotton",
        "lang": "sd"
    },

    # Rice Blast
    {
        "text": "rice paddy spindle diamond shaped lesions gray center brown border leaves drying Larkana Jacobabad blast",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "diagnosis",
        "crop": "Rice",
        "lang": "en"
    },
    {
        "text": "دھان کے پتوں پر آنکھ نما بادامی دھبے جن کا درمیان سرمئی اور کنارے بھورے ہیں لاڑکانہ شکارپور چاول جھلساؤ",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "diagnosis",
        "crop": "Rice",
        "lang": "ur"
    },
    {
        "text": "چانورن جي فصل تي اکين جهڙا نشان وچ سرمئي ۽ ڪنارا ناسي پن سڪي رهيا آهن لاڙڪاڻو ريس بلاسٽ بيماري",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "diagnosis",
        "crop": "Rice",
        "lang": "sd"
    },
    {
        "text": "treatment for rice leaf blast tricyclazole or azoxystrobin stop excess nitrogen reduce standing water humidity",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "treatment",
        "crop": "Rice",
        "lang": "en"
    },
    {
        "text": "چاول کے بلاسٹ کا علاج ٹرائی سائیکلازول سپرے کریں یوریا کھاد کا زیادہ استعمال بند کریں",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "treatment",
        "crop": "Rice",
        "lang": "ur"
    },
    {
        "text": "چانورن جي بلاسٽ بيماري لاء ٽرائي سائيڪلازول اسپري ڪيو ۽ نائٽروجن واري ڀاڻ جو گهڻو واهپو بند ڪيو",
        "disease": "Rice Blast (Pyricularia oryzae)",
        "intent": "treatment",
        "crop": "Rice",
        "lang": "sd"
    },

    # Sugarcane Red Rot
    {
        "text": "sugarcane red rot stalk internal red discoloration white patches sour alcoholic smell drying leaves Badin Thatta",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "diagnosis",
        "crop": "Sugarcane",
        "lang": "en"
    },
    {
        "text": "کماد گنے کا تنے اندر سے لال ہو گیا ہے سفید دھبے اور کھٹی شراب جیسی بو آرہی ہے پتے سوکھ رہے ہیں بدین ٹھٹہ",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "diagnosis",
        "crop": "Sugarcane",
        "lang": "ur"
    },
    {
        "text": "ڪمند جو ڳنڍ اندر مان ڳاڙهو ٿي ويو آهي کٽي ڌپ پئي اچي پن پيلا ٿي سڪي رهيا آهن بدين ۽ ٺٽي ۾ ريڊ روٽ",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "diagnosis",
        "crop": "Sugarcane",
        "lang": "sd"
    },
    {
        "text": "control sugarcane red rot uproot affected stools use disease free certified seed setts carbendazim sett dip",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "treatment",
        "crop": "Sugarcane",
        "lang": "en"
    },
    {
        "text": "گنے کے سرخ سڑاؤ کا علاج متاثرہ پودوں کو جڑ سے اکھاڑ کر جلائیں اور بیج کو کاربینڈازم دوا سے دھو کر لگائیں",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "treatment",
        "crop": "Sugarcane",
        "lang": "ur"
    },
    {
        "text": "ڪمند جي ڳاڙهي سڙڻ واري بيماري لاء بيمار ٻوٽن کي پاڙ کان پٽي ساڙيو ۽ صاف ٻج استعمال ڪيو",
        "disease": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "intent": "treatment",
        "crop": "Sugarcane",
        "lang": "sd"
    },

    # Wheat Yellow / Stripe Rust
    {
        "text": "wheat leaves showing yellow orange powdery stripes linear pustules rubbed on fingers Hyderabad Mirpurkhas rust",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "diagnosis",
        "crop": "Wheat",
        "lang": "en"
    },
    {
        "text": "گندم کے پتوں پر پیلی نارنجی دھاریاں سفوف ہاتھ لگانے سے پیلا رنگ لگ جاتا ہے زرد کنگی حیدرآباد نوابشاہ",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "diagnosis",
        "crop": "Wheat",
        "lang": "ur"
    },
    {
        "text": "ڪڻڪ جي پنن تي پيليون ڌاريون ۽ هٿ لڳائڻ سان پيلو سفوف لڳي ٿو ڪنگائي حيدرآباد ميرپورخاص گندم بيماري",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "diagnosis",
        "crop": "Wheat",
        "lang": "sd"
    },
    {
        "text": "fungicide spray for wheat yellow rust tebuconazole or propiconazole tilt spray early morning cool temperatures",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "treatment",
        "crop": "Wheat",
        "lang": "en"
    },
    {
        "text": "گندم کی زرد کنگی کا علاج ٹیبوکونازول یا ٹلٹ پروپیکونازول فنگس کش دوا کا سپرے کریں",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "treatment",
        "crop": "Wheat",
        "lang": "ur"
    },
    {
        "text": "ڪڻڪ جي زرد ڪنگائي لاء ٽيبيوڪونازول يا پروپيڪونازول ڦڦوند ڪش دوا جو فوري اسپري ڪيو",
        "disease": "Wheat Yellow/Stripe Rust (Puccinia striiformis)",
        "intent": "treatment",
        "crop": "Wheat",
        "lang": "sd"
    },

    # Chilli Leaf Curl & Anthracnose
    {
        "text": "chilli pepper leaves puckering curling upward brittle small fruits dropping rot Kunri Mirpurkhas thrips mite",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "diagnosis",
        "crop": "Chilli",
        "lang": "en"
    },
    {
        "text": "مرچ کے پتے اوپر کی طرف مڑ کر سخت ہو گئے ہیں مرچیں گر رہی ہیں کنری عمرکوٹ مرچ کی چورمر بیماری تھرپس",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "diagnosis",
        "crop": "Chilli",
        "lang": "ur"
    },
    {
        "text": "مرچن جا پن مڙي ويا آهن چُرمر بيماري ميرپورخاص ڪنري مرچن جو فصل مٽي ۽ ٿرپس جو حملو",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "diagnosis",
        "crop": "Chilli",
        "lang": "sd"
    },
    {
        "text": "remedy for chilli leaf curl control thrips with spinosad or acetamiprid for anthracnose use difenoconazole",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "treatment",
        "crop": "Chilli",
        "lang": "en"
    },
    {
        "text": "مرچ کی بیماری کا علاج تھرپس کے لیے اسپائنوساد یا ایسٹامیپرڈ سپرے کریں کوڑھی کے لیے ڈائیفینوکونازول استعمال کریں",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "treatment",
        "crop": "Chilli",
        "lang": "ur"
    },
    {
        "text": "مرچن جي چُرمر جو علاج ٿرپس جي روڪٿام لاء اسپائنوساد يا ايسيٽاميپرڊ جو اسپري ڪيو",
        "disease": "Chilli Leaf Curl & Anthracnose",
        "intent": "treatment",
        "crop": "Chilli",
        "lang": "sd"
    },

    # Tomato Blight
    {
        "text": "tomato dark concentric brown rings target spots yellow halo fruit rot lower leaves blighted Badin Sukkur",
        "disease": "Tomato Early & Late Blight",
        "intent": "diagnosis",
        "crop": "Tomato",
        "lang": "en"
    },
    {
        "text": "ٹماٹر کے نچلے پتوں پر سیاہ گول دائرے بن رہے ہیں پھل گل رہا ہے سکھر خیرپور ٹماٹر کا جھلساؤ فنگس",
        "disease": "Tomato Early & Late Blight",
        "intent": "diagnosis",
        "crop": "Tomato",
        "lang": "ur"
    },
    {
        "text": "ٽماٽي جي هيٺين پنن تي ڪارا گولا نشان داغ پيل ٿي رهيا آهن ٽماٽو سڙي رهيو آهي بدين خيرپور جھلساؤ بيماري",
        "disease": "Tomato Early & Late Blight",
        "intent": "diagnosis",
        "crop": "Tomato",
        "lang": "sd"
    },

    # Mango Malformation
    {
        "text": "mango blossom bunchy deformed flowers compact vegetative shoots no fruit set Mirpurkhas Tando Allahyar",
        "disease": "Mango Blossom Malformation",
        "intent": "diagnosis",
        "crop": "Mango",
        "lang": "en"
    },
    {
        "text": "آم کے بور کے گچھے جھاڑو نما بن رہے ہیں پھل نہیں بن رہا گچھامو بيماری میرپورخاص حیدرآباد",
        "disease": "Mango Blossom Malformation",
        "intent": "diagnosis",
        "crop": "Mango",
        "lang": "ur"
    },
    {
        "text": "انب جي ٻور جو گڇن وانگر ڦرجي وڃڻ ڦر نٿو ويهي ميرپورخاص ٽنڊو الهيار مالفارميشن بيماري",
        "disease": "Mango Blossom Malformation",
        "intent": "diagnosis",
        "crop": "Mango",
        "lang": "sd"
    },

    # Fertilizer, Irrigation & Healthy Crop Management
    {
        "text": "fertilizer schedule for wheat DAP urea potash recommended doses sowing tillering heading Sindh soil",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "fertilizer",
        "crop": "Wheat",
        "lang": "en"
    },
    {
        "text": "گندم کے لیے کھاد کا شیڈول بوائی کے وقت ایک بوری ڈی اے پی اور پہلے دوسرے پانی پر یوریا کھاد ڈالیں",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "fertilizer",
        "crop": "Wheat",
        "lang": "ur"
    },
    {
        "text": "ڪڻڪ لاء ڀاڻ جو چارٽ پوکڻ مهل هڪ ٻوري ڊي اي پي ۽ پهرين ٻئي پاڻي تي يوريا ڀاڻ استعمال ڪيو",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "fertilizer",
        "crop": "Wheat",
        "lang": "sd"
    },
    {
        "text": "cotton irrigation schedule canal water shortage saline soil management drip or bed planting Sindh",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "irrigation",
        "crop": "Cotton",
        "lang": "en"
    },
    {
        "text": "کپاس کی آبپاشی کا طریقہ نہری پانی کی کمی میں پٹڑیوں پر کاشت کریں اور کم پانی سے بہتر پیداوار لیں",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "irrigation",
        "crop": "Cotton",
        "lang": "ur"
    },
    {
        "text": "ڪپھ کي پاڻي ڏيڻ جو طريقو پٽڙين تي پوکڻ سان پاڻي جي بچت ٿيندي ۽ سنڌ ۾ سٺو فصل لٿو پوندو",
        "disease": "Healthy Crop / Good Agronomy",
        "intent": "irrigation",
        "crop": "Cotton",
        "lang": "sd"
    }
]


class SindhTokenizer:
    """Multilingual Tokenizer for English, Urdu, and Sindhi text."""
    def __init__(self, vocab=None):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        if vocab:
            self.word2idx = vocab
            self.idx2word = {v: k for k, v in vocab.items()}

    def tokenize(self, text):
        text = text.lower()
        # Clean punctuation except unicode characters for urdu/sindhi
        tokens = re.findall(r"[\w\u0600-\u06FF\u0750-\u077F]+", text)
        return tokens

    def build_vocab(self, texts, min_freq=1):
        freqs = {}
        for text in texts:
            for token in self.tokenize(text):
                freqs[token] = freqs.get(token, 0) + 1
        for token, count in freqs.items():
            if count >= min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
        return self.word2idx

    def encode(self, text, max_len=32):
        tokens = self.tokenize(text)
        idxs = [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]
        if len(idxs) < max_len:
            idxs = idxs + [self.word2idx["<PAD>"]] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        return idxs


class SindhAgronomyDataset(Dataset):
    """PyTorch Dataset for multi-task agricultural training."""
    def __init__(self, samples, tokenizer, max_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.disease_to_idx = {d: i for i, d in enumerate(DISEASE_CLASSES)}
        self.intent_to_idx = {it: i for i, it in enumerate(INTENT_CLASSES)}
        self.crop_to_idx = {c: i for i, c in enumerate(CROP_CLASSES)}
        self.lang_to_idx = {"en": 0, "ur": 1, "sd": 2}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = torch.tensor(self.tokenizer.encode(item["text"], self.max_len), dtype=torch.long)
        disease_label = torch.tensor(self.disease_to_idx.get(item["disease"], 0), dtype=torch.long)
        intent_label = torch.tensor(self.intent_to_idx.get(item["intent"], 0), dtype=torch.long)
        crop_label = torch.tensor(self.crop_to_idx.get(item["crop"], 0), dtype=torch.long)
        lang_label = torch.tensor(self.lang_to_idx.get(item.get("lang", "en"), 0), dtype=torch.long)

        return {
            "input_ids": tokens,
            "disease_label": disease_label,
            "intent_label": intent_label,
            "crop_label": crop_label,
            "lang_label": lang_label
        }
