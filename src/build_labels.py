"""Legacy one-off pilot: hand-written labels for batches 0004 and 0005.

Superseded by the batch pipeline (`make_batches.py` -> agent labels ->
`assemble_llm_labels.py`), which validates all 200 batches. Kept for
provenance of the original pilot labels.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # project root
BATCH_DIR = BASE / "artifacts" / "label_batches"
OUT_DIR = BASE / "artifacts" / "labels_out"

LABELS = {
    "batch_0004": [
        "neutral",   # rfid ops cost explanation
        "neutral",   # averaging cost assumption, cmiiw
        "neutral",   # factual: no rice->potato swap
        "negative",  # nadiem pledge aneh/janggal, opini pressure hakim
        "positive",  # thank you + warm acceptance
        "neutral",   # fatalistic proverb
        "negative",  # "lebih parah daripada dikirim ke Detroit"
        "neutral",   # no proof, it's just a joke
        "negative",  # layoff, rupiah melemah, pertamax naik, "just **** me"
        "neutral",   # episode description
        "negative",  # savior syndrome / blind-to-conflict critique
        "neutral",   # question asking for explanation
        "negative",  # "Aku benci grup itu, admin sok banget ... admin sialan"
        "neutral",   # BBM stock observation
        "neutral",   # gaydar joke
        "neutral",   # advice/prediction
        "negative",  # calls original source a hoax
        "neutral",   # comparison of food cleanliness
        "negative",  # ancur2an logistik, desperate cannon fodder; "TLDR: don't"
        "neutral",   # procedural answer
        "neutral",   # joke about plates/detergent
        "negative",  # "Terminal internasionalnya aja busuk gitu, kasian nanti"
        "neutral",   # long cultural advice/explanation
        "negative",  # hostile argument, criticism of the other's system
        "negative",  # BI "ternodai", red flag, aneh, rupiah slide
        "neutral",   # place names + "??"
        "neutral",   # gambling strategy sharing
        "neutral",   # amused anecdote, no valence
        "negative",  # sarcastic translation accusing govt of taxing kids
        "neutral",   # "which language?" question
        "neutral",   # "moba apa"
        "neutral",   # publisher profit remark, wkwk
        "neutral",   # relationship advice
        "neutral",   # product observation
        "positive",  # "Keren sih yg bawa kendaraan ratusan km"
        "neutral",   # "coba cek di jalan sabang juga"
        "negative",  # complaint about mods
        "negative",  # mocking: libur 2 minggu krn mental health
        "neutral",   # business observation
        "neutral",   # dark joke with 😑
        "neutral",   # personal report (dbd, batuk)
        "neutral",   # speculation about survivorship bias
        "negative",  # "oknumindonistan bikin onar"
        "neutral",   # observation about tourism knowledge
        "neutral",   # question
        "negative",  # "mana ada win win" + detailed harm to Indonesia
        "neutral",   # amused teasing + KK/alamat advice
        "neutral",   # "saatnya deploy di vps dan pasang tmux"
        "neutral",   # dream anecdote
        "neutral",   # recall about show almost cancelled
    ],
    "batch_0005": [
        "neutral",   # puzzled question about "I'll fix her"
        "neutral",   # proverb-like joke
        "neutral",   # "Never beat allegations"
        "negative",  # "Tolol banget ngapain coba bikin janji"
        "neutral",   # "game apa ni"
        "neutral",   # "whats red note"
        "neutral",   # coffee routine amounts
        "neutral",   # belief in "rezeki jalur langit" story
        "neutral",   # "I'm here"
        "neutral",   # technical correction, half right
        "neutral",   # conditional statement
        "positive",  # sharing bands + "Gunkid pride hehe"
        "neutral",   # question about buying used goods
        "neutral",   # stance on forgiving a cheater
        "neutral",   # report of tiktok observation
        "negative",  # "Bawa budaya fjb lu ke fesnuk. Bukan di shopee."
        "neutral",   # road lane observation
        "neutral",   # "Kok baru sadar?"
        "negative",  # mocking "Pada ngamuk noh"
        "neutral",   # "true, wkwkwkw"
        "positive",  # happy about muffins/sarapan, nasgor solaria longing
        "positive",  # "Thank you."
        "neutral",   # pointer to facebook group
        "neutral",   # "Cakalang mana ikan cakalang"
        "positive",  # "Wih mantap bro, thanks ya"
        "neutral",   # living situation report
        "negative",  # mocking "presidennya si gemoy, org ginian pada fomo"
        "neutral",   # "Konteks apa sih?"
        "neutral",   # cost-of-living observation
        "neutral",   # Javanese phrase
        "neutral",   # analytical argument about perception
        "negative",  # "MBG = Makanan Beracun Gratis"
        "positive",  # recipe rec + "Thank me later bunda"
        "neutral",   # "maklumin aja" + gif
        "neutral",   # help request about withdrawing e-money
        "negative",  # "kayak orang autis ... emang mirip"
        "neutral",   # description of railway crossing
        "negative",  # "dokter indonesia nyarinya duid bukan ilmu"
        "positive",  # "use Wise"
        "negative",  # "Don't play with fire.."
        "neutral",   # measured political analysis
        "negative",  # "they cannot solve judol, they messed somewhere else"
        "neutral",   # railway network analysis
        "neutral",   # American visitor offering help
        "neutral",   # "gw gamau. Gw maunya Prabowo tetep presiden sampe 2029"
        "neutral",   # "Yaa kan kita smeua nimbrung disana"
        "neutral",   # long policy discussion
        "neutral",   # "Apakah OP lupa kalau Jokowi dulu selalu diledek plonga-plogo?"
        "negative",  # mocking "tas sejuta umat sejak ibu kamu"
        "neutral",   # BKP observation with shrug emoji
    ],
}

outdir = OUT_DIR
outdir.mkdir(parents=True, exist_ok=True)

for batch, labels in LABELS.items():
    src = BATCH_DIR / f"{batch}.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    if len(data) != len(labels):
        raise SystemExit("MISMATCH %s: input=%d labels=%d" % (batch, len(data), len(labels)))
    out = [{"comment_id": item["comment_id"], "label": lab} for item, lab in zip(data, labels)]
    dst = outdir / f"{batch}.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(batch, "input", len(data), "written", len(out))
