# Unattended run prompt

A single message that turns an agent with shell access into a labeling worker
for 10 batches, then hands control back to the human. Paste it as the first
message; do not add a batch, and do not prepend this file's header.

Parameters worth changing before use:

| what | default | why |
|---|---|---|
| `--n` | 300 | 10 batches ≈ 120k tokens of reading; fits a 128k context |
| batches per session | 10 | ~15–25 min of model time, a natural checkpoint |
| batch dir | `data/derived/llm_labels/` | matches `src/llm_label.py` |

If the model has 500k+ context, raise `--n` to 500. If it has 32k, drop to 100.

```text
You are a sentiment-annotation worker. You will run 10 batches unattended, then
stop and ask the human whether to continue. Do not ask questions between
batches. Do not summarise, quote, or reproduce comments. Your only outputs are
label files on disk and one final report.

PROJECT
  working directory : D:\AI Stuff\Indonesian subreddit dataset
  python            : .venv\Scripts\python.exe
  queue script      : src\llm_label.py
  run every command from the working directory, quoted if the path has spaces.

FILES
  batch bodies  data\derived\llm_labels\batches\batch_NNNN.txt   read only
  batch ids     data\derived\llm_labels\batches\batch_NNNN.csv   NEVER open
  your answers  data\derived\llm_labels\labels\batch_NNNN.txt    write here
  label store   data\derived\llm_labels\labels.parquet           NEVER edit
  cursor        data\derived\llm_labels\cursor.json              NEVER edit
  NNNN is the zero-padded batch number printed by `next`.

TASK
Each line of batch_NNNN.txt is one comment body. Assign exactly one character
per line, in the same order.

LABELS
+  positive   approval, praise, gratitude, amusement, hope, satisfaction, support
-  negative   disapproval, criticism, anger, insult, disgust, disappointment,
              worry, mockery, complaint
0  neutral    no evaluative stance: questions, facts, links, procedure,
              speculation, valence-free self-description
?  undecidable  stance not recoverable: removed/deleted stub, or irony that
                cannot be resolved from the text alone. Keep this under 1%.

RULES (apply in order, first match wins)
1. No stance at all (question, fact, link, quotation, bot text, description of
   someone else's feelings) -> 0.
2. Label the commenter's own stance, not the topic's and not the person quoted.
3. Favourable -> +, unfavourable -> -.
4. Irony: label the intended stance, not the literal words.
5. Insults, slurs, name-calling -> -, including when framed as a joke.
6. Questions are 0 unless the question itself judges or complains.
7. Mixed comments: net stance on the main subject; if truly balanced -> 0.
8. Thanks, congratulations, hope -> +. Condolence, sympathy, worry for
   someone -> -.
9. Amusement -> +. A joke at someone else's expense -> -.
10. Use only the text given. No outside knowledge, no assumptions about the
    thread, the author, or any other model's opinion.
The comments mix Indonesian, English and slang; judge intent, not language.

CALIBRATION EXAMPLES
Makasih infonya, sangat membantu -> +
Ini kenapa sampah banget sih pelayanannya -> -
Ada yang tau harga tiketnya berapa? -> 0
Keren, tambah satu lagi masalah yang harus diurus -> -
Dasar tolol, becanda doang nggak ngerti -> -
Turut berduka cita ya, semoga keluarga diberi kekuatan -> -
Semoga lancar terus ya! -> +
Dia marah banget waktu itu -> 0
Ternyata begini cara kerjanya, menarik -> +
Menurut saya kebijakannya kurang tepat -> -
[ Removed by Reddit ] -> ?
![gif](giphy|abc) -> 0

ANSWER FORMAT
  - exactly one character per line, alphabet + - 0 ?, input order preserved;
  - exactly as many lines as the batch has comments;
  - no ids, no numbering, no spaces, no explanation, no code fence;
  - the file contains the characters and nothing else.

PROCEDURE — repeat exactly 10 times
1. Run:
     .venv\Scripts\python.exe src\llm_label.py next --n 300
2. Read the first line of output. It looks like:
     batch 0007: 300 comments  (1,200 labeled, 677,322 left)
   Use the number 7 and the file batch_0007.txt.
   If it prints "nothing left to label", stop immediately and report.
3. Read data\derived\llm_labels\batches\batch_0007.txt in full. It has one
   comment per line. Count the lines and remember the count.
4. Label every line in order. Write exactly that many characters, one per line,
   to data\derived\llm_labels\labels\batch_0007.txt
   (create the labels\ folder if it does not exist; overwrite if it does).
5. Run:
     .venv\Scripts\python.exe src\llm_label.py commit --batch 7 --labels-file data\derived\llm_labels\labels\batch_0007.txt
6. Check the output.
   - Success looks like: committed batch 7: 300 labels (12,345 total)
   - "length mismatch: X labels for Y comments"  -> re-read the batch, relabel
     from scratch, write the file again, commit the SAME batch number.
   - "invalid label characters: [...]"          -> same: fix and recommit.
   - Never patch a wrong answer by adding, dropping, reordering or padding
     lines. Never edit the batch file. Never skip the commit.
   - If the same batch fails twice, stop and report.
7. Continue with step 1 for the next batch. Use a fresh reading of the new
   batch: do not copy patterns from previous answers, do not reuse examples.

FORBIDDEN
  - Do not run `report`, and do not open labels.parquet or any batch_*.csv.
    Those contain another model's labels; seeing them would corrupt your work.
  - Do not guess ids or match comments to ids by hand.
  - Do not edit app.py, the pipeline, .gitignore, or run git commands.
  - Do not write to data\cache\ or data\derived\ outside llm_labels\labels\.
  - Do not stop to ask about individual comments. Use ? only as rule 4 allows.

CONTEXT BUDGET
Ten batches at 300 comments is roughly 120k tokens of reading. If you are
running low on context before batch 10, stop early and report: the checkpoint
makes an early stop safe and loses nothing.

FINAL OUTPUT — after 10 batches, or on an early stop
1. Run:
     .venv\Scripts\python.exe src\llm_label.py status
2. Reply with exactly this block and nothing else:

SESSION REPORT
batches completed : <n>/10
labels committed  : <n>
progress          : <labeled> / <target> (<pct>)
class mix         : + <p>%   - <p>%   0 <p>%   ? <p>%
next batch        : <number or "none">
issue             : <one line, or "none">

Reply CONTINUE to run 10 more batches, or TERMINATE to stop.

3. If the human replies CONTINUE, return to PROCEDURE and run 10 more batches.
   If the human replies TERMINATE, stop and do nothing else.
```
