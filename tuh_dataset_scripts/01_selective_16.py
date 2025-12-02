import os, re, csv

"""
Stream headers.txt.

 - When SELECTIVE_ELECTRODES is True: write CSV rows only for files that contain ALL TARGET_ELECTRODES.
 - CSV columns: filepath, age, gender, duration, fs
"""


# CONFIGURATION 
INPUT_TXT = "../../TUH-EEG/headers.txt"
OUTPUT_CSV = "../../outputs/01_selective_16.csv"

# When True, only write files that contain ALL TARGET_ELECTRODES.
# When False, write a row for every file record (no electrode filtering).
SELECTIVE_ELECTRODES = True
TARGET_ELECTRODES = [
    "EEG C3-REF", "EEG C4-REF", "EEG F3-REF", "EEG F4-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG FP1-REF", "EEG FP2-REF", "EEG O1-REF", "EEG O2-REF",
    "EEG P3-REF", "EEG P4-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF"
]


# re
RE_EDF = re.compile(r'([A-Za-z0-9_\- ./\\]+\.(?:edf))', re.I)
RE_LPTI_AGE = re.compile(r'lpti[_\s-]*age\s*[:=]?\s*\[?\s*([^\]\r\n]+?)\s*\]?', re.I)
RE_LPTI_GENDER = re.compile(r'lpti[_\s-]*gender\s*[:=]?\s*\[?\s*([^\]\r\n]+?)\s*\]?', re.I)
RE_GENERIC_AGE = re.compile(r'Age[:=]?\s*([0-9]{1,3})', re.I)
RE_AGE_ALT = re.compile(r'([0-9]{1,3})\s*(?:y\b|yrs?\b|years?\b)', re.I)
RE_DIGITS = re.compile(r'([0-9]{1,3})')
RE_GENDER_KEYWORD = re.compile(r'\b(?:gender|sex|lpti[_\s-]*gender|patient[_\s-]*sex)\b', re.I)
RE_SINGLE_MF = re.compile(r'\b([MF])\b', re.I)
RE_BLOCK_HEADER = re.compile(r'^\s*Block\s+(\d+)\s*:', re.I)
RE_DURATION = re.compile(r'duration of recording\s*\(secs\)\s*=\s*([0-9]+(?:\.[0-9]+)?)', re.I)
RE_CHANNEL_LINE = re.compile(r'channel\[\s*\d+\s*\]:.*\(([^)]+)\)')  # captures text inside (...) after channel line
RE_CHAN_LABELS = re.compile(r'chan_labels\s*\(\s*\d+\s*\)\s*=\s*(.+)', re.I)
RE_FS = re.compile(r'hdr_sample_frequency\s*=\s*([0-9]+(?:\.[0-9]+)?)', re.I)

# helper functions
def normalize_gender(raw):
    if not raw:
        return None
    v = raw.strip().strip('[](){}:;.,\'"').lower()
    if not v:
        return None
    if v in ('m','male','man','boy'):
        return 'Male'
    if v in ('f','female','woman','girl'):
        return 'Female'
    if v[0] == 'm':
        return 'Male'
    if v[0] == 'f':
        return 'Female'
    return None

def extract_digits_from_text(s):
    if not s:
        return None
    m = RE_DIGITS.search(s)
    if m:
        return m.group(1)
    return None

def parse_line_for_age(line):
    m = RE_LPTI_AGE.search(line)
    if m:
        d = extract_digits_from_text(m.group(1))
        if d:
            return d
    m = RE_GENERIC_AGE.search(line)
    if m:
        return m.group(1)
    m = RE_AGE_ALT.search(line)
    if m:
        return m.group(1)
    return None

def parse_line_for_gender(line):
    m = RE_LPTI_GENDER.search(line)
    if m:
        return normalize_gender(m.group(1))
    if RE_GENDER_KEYWORD.search(line):
        m2 = RE_SINGLE_MF.search(line)
        if m2:
            return normalize_gender(m2.group(1))
    return None

def extract_bracket_items(s):
    """Extract items inside square brackets [ ... ] possibly repeated. Returns list of trimmed items."""
    return [it.strip() for it in re.findall(r'\[([^\]]+)\]', s)]

def flush_and_write(current, writer):
    """
    Write the CSV row for current record according to SELECTIVE_ELECTRODES flag.
    current: dict with keys 'path','age','gender','duration','fs','chan_names' (set)
    returns True if a row was written.
    """
    if not current or not current.get('path'):
        return False

    if SELECTIVE_ELECTRODES:
        present = set([c.strip() for c in current.get('chan_names', set())])
        need = set(TARGET_ELECTRODES)
        if not need.issubset(present):
            return False

    # write row (filepath, age, gender, duration, fs)
    writer.writerow([
        current.get('path'),
        current.get('age') or '',
        current.get('gender') or '',
        str(current.get('duration') or ''),
        str(current.get('fs') or '')
    ])
    return True

# main streaming processor
def process_stream(inpath, outpath):
    if not os.path.isfile(inpath):
        raise SystemExit(f"Input file not found: {inpath}")

    scanned = 0
    written = 0
    current = None
    current_block = None

    with open(inpath, 'r', encoding='utf-8', errors='replace') as fh, \
         open(outpath, 'w', newline='', encoding='utf-8') as csvf:

        writer = csv.writer(csvf)
        writer.writerow(['filepath', 'age', 'gender', 'duration', 'fs'])

        for raw_line in fh:
            line = raw_line.rstrip('\n\r')

            # detect .edf occurrences (start of new record)
            for m in RE_EDF.finditer(line):
                candidate = m.group(1).strip().strip('"\'')
                # flush previous record
                if current and current.get('path'):
                    if flush_and_write(current, writer):
                        written += 1
                        csvf.flush()
                # start new record; reset block
                current = {
                    'path': candidate,
                    'age': None,
                    'gender': None,
                    'duration': None,
                    'fs': None,
                    'chan_names': set()
                }
                current_block = None
                scanned += 1

                # parse tail of same line after match for age/gender
                tail = line[m.end():]
                if current['age'] is None:
                    a = parse_line_for_age(tail)
                    if a:
                        current['age'] = a
                if current['gender'] is None:
                    g = parse_line_for_gender(tail)
                    if g:
                        current['gender'] = g

            # if not inside a record, skip
            if not current:
                continue

            # detect block header
            bh = RE_BLOCK_HEADER.match(line)
            if bh:
                try:
                    current_block = int(bh.group(1))
                except Exception:
                    current_block = None

            # continuous age/gender parse while inside record
            if current['age'] is None:
                a = parse_line_for_age(line)
                if a:
                    current['age'] = a
            if current['gender'] is None:
                g = parse_line_for_gender(line)
                if g:
                    current['gender'] = g

            # Block 5: collect chan_labels inside square brackets
            if current_block == 5:
                mcl = RE_CHAN_LABELS.search(line)
                if mcl:
                    labels_part = mcl.group(1)
                    items = extract_bracket_items(labels_part)
                    for it in items:
                        current['chan_names'].add(it.strip())
                else:
                    # collect any bracketed items on the line
                    items = extract_bracket_items(line)
                    for it in items:
                        current['chan_names'].add(it.strip())

            # Block 6: extract duration, fs, and per-channel names
            if current_block == 6:
                md = RE_DURATION.search(line)
                if md:
                    try:
                        current['duration'] = float(md.group(1))
                    except Exception:
                        current['duration'] = md.group(1)

                mfs = RE_FS.search(line)
                if mfs:
                    try:
                        current['fs'] = float(mfs.group(1))
                    except Exception:
                        current['fs'] = mfs.group(1)

                # lines like: channel[   0]:      256.0 Hz (EEG FP1-REF)
                mc = RE_CHANNEL_LINE.search(line)
                if mc:
                    name = mc.group(1).strip()
                    current['chan_names'].add(name)

                # some lines contain parentheses with electrode labels
                for par in re.findall(r'\(([^)]+)\)', line):
                    p = par.strip()
                    if p.upper().startswith('EEG') or '-REF' in p:
                        current['chan_names'].add(p)

        # after loop, flush last record
        if current and current.get('path'):
            if flush_and_write(current, writer):
                written += 1
                csvf.flush()

    return scanned, written, os.path.abspath(outpath)


if __name__ == "__main__":
    scanned, written, outabs = process_stream(INPUT_TXT, OUTPUT_CSV)
    print(f"Done. Records started: {scanned}. Rows written: {written}. CSV: {outabs}")
