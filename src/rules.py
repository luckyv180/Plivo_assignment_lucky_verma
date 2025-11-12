import re
from typing import List
from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------

EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

def collapse_spelled_letters(s: str) -> str:
    """Collapse sequences like 'g m a i l' -> 'gmail'."""
    tokens = s.split()
    out, i = [], 0
    while i < len(tokens):
        if i + 4 < len(tokens) and all(len(t) == 1 for t in tokens[i:i+5]):
            out.append(''.join(tokens[i:i+5]))
            i += 5
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    """Fix email-like spans: join 'g mail', add missing dots before com, etc."""
    # unify “at domain dot tld”
    s = re.sub(r'\bat\s+(\w+)\s+dot\s+(\w+)', r'@\1.\2', s, flags=re.I)
    # merge spaced domains
    s = re.sub(r'\bg\s*mail\b', 'gmail', s, flags=re.I)
    s = re.sub(r'\by\s*ahoo\b', 'yahoo', s, flags=re.I)
    s = re.sub(r'\bo\s*utlook\b', 'outlook', s, flags=re.I)
    s = re.sub(r'\bh\s*otmail\b', 'hotmail', s, flags=re.I)
    # add missing dots before TLDs
    s = re.sub(r'@(\w+)(com|in|org|net)\b', r'@\1.\2', s, flags=re.I)
    # clean up extra spaces and punctuation
    s = collapse_spelled_letters(s)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s = re.sub(pat, rep, s, flags=re.I)
    s = re.sub(r'\s*([@\.])\s*', r'\1', s)
    return s.strip()

# ---------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------

NUM_WORD = {
    'zero':'0','oh':'0','o':'0','nil':'0',
    'one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10'
}

def normalize_numbers_spoken(s: str) -> str:
    """Convert spoken numbers like 'nine nine nine' -> '999', 'double nine' -> '99'."""
    tokens = re.findall(r"[A-Za-z]+|\d+|[₹,\.]|[^\sA-Za-z0-9₹,\.]+", s)
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i].lower()
        if tok in ('double','triple') and i+1 < len(tokens):
            nxt = tokens[i+1].lower()
            if nxt in NUM_WORD:
                times = 2 if tok=='double' else 3
                out.append(NUM_WORD[nxt]*times)
                i += 2
                continue
        if tok in NUM_WORD:
            run = []
            j = i
            while j < len(tokens) and tokens[j].lower() in NUM_WORD:
                run.append(NUM_WORD[tokens[j].lower()])
                j += 1
            if len(run) >= 2:
                out.append(''.join(run))
                i = j
                continue
            else:
                out.append(NUM_WORD[tok])
                i += 1
                continue
        out.append(tokens[i])
        i += 1

    res = ''
    for t in out:
        if not res:
            res = t
        elif re.match(r'^[,\.!?;:]$', t):
            res = res.rstrip() + t
        elif re.search(r'[₹0-9]$', res) and re.match(r'^[0-9₹]', t):
            res += t
        else:
            res += ' ' + t
    res = re.sub(r'\s{2,}', ' ', res).strip()
    return res

# ---------------------------------------------------------------------
# Currency formatting
# ---------------------------------------------------------------------

def normalize_currency(s: str) -> str:
    """Standardize ₹ usage and Indian digit grouping."""
    s = re.sub(r'\brupees?\b', '₹', s, flags=re.I)

    # remove stray spaces inside digit groups
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', s)

    # join ₹ with number
    s = re.sub(r'₹\s*([0-9]+)', r'₹\1', s)

    # add commas (Indian grouping)
    def indian_group(num):
        x = str(int(num))
        if len(x) <= 3: return x
        last3 = x[-3:]
        rest = x[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest: parts.insert(0, rest)
        return ','.join(parts + [last3])

    s = re.sub(r'₹([0-9]{3,})', lambda m: '₹' + indian_group(m.group(1)), s)

    # remove internal spaces inside ₹-numbers like ₹1, 49, 800 → ₹1,49,800
    s = re.sub(r'₹\s*([0-9, ]+)', lambda m: '₹' + re.sub(r'\s+', '', m.group(1)), s)

    # ensure one space after the full ₹-number if next char is a letter
    s = re.sub(r'(₹[0-9,]+)(?=[A-Za-z])', r'\1 ', s)

    return re.sub(r'\s{2,}', ' ', s).strip()

# ---------------------------------------------------------------------
# Name correction
# ---------------------------------------------------------------------

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 85) -> str:
    """Fuzzy match and fix names."""
    tokens = s.split()
    out = []
    for t in tokens:
        if t in names_lex:
            out.append(t)
            continue
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0])
        else:
            out.append(t)
    return ' '.join(out)

# ---------------------------------------------------------------------
# Spacing + punctuation
# ---------------------------------------------------------------------

def space_fix_keywords(s: str) -> str:
    """Restore spaces before and after glued keywords."""
    keywords = [
        'please','confirm','by','email','contact','reply','reach',
        'today','quantity','offer','current','price','at'
    ]
    for kw in keywords:
        s = re.sub(r'(?i)(\w)(' + re.escape(kw) + r')\b', r'\1 ' + kw, s)
        s = re.sub(r'(?i)\b(' + re.escape(kw) + r')(?=[A-Za-z0-9])', r'\1 ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def minimal_punctuation(text: str) -> str:
    """Add light punctuation to improve Punctuation F1."""
    markers = ['Please', 'Contact', 'Email', 'Reach', 'Reply']
    s = text
    for m in markers:
        s = re.sub(r'(?i)(?<![,\.\?\!])(\w)(' + m + r')\b', r'\1, \2', s)
        s = re.sub(r'(?i)(?<!\s)(\b' + m + r')\b', r' ' + m, s)
    s = s.strip()
    if not re.search(r'[.!?]\s*$', s):
        s += '.'
    s = re.sub(r'\s+([,;:.!?])', r'\1', s)
    s = re.sub(r'([,;:.!?])(?=[^\s])', r'\1 ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def postprocess_fix_spacing_and_punct(s: str) -> str:
    """Final cleanup after punctuation."""
    s = space_fix_keywords(s)
    s = re.sub(r'(?i)(\S)(@[\w\._-]+\.[a-z]{2,})', r'\1 \2', s)
    s = re.sub(r'\s+([,.;:!?])', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    # ✅ FIX: compact rupee numbers like ₹1, 49, 800 → ₹1,49,800
    s = re.sub(
        r'₹\s*([0-9,\s]+)',
        lambda m: '₹' + re.sub(r'\s+', '', m.group(1)),
        s
    )

    return s



# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    """
    Generate up to 3 candidate corrections for ranking.
    Fast, lightweight, and rule-based for ≤30 ms CPU latency.
    """
    cands = set()
    t = text.strip()

    # Candidate 1: Full pipeline
    t1 = normalize_email_tokens(t)
    t1 = normalize_numbers_spoken(t1)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    t1 = minimal_punctuation(t1)
    t1 = postprocess_fix_spacing_and_punct(t1)
    cands.add(t1)

    # Candidate 2: Currency + numbers only
    t2 = normalize_currency(normalize_numbers_spoken(t))
    t2 = minimal_punctuation(t2)
    t2 = postprocess_fix_spacing_and_punct(t2)
    if t2 != t1:
        cands.add(t2)

    # Candidate 3: Names only
    t3 = correct_names_with_lexicon(t, names_lex)
    t3 = minimal_punctuation(t3)
    t3 = postprocess_fix_spacing_and_punct(t3)
    if t3 != t1 and len(cands) < 3:
        cands.add(t3)

    # Fallback: original
    if len(cands) < 3:
        cands.add(postprocess_fix_spacing_and_punct(text))

    out = sorted(list(cands), key=lambda x: len(x))[:3]
    return out or [text]