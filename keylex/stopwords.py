"""Built-in stopword lists used by the extractors (English plus compact de/fr/es/it/el lists)."""

from __future__ import annotations

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "according",
        "across",
        "actually",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "an",
        "and",
        "another",
        "any",
        "anybody",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "aren't",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "done",
        "down",
        "due",
        "during",
        "each",
        "either",
        "else",
        "elsewhere",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "for",
        "former",
        "formerly",
        "from",
        "further",
        "get",
        "gets",
        "getting",
        "give",
        "given",
        "gives",
        "go",
        "goes",
        "going",
        "gone",
        "got",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "hence",
        "her",
        "here",
        "here's",
        "hereafter",
        "hereby",
        "herein",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "however",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "ie",
        "if",
        "in",
        "indeed",
        "instead",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "latter",
        "latterly",
        "least",
        "less",
        "let",
        "let's",
        "like",
        "likely",
        "made",
        "make",
        "makes",
        "many",
        "may",
        "maybe",
        "me",
        "meanwhile",
        "might",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "much",
        "must",
        "mustn't",
        "my",
        "myself",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "ones",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "overall",
        "own",
        "per",
        "perhaps",
        "please",
        "put",
        "quite",
        "rather",
        "really",
        "said",
        "same",
        "say",
        "says",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "several",
        "shall",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "since",
        "so",
        "some",
        "somebody",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "there's",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "though",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "toward",
        "towards",
        "under",
        "unless",
        "until",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "uses",
        "using",
        "very",
        "via",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "well",
        "were",
        "weren't",
        "what",
        "what's",
        "whatever",
        "when",
        "when's",
        "whence",
        "whenever",
        "where",
        "where's",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "who's",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "why's",
        "will",
        "with",
        "within",
        "without",
        "won't",
        "would",
        "wouldn't",
        "yes",
        "yet",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

__all__ = ["ENGLISH_STOPWORDS"]

#: Base forms of common English verbs that are rarely nouns.
_VERB_BASES = """
    accept achieve add adjust agree allow appear apply argue arrive ask assume avoid become begin believe
    belong bring buy carry choose come consider consist contain continue create decide deliver depend
    describe determine develop discuss enable encourage ensure enter establish exist expect explain expose
    fail feel find follow get give go grow happen identify imply improve include indicate
    involve keep know learn leave let lose maintain manage meet obtain occur
    operate prefer prepare prevent produce propose protect prove provide publish realise realize
    receive recognise recognize reduce refer reflect rely remain remember remove replace represent require
    respond reveal say see seek seem send serve sit speak spend stay suggest suppose take tell
    tend think try understand want warn wonder write expand invest
"""

#: Base forms of verbs that are just as often nouns or adjectives ("mean squared error", "free fall",
#: "talk show"). They are not stopwords: the extractors read them as verbs only after a subject-like
#: word ("prices fall", "we need") and as ordinary content words otherwise.
_AMBIGUOUS_BASES = """
    fall hold hope lie live look mean need praise raise reach stand talk wear win wish worry wrap
"""


def _third_person(verb: str) -> str:
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in ("a", "e", "i", "o", "u"):
        return verb[:-1] + "ies"
    return verb + "s"


#: Common English verbs and verb forms that are rarely nouns: base and 3rd-person "-s" forms,
#: auxiliaries, irregular past/participle forms. A keyphrase should neither start nor end with one
#: ("cluster consists", "control plane runs"), so the English extractors treat them as stopwords.
#: Regular "-ed" forms are handled by a separate heuristic, ambiguous forms ("mean", "left") by
#: :data:`EXTRACTION_AMBIGUOUS`.
EXTRACTION_VERBS: frozenset[str] = frozenset(
    [v for base in _VERB_BASES.split() for v in (base, _third_person(base))]
    + """
    is are was were be been being am has have had having do does did done doing will would shall should
    can could may might must ought
    became began begun bought brought came chose felt gave got gotten
    goes went gone grew knew led makes meant met ran runs
    rose risen said saw seen sent sold stood took told thought won wore wrote
    slows slowed adding including following according remaining continuing becoming expecting staying
    sought caught taught dealt drew drove fought flew forgot spoke struck threw automate automates
    """.split()
)

#: Ambiguous words (participles, "number") that may start or sit inside a keyphrase but never end
#: one ("left ventricle", "number theory", but not "the patient left" or "a number").
EXTRACTION_WEAK_END: frozenset[str] = frozenset(
    """
    built chosen fallen fell found given grown held kept known left lost made paid running spent taken
    written number
    """.split()
)

#: Verb forms that may also be nouns/adjectives. The English extractors resolve them from the
#: preceding word: after a subject-like word they act as verbs (stopwords), otherwise they are
#: content words ("the mean squared error", "an enlarged left ventricle", "free fall"); words of
#: :data:`EXTRACTION_WEAK_END` still never end a phrase.
EXTRACTION_AMBIGUOUS: frozenset[str] = (
    frozenset(v for base in _AMBIGUOUS_BASES.split() for v in (base, _third_person(base))) | EXTRACTION_WEAK_END
)

#: Extra words that rarely belong in a key phrase: frequent verbs (see :data:`EXTRACTION_VERBS`),
#: weekdays, months and abbreviations. Used by the English extractors only, never by matching.
EXTRACTION_STOPWORDS: frozenset[str] = (
    ENGLISH_STOPWORDS
    | EXTRACTION_VERBS
    | frozenset(
        """
        announced reported asked added wanted last
        day days week weeks month months year years
        lot lots kind kinds way ways thing things
        monday tuesday wednesday thursday friday saturday sunday today yesterday tomorrow
        january february march april june july august september october november december
        e.g. i.e. etc. vs. cf. mr mrs ms dr
        """.split()
    )
) - EXTRACTION_AMBIGUOUS

#: Compact stopword lists for other languages (function words and a few very common verbs), selected
#: by the extractors with ``language="de"`` etc. Pass ``stopwords=`` for anything more specialised.
GERMAN_STOPWORDS: frozenset[str] = frozenset(
    """
    aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes auch
    auf aus bei beim bin bis bist da dabei damit dann das dass daß dein deine dem den denn der deren des
    dessen dich die dies diese diesem diesen dieser dieses dir doch dort du durch ein eine einem einen
    einer eines einige er es etwa etwas euch euer für gegen gewesen hab habe haben hat hatte hatten hier
    hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer im in indem ins ist ja jede jedem jeden jeder
    jedes jetzt kann kein keine können könnte machen man manche mehr mein meine mit muss musste nach
    nicht nichts noch nun nur ob oder ohne sehr sein seine seinem seinen seiner sich sie sind so solche
    soll sollte sondern sonst über um und uns unser unter viel vom von vor wann war waren warum was weil
    weiter welche welchem welchen welcher wenn wer werde werden wie wieder will wir wird wo wollen worden
    wurde wurden zu zum zur zwar zwischen sagt sagte gibt kündigte neue neuen neuer neues heute gestern
    """.split()
)
FRENCH_STOPWORDS: frozenset[str] = frozenset(
    """
    a à afin ai aie aient aies ait alors as au aucun aura aurait aussi autre aux avaient avais avait avant
    avec avez avoir avons ayant c ça car ce ceci cela celle celles celui ces cet cette ceux chaque chez ci
    comme comment d dans de des deux doit donc dont du elle elles en encore entre est et étaient était
    été être eu eux fait faire fois font ici il ils j je jusqu l la le les leur leurs lui m ma mais me
    même mes moi moins mon n ne ni nos notre nous on ont ou où par parce pas peu peut plus pour pourquoi
    qu quand que quel quelle quelles quels qui sa sans se sera serait ses si sien son sont sous sur ta te
    tes toi ton tous tout toute toutes très tu un une vers vos votre vous y dit selon
    """.split()
)
SPANISH_STOPWORDS: frozenset[str] = frozenset(
    """
    a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos e el él ella
    ellas ellos en entre era eran es esa esas ese eso esos esta está están estaba estado estar estas este
    esto estos fue fueron ha había han hasta hay la las le les lo los más me mi mis mucho muy nada ni no
    nos nosotros o os otra otras otro otros para pero poco por porque que qué quien se sea según ser si sí
    siempre sin sobre son su sus también tan tanto te tiene tienen todo todos tu tus un una uno unos y ya
    yo dijo puede pueden hace
    """.split()
)
ITALIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    a ad agli ai al alla alle allo anche avere aveva c che chi ci come con contro cui da dagli dai dal
    dalla dalle dallo degli dei del della delle dello di dove e è ed era erano essere fa gli ha hanno i il
    in io la le lei li lo loro lui ma me mi mia mio ne negli nei nel nella nelle nello noi non nostro o
    per perché più può quale quando quella quelle quello questa queste questo se sei si sia siamo sono
    sta su sua sue sui sul sulla suo tra tu tutti tutto un una uno vi voi detto stato stata
    """.split()
)
GREEK_STOPWORDS: frozenset[str] = frozenset(
    """
    ο η το οι τα του της των τον την τους τις ένα ένας μία μια ενός μιας και κι να θα δεν δε μη μην
    για με σε στο στη στην στον στα στις στους από προς κατά μετά παρά χωρίς ως όπως που πού ότι αν
    είναι ήταν έχει έχουν είχε αυτό αυτή αυτός αυτά αυτοί αυτές αυτού αυτής αυτών εγώ εσύ εμείς εσείς
    μου σου μας σας τη ή όμως αλλά επίσης ακόμα ακόμη πολύ πιο όλα όλοι όλες κάθε κάποιο κάποια κάποιος
    εδώ εκεί τότε τώρα πριν μέχρι ενώ λοιπόν δηλαδή παντού
    """.split()
)

#: Stopword lists selectable with ``language=`` on every extractor.
STOPWORDS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "en": EXTRACTION_STOPWORDS,
    "de": GERMAN_STOPWORDS,
    "fr": FRENCH_STOPWORDS,
    "es": SPANISH_STOPWORDS,
    "it": ITALIAN_STOPWORDS,
    "el": GREEK_STOPWORDS,
}


def get_stopwords(language: str = "en") -> frozenset[str]:
    """Extraction stopwords for ``language`` (ISO 639-1: ``en``, ``de``, ``fr``, ``es``, ``it``, ``el``)."""
    code = language.lower().replace("_", "-").split("-")[0] if isinstance(language, str) else ""
    if code not in STOPWORDS_BY_LANGUAGE:
        raise ValueError(
            f"No built-in stopwords for language {language!r}; choose from {sorted(STOPWORDS_BY_LANGUAGE)} "
            "or pass stopwords=[...]"
        )
    return STOPWORDS_BY_LANGUAGE[code]


__all__ += [
    "EXTRACTION_AMBIGUOUS",
    "EXTRACTION_STOPWORDS",
    "EXTRACTION_VERBS",
    "EXTRACTION_WEAK_END",
    "FRENCH_STOPWORDS",
    "GERMAN_STOPWORDS",
    "GREEK_STOPWORDS",
    "ITALIAN_STOPWORDS",
    "SPANISH_STOPWORDS",
    "STOPWORDS_BY_LANGUAGE",
    "get_stopwords",
]
