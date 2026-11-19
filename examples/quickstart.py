"""Run with: python examples/quickstart.py"""

from pathlib import Path

from keylex import KeywordList, KeywordMatcher, extract_keywords
from keylex.rules import RuleEngine

HERE = Path(__file__).parent
text = (HERE / "article.txt").read_text(encoding="utf-8")

# 1. Match a curated keyword list
keywords = KeywordList.load(HERE / "keywords.json")
matcher = KeywordMatcher(keywords, stemmer="light")
print("Matches:")
for m in matcher.find(text):
    print(f"  {m.text!r:<22} -> {m.keyword} ({m.kind}, {m.category})")
print("\nHighlighted:\n" + matcher.highlight(text))

# 2. Evaluate rules
engine = RuleEngine.load(HERE / "rules.json")
result = engine.evaluate(text)
print("\nRules fired:", result.fired, "score:", result.score)
print(result.explain())

# 3. Discover keywords without a list
print("\nExtracted keywords:")
for k in extract_keywords(text, top_n=6):
    print(f"  {k.phrase:<28} {k.score:.2f}")
