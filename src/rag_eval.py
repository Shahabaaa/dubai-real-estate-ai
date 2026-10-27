"""Measure how well the retrieval step finds the right passage.

Each test question names the document and a phrase that the correct passage contains.
A question is a hit at k if, among the top k retrieved passages, one comes from that document and contains the phrase.

Run from the project root:  python3 src/rag_eval.py
Documents are recognised by keywords in their file names (see the D_ lines below); edit them if your files are named differently.
"""
from pathlib import Path

import re

import pandas as pd

import rag

# Keywords that identify each document by its file name (spaces, underscores and capitals do not matter)
D_FEES, D_RENT, D_ESCROW = "ecr 30", "decree no", "law no 8"
D_TENANCY, D_INDEX = "33 of 2008", "smart rental"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

# (question, document, phrase that the right passage contains, or None if nothing should be found)
TESTS = [
    ("Who pays the registration fee when a property is sold?", D_FEES, "shared equally by the seller"),
    ("What is the fee for registering a sale contract?", D_FEES, "4% of the value of the sale contract"),
    ("What does it cost to register a mortgage?", D_FEES, "0.25% of the mortgage"),
    ("What is the fee to issue a title deed?", D_FEES, "issuing title deed of real property"),
    ("What is the penalty for evading Land Department fees?", D_FEES, "double the prescribed fee"),
    ("How much can rent go up if the current rent is 25% below the market average?", D_RENT, "ten percent (10%) of the rent"),
    ("Which index decides the average rental value?", D_RENT, "rent index of the emirate of dubai"),
    ("Does the rent increase decree apply in free zones?", D_RENT, "free zones such as the dubai international financial centre"),
    ("Can a developer's creditors seize money in an escrow account?", D_ESCROW, "no attachment may be imposed"),
    ("How much must the escrow agent retain after the completion certificate?", D_ESCROW, "five percent (5%)"),
    ("What happens if an off-plan project is not completed?", D_ESCROW, "refunded their payments"),
    ("Can a developer advertise off-plan units without permission?", D_ESCROW, "may not advertise"),
    ("How much notice is needed before changing the rent at renewal?", D_TENANCY, "ninety (90) days"),
    ("When can a landlord evict a tenant for not paying rent?", D_TENANCY, "within thirty (30) days from the date of service"),
    ("Must tenancy contracts be registered?", D_TENANCY, "registered with rera"),
    ("How long must a landlord wait before re-letting after taking the property back for personal use?", D_TENANCY, "two (2) years"),
    ("When was the Smart Rental Index launched?", D_INDEX, "2 january 2025"),
    ("How many rental contracts were registered in 2024?", D_INDEX, "900,000"),
    ("What is the weather in Paris?", None, None),
    ("What is the best football team in Spain?", None, None),
    ("How do I bake sourdough bread?", None, None),
    ("Who won the football world cup?", None, None),
    ("What is the capital of France?", None, None),
    ("Explain quantum computing", None, None),
]


def main():
    index = rag.Index(rag.build_chunks())
    rows = []
    for q, doc, phrase in TESTS:
        hits = index.search(q, k=5)
        if doc is None:
            ok1 = ok3 = len(hits) == 0
            rank = "-"
        else:
            rank = next((i + 1 for i, h in enumerate(hits)
                         if doc in norm(h["source"]) and phrase in " ".join(h["text"].lower().split())), None)
            ok1, ok3 = rank == 1, rank is not None and rank <= 3
        rows.append({"question": q, "expected": doc or "(nothing)", "rank_of_right_passage": rank or "not found",
                     "top1": ok1, "top3": ok3})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(df[["question", "rank_of_right_passage", "top1", "top3"]].to_string(index=False))
    real, neg = df[df["expected"] != "(nothing)"], df[df["expected"] == "(nothing)"]
    print(f"\nQuestions with an answer in the documents: {len(real)}")
    print(f"  right passage ranked first:        {real['top1'].mean():.0%}")
    print(f"  right passage within the top 3:    {real['top3'].mean():.0%}")
    print(f"Off-topic questions correctly given no answer: {neg['top1'].mean():.0%} ({len(neg)} questions)")
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/rag_eval.csv", index=False)


if __name__ == "__main__":
    main()
