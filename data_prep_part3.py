import json
import re
import collections
import random
import os

CATEGORIES = {
    "Politics": ["الیکشن", "حکومت", "وزیر", "پارلیمنٹ", "سیاست", "پارٹی", "عدالت"],
    "Sports": ["کرکٹ", "میچ", "ٹیم", "کھلاڑی", "اسکور", "ورلڈ کپ", "ٹرافی"],
    "Economy": ["مہنگائی", "تجارت", "بینک", "بجٹ", "معیشت", "روپے", "قرض"],
    "International": ["اقوام متحدہ", "معاہدہ", "خارجہ", "دو طرفہ", "تنازع", "امریکہ", "چین", "روس"],
    "Health & Society": ["ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم", "صحت", "اسکول"]
}

CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES.keys())}

def classify_article(title, text):
    content = title + " " + text
    scores = collections.Counter()
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in content:
                scores[cat] += 1
    if not scores:
        return None
    return scores.most_common(1)[0][0]

def main():
    with open("Metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    with open("cleaned.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    articles = re.split(r'\[(\d+)\]', content)
    dataset = []
    
    for i in range(1, len(articles), 2):
        article_id = articles[i]
        text = articles[i+1].strip()
        if article_id in meta:
            title = meta[article_id].get("title", "")
            cat = classify_article(title, text)
            if cat:
                dataset.append({
                    "id": article_id,
                    "text": text.replace("\n", " "),
                    "label": CAT_TO_IDX[cat]
                })
    
    random.shuffle(dataset)
    
    # Split 80/10/10
    n = len(dataset)
    train = dataset[:int(0.8*n)]
    val = dataset[int(0.8*n):int(0.9*n)]
    test = dataset[int(0.9*n):]
    
    os.makedirs("data/classification", exist_ok=True)
    with open("data/classification/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open("data/classification/val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    with open("data/classification/test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset prepared: {len(train)} train, {len(val)} val, {len(test)} test.")

if __name__ == "__main__":
    main()
