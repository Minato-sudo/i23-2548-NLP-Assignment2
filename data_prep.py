import json
import re
import random
import os
import collections

# Topic Category Mapping
CATEGORIES = {
    "Politics": ["الیکشن", "حکومت", "وزیر", "پارلیمنٹ", "سیاست", "پارٹی", "عدالت"],
    "Sports": ["کرکٹ", "میچ", "ٹیم", "کھلاڑی", "اسکور", "ورلڈ کپ", "ٹرافی"],
    "Economy": ["مہنگائی", "تجارت", "بینک", "بجٹ", "معیشت", "روپے", "قرض"],
    "International": ["اقوام متحدہ", "معاہدہ", "خارجہ", "دو طرفہ", "تنازع", "امریکہ", "چین", "روس"],
    "Health & Society": ["ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم", "صحت", "اسکول"]
}

def classify_article(title, text):
    content = title + " " + text
    scores = collections.Counter()
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in content:
                scores[cat] += 1
    if not scores:
        return "Other"
    return scores.most_common(1)[0][0]

def load_data():
    with open("Metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    with open("cleaned.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by [id]
    articles = re.split(r'\[(\d+)\]', content)
    article_dict = {}
    for i in range(1, len(articles), 2):
        article_id = articles[i]
        text = articles[i+1].strip()
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        article_dict[article_id] = sentences
        
    return meta, article_dict

def main():
    meta, article_dict = load_data()
    
    cat_sentences = collections.defaultdict(list)
    for aid, sentences in article_dict.items():
        if aid in meta:
            title = meta[aid].get("title", "")
            full_text = " ".join(sentences)
            category = classify_article(title, full_text)
            cat_sentences[category].extend(sentences)
    
    # Select 100 sentences from 3 distinct topic categories
    selected_sentences = []
    topics = ["Politics", "Sports", "Economy", "International", "Health & Society"]
    available_topics = [t for t in topics if len(cat_sentences[t]) >= 100]
    
    if len(available_topics) < 3:
        # Fallback to available topics
        available_topics = topics
        
    chosen_topics = random.sample(available_topics, 3)
    print(f"Selected topics for Part 2: {chosen_topics}")
    
    for topic in chosen_topics:
        selected_sentences.extend(random.sample(cat_sentences[topic], 100))
    
    # Add 200 more to reach 500
    remaining = []
    for t in cat_sentences:
        remaining.extend(cat_sentences[t])
    
    # Remove duplicates from remaining
    remaining = list(set(remaining) - set(selected_sentences))
    selected_sentences.extend(random.sample(remaining, 500 - len(selected_sentences)))
    
    random.shuffle(selected_sentences)
    
    # Save the selected sentences for the next stage (tagging)
    with open("data/selected_sentences.txt", "w", encoding="utf-8") as f:
        for s in selected_sentences:
            f.write(s + "\n")
    
    print(f"Successfully selected 500 sentences and saved to data/selected_sentences.txt")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
