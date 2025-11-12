import json
import random

names = [
    "Aarav", "Priya", "Rahul", "Neha", "Vikram", "Meera", "Ankit", 
    "Divya", "Karan", "Riya", "Aditya", "Sneha", "Rohan", "Kavya",
    "Arjun", "Ishita", "Siddharth", "Pooja", "Amit", "Ritu",
    "Aman", "Nisha", "Varun", "Deepa", "Rajesh", "Sunita",
    "Gaurav", "Pallavi", "Nikhil", "Sakshi", "Tarun", "Bhavya"
]

domains = ["gmail", "yahoo", "outlook", "hotmail"]

def indian_comma(n):
    s = str(n)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ','.join(parts + [last3])

price_patterns = [
    (299, "double nine nine"),
    (499, "four nine nine"),
    (599, "five nine nine"),
    (799, "seven nine nine"),
    (899, "eight nine nine"),
    (999, "nine nine nine"),
    (1299, "₹1299"),
    (1499, "₹1499"),
    (1799, "₹1799"),
    (1999, "₹1999"),
    (2499, "₹2499"),
    (2799, "₹2799"),
    (2999, "₹2999"),
    (5000, "five thousand"),
    (9999, "nine nine nine nine"),
    (10799, "₹10799"),
    (15000, "₹15000"),
    (25000, "₹25000"),
    (49800, "₹49800"),
    (99000, "₹99000"),
    (149800, "₹149800"),
    (199000, "₹199000"),
    (250000, "₹250000"),
]

templates = [
    "offering",
    "close",
    "counter",
    "can_you"
]

def generate_sample(id_num):
    name1 = random.choice(names)
    name2 = random.choice([n for n in names if n != name1])
    domain = random.choice(domains)
    price, spoken = random.choice(price_patterns)
    template = random.choice(templates)
    
    gold_price = f"₹{indian_comma(price)}"
    
    email_variants = [
        f"{name1.lower()}{name2.lower()}@{domain}com",
        f"{name1.lower()}{name2.lower()} at {domain} dot com",
        f"{name1.lower()} {name2.lower()}@{domain}com",
        f"{name1.lower()}{name2.lower()}@{domain.replace('mail', ' mail')}com",
    ]
    email_noisy = random.choice(email_variants)
    email_gold = f"{name1.lower()}.{name2.lower()}@{domain}.com"
    
    if template == "offering":
        noisy = f"{name1.lower()} I'm offering {spoken} for this item listed at ₹{price + random.randint(100, 500)} Please confirm by email {email_noisy}"
        gold = f"{name1}, I'm offering {gold_price} for this item, listed at ₹{indian_comma(price + random.randint(100, 500))}. Please confirm by email: {email_gold}."
    
    elif template == "close":
        noisy = f"{name1.lower()} let's close at {spoken} today Contact {email_noisy}"
        gold = f"{name1}, let's close at {gold_price} today. Contact {email_gold}."
    
    elif template == "counter":
        noisy = f"Counteroffer from {name1.lower()} {spoken} Current price ₹{price + 200} Reply at {email_noisy}"
        gold = f"Counter-offer from {name1}: {gold_price}. Current price ₹{indian_comma(price + 200)}. Reply at {email_gold}."
    
    else:
        price2 = price + random.randint(200, 1000)
        noisy = f"Hi {name2.lower()} this is {name1.lower()} Can you do {spoken} instead of ₹{price2} Email me at {email_noisy}"
        gold = f"Hi {name2}, this is {name1}. Can you do {gold_price} instead of ₹{indian_comma(price2)}? Email me at {email_gold}."
    
    return (
        {"id": id_num, "text": noisy},
        {"id": id_num, "text": gold}
    )

def main():
    noisy_samples = []
    gold_samples = []
    
    for i in range(10, 80):
        noisy, gold = generate_sample(i)
        noisy_samples.append(noisy)
        gold_samples.append(gold)
    
    with open('data/noisy_transcripts.jsonl', 'a', encoding='utf-8') as f:
        for sample in noisy_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open('data/gold.jsonl', 'a', encoding='utf-8') as f:
        for sample in gold_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open('data/noisy_transcripts.jsonl', 'r') as f:
        noisy_count = len(f.readlines())
    with open('data/gold.jsonl', 'r') as f:
        gold_count = len(f.readlines())
    
    if noisy_count != 80 or gold_count != 80:
        print(f"Warning: Counts are {noisy_count} and {gold_count}")

if __name__ == "__main__":
    main()