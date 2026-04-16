import pandas as pd
import random

# Generating a synthetic hospital dispatch dataset
data = []
symptoms_critical = ["severe chest pain", "unresponsive", "gunshot wound", "cardiac arrest", "severe head trauma"]
symptoms_moderate = ["fractured arm", "deep laceration", "high fever", "breathing difficulty", "severe abdominal pain"]
symptoms_low = ["nausea", "minor scrape", "mild headache", "sprained ankle", "dehydration"]

for i in range(500):
    category = random.choice(["CRITICAL", "MODERATE", "LOW"])
    if category == "CRITICAL":
        text = f"Patient presents with {random.choice(symptoms_critical)}. Vitals unstable. ETA 5 mins."
    elif category == "MODERATE":
        text = f"Patient complains of {random.choice(symptoms_moderate)}. Vitals stable but needs urgent care. ETA 15 mins."
    else:
        text = f"Routine transport. {random.choice(symptoms_low)}. Fully conscious. ETA 25 mins."
    
    data.append({"text": text, "severity": category})

df = pd.DataFrame(data)
df.to_csv("data/mtsamples.csv", index=False)
print("✅ raw_data.csv successfully generated in the /data folder!")