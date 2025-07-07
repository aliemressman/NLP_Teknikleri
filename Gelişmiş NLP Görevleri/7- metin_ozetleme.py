from transformers import pipeline

# ozetleme pipeline yukle
summarize = pipeline("summarization")

text = """
ali baba bir gun evden cikmis. sonra okula gidip keman kursundan geri dönerken bir kaza gecirmis.

"""

# metni ozetleme
summary = summarize(
    text,
    max_length=30,
    min_length=5,
    do_sample=True)


print(summary[0]["summary_text"])