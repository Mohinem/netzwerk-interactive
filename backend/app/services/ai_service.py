from transformers import pipeline

# GPT-Neo small model for free-tier usage
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

def ai_explain_grammar(topic):
    prompt = f"Explain German grammar concept '{topic}' simply:\n"
    result = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return result.replace(prompt, "").strip()
