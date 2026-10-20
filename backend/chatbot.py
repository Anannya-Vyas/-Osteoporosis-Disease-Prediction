import os
import requests
from flask import Blueprint, request, jsonify

chatbot_bp = Blueprint('chatbot_bp', __name__)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
PRIMARY_MODEL = 'qwen/qwen3.8-27b'
FALLBACK_MODEL = 'openai/gpt-oss-120b'

SYSTEM_PROMPTS = {
    'en': (
        "You are OsteoCare AI, an empathetic, highly knowledgeable medical assistant specializing in "
        "bone health, osteoporosis prevention, DEXA bone mineral density (BMD) scans, T-scores, knee X-rays, "
        "calcium and Vitamin D nutrition, and lifestyle recommendations. "
        "Explain medical terms simply and clearly in English. "
        "Include a brief reminder that advice does not replace a doctor's personal consultation."
    ),
    'hi': (
        "आप OsteoCare AI हैं — हड्डियों के स्वास्थ्य और ऑस्टियोपोरोसिस विशेषज्ञ एक दयालु और अनुभवी AI सहायक। "
        "हड्डियों की कमजोरी, DEXA स्कैन, टी-स्कोर (T-score), कैल्शियम और विटामिन डी आहार, और व्यायाम के बारे में "
        "स्पष्ट, सरल और आदरपूर्ण हिंदी (देवनागरी लिपि) में जानकारी दें। "
        "मरीज को बताएं कि यह सामान्य जानकारी है और व्यक्तिगत इलाज के लिए हमेशा डॉक्टर से परामर्श लें।"
    ),
    'pa': (
        "ਤੁਸੀਂ OsteoCare AI ਹੋ — ਹੱਡੀਆਂ ਦੀ ਸਿਹਤ ਅਤੇ ਓਸਟੀਓਪੋਰੋਸਿਸ ਦੇ ਮਾਹਰ ਇੱਕ ਹਮਦਰਦ ਅਤੇ ਤਜ਼ਰਬੇਕਾਰ AI ਸਹਾਇਕ। "
        "ਹੱਡੀਆਂ ਦੀ ਮਜ਼ਬੂਤੀ, DEXA ਸਕੈਨ, ਟੀ-ਸਕੋਰ, ਕੈਲਸ਼ੀਅਮ, ਵਿਟਾਮਿਨ ਡੀ ਦੀ ਖੁਰਾਕ ਅਤੇ ਰੋਜ਼ਾਨਾ ਕਸਰਤ ਬਾਰੇ "
        "ਸਪੱਸ਼ਟ, ਸਰਲ ਅਤੇ ਸਤਿਕਾਰਯੋਗ ਪੰਜਾਬੀ (ਗੁਰਮੁਖੀ ਲਿਪੀ) ਵਿੱਚ ਜਾਣਕਾਰੀ ਦਿਓ। "
        "ਮਰੀਜ਼ ਨੂੰ ਯਾਦ ਦਿਵਾਓ ਕਿ ਨਿੱਜੀ ਇਲਾਜ ਲਈ ਆਪਣੇ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਜ਼ਰੂਰ ਲਵੋ।"
    )
}

def call_groq(messages, model=PRIMARY_MODEL):
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.6,
        'max_tokens': 1024
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=25)
    return resp

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get('message', '').strip()
        language = data.get('language', 'en').lower()
        history = data.get('history', [])

        if not user_message:
            return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400

        if language not in SYSTEM_PROMPTS:
            language = 'en'

        system_instruction = SYSTEM_PROMPTS[language]

        messages = [{'role': 'system', 'content': system_instruction}]

        # Include recent history (up to last 6 turns)
        for turn in history[-6:]:
            role = turn.get('role')
            content = turn.get('content')
            if role in ['user', 'assistant'] and content:
                messages.append({'role': role, 'content': content})

        messages.append({'role': 'user', 'content': user_message})

        # Try primary model first, fallback if needed
        resp = call_groq(messages, model=PRIMARY_MODEL)
        used_model = PRIMARY_MODEL

        if resp.status_code != 200:
            print(f"Primary model {PRIMARY_MODEL} returned {resp.status_code}: {resp.text}. Trying fallback {FALLBACK_MODEL}...")
            resp = call_groq(messages, model=FALLBACK_MODEL)
            used_model = FALLBACK_MODEL

        if resp.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': f'Groq API error ({resp.status_code}): {resp.text}'
            }), 502

        res_data = resp.json()
        reply = res_data['choices'][0]['message']['content']

        return jsonify({
            'status': 'success',
            'reply': reply,
            'language': language,
            'model': used_model
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
