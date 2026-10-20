import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageSquare, Send, Globe, Bot, User, RefreshCw, 
  Sparkles, Check, Copy, ChevronDown, ShieldAlert, HeartPulse 
} from 'lucide-react';
import { BASE_URL } from '../utils/api';

const LANGUAGES = [
  { code: 'en', name: 'English', native: 'English', flag: '🇬🇧' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी', flag: '🇮🇳' },
  { code: 'pa', name: 'Punjabi', native: 'ਪੰਜਾਬੀ', flag: '🇮🇳' }
];

const PROMPTS_BY_LANG = {
  en: {
    welcome: "Hello! I'm OsteoCare AI, your bone health and osteoporosis assistant. Ask me anything about your T-scores, DEXA scans, knee X-rays, diet, or prevention.",
    placeholder: "Ask about T-scores, bone health, diet, exercises...",
    suggestionsTitle: "Suggested Questions:",
    suggestions: [
      "What does a T-score of -2.5 mean?",
      "Which foods are best for Calcium & Vitamin D?",
      "Can osteoporosis be reversed or managed?",
      "How do Knee X-rays help detect bone loss?"
    ],
    send: "Send",
    clear: "Clear",
    copied: "Copied!",
    disclaimer: "Medical Disclaimer: OsteoCare AI provides health information for guidance. Always consult your physician for personalized medical advice."
  },
  hi: {
    welcome: "नमस्ते! मैं OsteoCare AI हूँ — आपका हड्डियों के स्वास्थ्य और ऑस्टियोपोरोसिस सहायक। आप DEXA टी-स्कोर, एक्स-रे, खान-पान, या हड्डियों की मजबूती के बारे में कुछ भी पूछ सकते हैं।",
    placeholder: "टी-स्कोर, हड्डियों की मजबूती, आहार या सावधानियों के बारे में पूछें...",
    suggestionsTitle: "सुझाए गए प्रश्न:",
    suggestions: [
      "ऑस्टियोपोरोसिस के शुरुआती लक्षण क्या हैं?",
      "DEXA टी-स्कोर -2.5 का क्या अर्थ है?",
      "हड्डियों को मजबूत करने के लिए क्या खाएं?",
      "क्या चलने या कसरत से हड्डियां मजबूत होती हैं?"
    ],
    send: "भेजें",
    clear: "साफ़ करें",
    copied: "कॉपी हो गया!",
    disclaimer: "चिकित्सा अस्वीकरण: यह जानकारी केवल स्वास्थ्य जागरूकता के लिए है। व्यक्तिगत उपचार और निदान के लिए हमेशा अपने डॉक्टर से परामर्श लें।"
  },
  pa: {
    welcome: "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ OsteoCare AI ਹਾਂ — ਤੁਹਾਡਾ ਹੱਡੀਆਂ ਦੀ ਸਿਹਤ ਅਤੇ ਓਸਟੀਓਪੋਰੋਸਿਸ ਸਹਾਇਕ। ਤੁਸੀਂ DEXA ਸਕੈਨ, ਟੀ-ਸਕੋਰ, ਖੁਰਾਕ, ਜਾਂ ਹੱਡੀਆਂ ਦੀ ਮਜ਼ਬੂਤੀ ਬਾਰੇ ਕੋਈ ਵੀ ਸਵਾਲ ਪੁੱਛ ਸਕਦੇ ਹੋ।",
    placeholder: "ਟੀ-ਸਕੋਰ, ਹੱਡੀਆਂ ਦੀ ਮਜ਼ਬੂਤੀ, ਖੁਰਾਕ ਜਾਂ ਕਸਰਤ ਬਾਰੇ ਪੁੱਛੋ...",
    suggestionsTitle: "ਸੁਝਾਏ ਗਏ ਸਵਾਲ:",
    suggestions: [
      "ਓਸਟੀਓਪੋਰੋਸਿਸ ਦੇ ਮੁੱਖ ਲੱਛਣ ਕੀ ਹਨ?",
      "ਕੈਲਸ਼ੀਅਮ ਅਤੇ ਵਿਟਾਮਿਨ ਡੀ ਲਈ ਕੀ ਖਾਣਾ ਚਾਹੀਦਾ ਹੈ?",
      "DEXA ਟੈਸਟ ਵਿੱਚ ਟੀ-ਸਕੋਰ -2.5 ਦਾ ਕੀ ਮਤਲਬ ਹੈ?",
      "ਕੀ ਨਿਯਮਤ ਕਸਰਤ ਨਾਲ ਹੱਡੀਆਂ ਮਜ਼ਬੂਤ ਹੁੰਦੀਆਂ ਹਨ?"
    ],
    send: "ਭੇਜੋ",
    clear: "ਸਾਫ਼ ਕਰੋ",
    copied: "ਕਾਪੀ ਹੋ ਗਿਆ!",
    disclaimer: "ਮੈਡੀਕਲ ਬੇਦਾਅਵਾ: ਇਹ ਜਾਣਕਾਰੀ ਸਿਰਫ਼ ਆਮ ਸਿਹਤ ਸਿੱਖਿਆ ਲਈ ਹੈ। ਨਿੱਜੀ ਇਲਾਜ ਅਤੇ ਜਾਂਚ ਲਈ ਹਮੇਸ਼ਾ ਆਪਣੇ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਲਵੋ।"
  }
};

export default function Chatbot({ isEmbedded = true }) {
  const [language, setLanguage] = useState('en');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: PROMPTS_BY_LANG['en'].welcome,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const messagesEndRef = useRef(null);

  const t = PROMPTS_BY_LANG[language] || PROMPTS_BY_LANG['en'];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleLanguageChange = (newLang) => {
    setLanguage(newLang);
    const welcomeText = PROMPTS_BY_LANG[newLang].welcome;
    // Update or prepend initial greeting in selected language
    setMessages(prev => {
      if (prev.length <= 1) {
        return [{
          role: 'assistant',
          content: welcomeText,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }];
      }
      return prev;
    });
  };

  const handleSend = async (messageToSend = input) => {
    const text = (messageToSend || '').trim();
    if (!text || loading) return;

    const userMsg = {
      role: 'user',
      content: text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      // Build history for context (exclude welcoming assistant msg if single)
      const history = newMessages.slice(-6).map(m => ({
        role: m.role,
        content: m.content
      }));

      const res = await fetch(`${BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          language: language,
          history: history
        })
      });

      const data = await res.json();
      if (data.status === 'success' && data.reply) {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: data.reply,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            model: data.model
          }
        ]);
      } else {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: data.message || "I apologize, but I could not process your request at this moment. Please try again.",
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }
        ]);
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: "Network error connecting to AI Assistant. Please ensure the backend server is running.",
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: t.welcome,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  };

  return (
    <div className={`flex flex-col bg-white rounded-2xl border border-slate-200/90 shadow-md overflow-hidden ${isEmbedded ? 'w-full h-[620px]' : 'w-full max-w-md h-[550px]'}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 via-teal-700 to-cyan-800 text-white px-5 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/15 backdrop-blur-sm flex items-center justify-center border border-white/20 shadow-inner">
            <HeartPulse className="text-teal-200" size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-base tracking-tight">OsteoCare AI</h3>
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold bg-emerald-400/20 text-emerald-200 px-2 py-0.5 rounded-full border border-emerald-400/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-300 animate-pulse"></span>
                Groq Fast
              </span>
            </div>
            <p className="text-xs text-teal-100/90">Multilingual Clinical Bone Assistant</p>
          </div>
        </div>

        {/* Language Selector */}
        <div className="flex items-center gap-1 bg-white/10 backdrop-blur-md p-1 rounded-xl border border-white/15">
          {LANGUAGES.map((lang) => {
            const active = language === lang.code;
            return (
              <button
                key={lang.code}
                onClick={() => handleLanguageChange(lang.code)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  active 
                    ? 'bg-white text-teal-900 shadow-sm scale-105' 
                    : 'text-teal-100 hover:bg-white/10 hover:text-white'
                }`}
                title={lang.name}
              >
                <span>{lang.flag}</span>
                <span>{lang.native}</span>
              </button>
            );
          })}
          <button 
            onClick={clearChat}
            className="text-teal-200 hover:text-white hover:bg-white/10 p-1.5 rounded-lg transition-colors ml-1"
            title={t.clear}
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/70">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div 
              key={idx} 
              className={`flex items-start gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'} animate-slide-up`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-sm ${
                isUser 
                  ? 'bg-gradient-to-tr from-indigo-600 to-teal-500 text-white' 
                  : 'bg-teal-600 text-white'
              }`}>
                {isUser ? <User size={16} /> : <Bot size={16} />}
              </div>

              <div className={`relative group max-w-[82%] rounded-2xl px-4 py-3 shadow-sm text-sm leading-relaxed ${
                isUser 
                  ? 'bg-gradient-to-r from-teal-600 to-cyan-600 text-white rounded-tr-none' 
                  : 'bg-white text-slate-800 border border-slate-200/80 rounded-tl-none'
              }`}>
                <div className="whitespace-pre-wrap font-sans">
                  {msg.content}
                </div>

                <div className={`flex items-center justify-between gap-3 mt-2 text-[10px] ${
                  isUser ? 'text-teal-100/80' : 'text-slate-400'
                }`}>
                  <span>{msg.time}</span>
                  {!isUser && (
                    <button
                      onClick={() => copyToClipboard(msg.content, idx)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity hover:text-teal-600 flex items-center gap-1"
                      title="Copy message"
                    >
                      {copiedIdx === idx ? (
                        <>
                          <Check size={12} className="text-emerald-500" />
                          <span className="text-emerald-500">{t.copied}</span>
                        </>
                      ) : (
                        <Copy size={12} />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="flex items-start gap-2.5 animate-slide-up">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white flex items-center justify-center shrink-0 shadow-sm">
              <Bot size={16} />
            </div>
            <div className="bg-white border border-slate-200/80 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce"></span>
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.2s]"></span>
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.4s]"></span>
                <span className="text-xs text-slate-400 ml-2 font-medium">Generating response in {LANGUAGES.find(l=>l.code===language)?.native}...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions Pills */}
      {messages.length <= 3 && !loading && (
        <div className="px-4 py-2 bg-slate-100/80 border-t border-slate-200">
          <p className="text-[11px] font-semibold text-slate-500 mb-1.5 flex items-center gap-1">
            <Sparkles size={12} className="text-teal-600" />
            {t.suggestionsTitle}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {t.suggestions.map((q, i) => (
              <button
                key={i}
                onClick={() => handleSend(q)}
                className="text-xs bg-white hover:bg-teal-50 hover:text-teal-700 text-slate-700 px-3 py-1 rounded-full border border-slate-200 transition-all duration-150 shadow-2xs hover:border-teal-300"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="p-3 bg-white border-t border-slate-200">
        <form 
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t.placeholder}
            className="flex-1 bg-slate-50 border border-slate-200 text-slate-800 text-sm rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 transition-all placeholder:text-slate-400"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="inline-flex items-center justify-center gap-1.5 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm px-4 py-2.5 rounded-xl transition-colors shadow-sm shrink-0"
          >
            <Send size={15} />
            <span className="hidden sm:inline">{t.send}</span>
          </button>
        </form>

        {/* Disclaimer */}
        <p className="text-[10px] text-slate-400 text-center mt-2 flex items-center justify-center gap-1 leading-tight">
          <ShieldAlert size={11} className="shrink-0 text-amber-500" />
          <span>{t.disclaimer}</span>
        </p>
      </div>
    </div>
  );
}
