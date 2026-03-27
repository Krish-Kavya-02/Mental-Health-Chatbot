from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import chromadb
import time
import re
from typing import Optional, List, Any

# ─── Serious topic keywords that require a professional disclaimer ───
SERIOUS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "self harm",
    "cutting", "hurt myself", "giving up", "no reason to live", "want to die",
    "hopeless", "worthless", "depressed", "depression", "panic attack", "panic attacks",
    "trauma", "ptsd", "abuse", "abused", "eating disorder", "anorexia", "bulimia",
    "addiction", "substance", "overdose", "hallucination", "psychosis",
    "bipolar", "schizophrenia", "ocd", "dissociation",
]

DISCLAIMER_NOTE = (
    "\n\n---\n"
    "**Important:** MindMate is a supportive companion, not a substitute for professional help. "
    "If you're experiencing persistent or severe symptoms, please consult a licensed mental health "
    "professional or reach out to a crisis helpline:\n"
    "- **988 Suicide & Crisis Lifeline** — Call or text 988 (24/7)\n"
    "- **Crisis Text Line** — Text HOME to 741741\n"
    "- **NAMI Helpline** — 1-800-950-NAMI (6264)"
)


def needs_disclaimer(text: str) -> bool:
    """Check if the query or response involves a serious mental health topic."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SERIOUS_KEYWORDS)


def extract_answer(doc_content: str) -> str:
    """Extract just the answer portion from a Q&A document."""
    match = re.search(r'A:\s*(.+)', doc_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return doc_content.strip()


def extract_question(doc_content: str) -> str:
    """Extract just the question portion from a Q&A document."""
    match = re.search(r'Q:\s*(.+?)(?:\nA:)', doc_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return doc_content.strip()


def create_rag_pipeline():
    print("Loading embedding model...")
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    from datasets import load_dataset
    
    chroma_client = chromadb.EphemeralClient()
    vector_store = Chroma(
        client=chroma_client,
        collection_name="mental_health_qa",
        embedding_function=huggingface_embeddings
    )
    
    print("Loading mental health data...")
    docs = []
    
    # Professional built-in Q&A pairs
    builtin_docs = [
        # Greetings
        ("Hi", "Hello, welcome to MindMate. I'm here to listen and provide support on topics related to mental well-being. How are you feeling today?"),
        ("Hello", "Hi there, I'm glad you reached out. I'm here to help you explore what's on your mind — whether it's stress, anxiety, or anything else. How can I support you today?"),
        ("Hey", "Hello! I'm here and ready to listen. Feel free to share whatever is on your mind — there's no judgment here."),
        ("How are you", "Thank you for asking. I'm here and ready to support you. More importantly — how are you doing? Is there something you'd like to talk about?"),
        ("Good morning", "Good morning. I hope your day is off to a good start. I'm here if there's anything you'd like to talk about."),
        ("Good evening", "Good evening. I'm here whenever you need someone to talk to. How has your day been?"),
        ("Thank you", "You're welcome. Remember, reaching out for support is a positive step. I'm here whenever you need to talk."),
        ("Thanks", "You're welcome. I'm always here if you want to continue the conversation. Take care."),
        ("Bye", "Take care of yourself. Remember, I'm here whenever you need support. Wishing you well."),
        ("Goodbye", "Goodbye for now. I'm always available whenever you'd like to talk. Take care."),
        
        # Anxiety & Nervousness
        ("I feel nervous", "Nervousness is a natural response to uncertainty, and it's completely valid. Here are a few techniques that may help:\n\n- **Box breathing** — Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4. Repeat.\n- **Grounding (5-4-3-2-1)** — Identify 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.\n- **Name it** — Sometimes simply identifying the source of nervousness can reduce its intensity.\n\nWould you like to share what's making you feel this way?"),
        ("I'm feeling anxious", "Anxiety is something many people experience, and there are effective ways to manage it:\n\n- **Controlled breathing** — Slow, deep breaths help activate your body's relaxation response.\n- **Cognitive reframing** — Ask yourself: Is this worry based on evidence, or on assumption?\n- **Start small** — If a task feels overwhelming, break it into the smallest possible step.\n- **Movement** — Even a short walk can help regulate anxiety.\n\nWhat's been triggering your anxiety? I'm here to listen."),
        ("I feel sad", "Sadness is a natural part of the human experience, and it's important to acknowledge it rather than suppress it. A few things that may help:\n\n- **Allow the feeling** — Give yourself permission to feel sad without judgment.\n- **Connect with someone** — Talking to a trusted person can provide comfort.\n- **Engage in small activities** — Listening to music, journaling, or going for a walk can help shift your mood.\n- **Self-compassion** — Treat yourself with the same kindness you'd offer a close friend.\n\nWould you like to talk about what's on your mind?"),
        ("I feel depressed", "I hear you, and I appreciate you sharing that. Depression can feel overwhelming, but it's important to know that support is available.\n\n- **You're not alone** — Depression is one of the most common mental health conditions.\n- **Small steps count** — Even basic self-care like eating a meal or stepping outside is meaningful progress.\n- **Professional support** — A licensed therapist or counselor can provide personalized guidance and treatment.\n- **Be patient** — Recovery isn't linear, and that's okay.\n\nWould you like to tell me more about what you've been experiencing?"),
        ("I can't sleep", "Sleep difficulties are common and often linked to stress or an overactive mind. Here are some evidence-based strategies:\n\n- **Consistent schedule** — Go to bed and wake up at the same time each day.\n- **Wind-down routine** — Reduce screen exposure 30–60 minutes before bed.\n- **Relaxation techniques** — Progressive muscle relaxation or guided breathing exercises.\n- **Worry journal** — Write down concerns before bed to help clear your mind.\n- **Limit stimulants** — Avoid caffeine after early afternoon.\n\nHow long has sleep been a challenge for you?"),
        ("I feel stressed", "Stress is a natural response, but chronic stress can impact your well-being. Here are some approaches to manage it:\n\n- **Identify the source** — Understanding what's driving your stress makes it more manageable.\n- **Prioritize** — Distinguish between what's urgent and what can wait.\n- **Schedule breaks** — Regular short breaks help prevent burnout.\n- **Physical activity** — Exercise is one of the most effective stress management tools.\n- **Set boundaries** — It's okay to say no when you're stretched too thin.\n\nWhat's been the main source of stress for you lately?"),
        ("I feel lonely", "Loneliness can be deeply painful, and I want you to know that reaching out here is a positive step. Some things that may help:\n\n- **Reconnect** — Even a brief message to someone you haven't spoken to in a while can help.\n- **Community involvement** — Joining a group, class, or volunteer activity can foster connection.\n- **Online communities** — Support groups and forums can provide meaningful interaction.\n- **Self-reflection** — Explore the difference between being alone and feeling lonely.\n\nWould you like to explore what might be contributing to these feelings?"),
        ("I'm angry", "Anger is a valid emotion — what matters is how we process and express it. Here are some healthy approaches:\n\n- **Pause before reacting** — Take a few deep breaths or count to 10.\n- **Physical outlet** — Exercise, walking, or stretching can help release tension.\n- **Journaling** — Writing about what triggered your anger can provide clarity.\n- **Identify patterns** — Understanding your triggers helps you respond rather than react.\n- **'I' statements** — When ready, express your feelings with statements like 'I feel frustrated when...'\n\nWhat's been causing your frustration?"),
        ("I feel overwhelmed", "Feeling overwhelmed usually signals that you're carrying more than feels manageable. Here's how to start regaining a sense of control:\n\n- **Brain dump** — Write everything on your mind down on paper.\n- **Focus narrowly** — Choose just one or two priorities for today.\n- **Delegate** — Identify tasks you can ask for help with.\n- **Decompose** — Break large tasks into small, concrete steps.\n- **Pause** — Take 5 slow breaths right now. It helps more than you'd think.\n\nWhat's weighing on you the most?"),
        ("I have panic attacks", "Panic attacks can be very frightening, but understanding them can help. During a panic attack:\n\n- **Remind yourself** — Panic attacks are temporary and not dangerous. They typically peak within 10 minutes.\n- **Grounding (5-4-3-2-1)** — Engage your senses to anchor yourself in the present.\n- **Slow breathing** — Inhale for 4 counts, hold 4, exhale for 6.\n- **Don't fight it** — Resistance often intensifies the experience. Let it pass.\n- **Professional help** — If panic attacks are recurring, a mental health professional can help with evidence-based treatments.\n\nHow frequently do you experience these?"),
        ("I feel like giving up", "I hear you, and what you're feeling right now matters. Please know that help is available:\n\n- **You don't have to face this alone** — Reach out to someone you trust, or contact a crisis helpline.\n- **988 Lifeline** — Call or text 988 at any time, 24/7.\n- **Crisis Text Line** — Text HOME to 741741.\n- **One moment at a time** — You don't need to solve everything right now.\n\nYour feelings are valid, and so is your need for support."),
        ("I need help", "I'm here for you. Whatever you're going through, talking about it is an important first step.\n\n- **Share what's on your mind** — I'm here to listen without judgment.\n- **Coping strategies** — I can suggest evidence-based techniques for managing stress, anxiety, and other challenges.\n- **Professional resources** — I can help guide you toward appropriate professional support.\n\nWhat would be most helpful to talk about right now?"),
        ("I'm having relationship problems", "Relationship challenges can be emotionally draining. Here are some constructive approaches:\n\n- **Communication** — Express your feelings using 'I' statements rather than assigning blame.\n- **Active listening** — Ensure both sides feel genuinely heard.\n- **Boundaries** — Healthy relationships require clear, respected boundaries.\n- **Space when needed** — It's okay to take time to process your feelings.\n- **Professional guidance** — A couples counselor can provide a structured, safe space for difficult conversations.\n\nWould you like to share more about the situation?"),
        ("I don't feel motivated", "A lack of motivation is common and can stem from various factors. Here are some ways to work through it:\n\n- **Start with the minimum** — Commit to just 2 minutes of a task. Often, starting is the hardest part.\n- **Release the pressure** — Productivity doesn't define your worth.\n- **Change your setting** — A new environment can help reset your focus.\n- **Acknowledge progress** — Recognize even small accomplishments.\n- **Check fundamentals** — Sleep, nutrition, and hydration directly affect motivation.\n\nIs there something specific you've been struggling to start?"),
        ("I'm dealing with grief", "Grief is one of the most difficult experiences, and there is no right or wrong way to process it.\n\n- **No timeline** — Grief doesn't follow a schedule. Give yourself permission to feel at your own pace.\n- **All emotions are valid** — Sadness, anger, guilt, and even moments of peace are all part of the process.\n- **Honor your loss** — Share memories, write, or create something meaningful.\n- **Seek support** — Grief support groups or counseling can provide meaningful comfort.\n- **Self-care basics** — Rest, eat, and be gentle with yourself during this time.\n\nI'm here to listen if you'd like to share."),
        ("I feel worthless", "I want you to know that feeling worthless does not mean you are worthless. This is an important distinction.\n\n- **Challenge the narrative** — Would you say this about someone you care about? Extend that same compassion to yourself.\n- **Recognize your strengths** — Even small ones matter more than you think.\n- **Thoughts are not facts** — Negative self-perception is often a distortion, not reality.\n- **Professional support** — A therapist can help you develop healthier thought patterns.\n- **Be patient** — Rebuilding self-worth is a process, not an event.\n\nYou matter. Would you like to explore these feelings further?"),
    ]
    
    for q, a in builtin_docs:
        docs.append(Document(
            page_content=f"Q: {q}\nA: {a}",
            metadata={"source": "builtin"}
        ))
    
    # Load from HuggingFace dataset
    try:
        ds = load_dataset("ShenLab/MentalChat16K", split="train[:500]")
        
        dataset_count = 0
        for idx, row in enumerate(ds):
            user_input = row['input'].strip()
            bot_reply = row['output'].strip()
            
            if len(user_input) < 10 or len(bot_reply) < 10:
                continue
            
            docs.append(Document(
                page_content=f"Q: {user_input}\nA: {bot_reply}",
                metadata={"source": "MentalChat16K", "index": idx}
            ))
            dataset_count += 1
        
        print(f"Loaded {dataset_count} documents from MentalChat16K dataset.")
        
    except Exception as e:
        print(f"Could not load HuggingFace dataset: {e}")
        print("Continuing with built-in Q&A pairs only.")

    vector_store.add_documents(docs)
    print(f"Vector store ready — {len(docs)} total Q&A pairs indexed.")

    # ─── Main retrieval function ───
    def ask_with_followups(question: str, chat_history: list) -> dict:
        start = time.time()
        
        results = vector_store.similarity_search(question, k=5)
        
        if results:
            best_doc = results[0]
            answer = extract_answer(best_doc.page_content)
            matched_q = extract_question(best_doc.page_content)
            
            print(f"\n  Query: '{question}'")
            print(f"  Match: '{matched_q[:80]}...'")
            print(f"  Source: {best_doc.metadata.get('source', 'unknown')}")
            
            # Append professional disclaimer for serious topics
            if needs_disclaimer(question) or needs_disclaimer(answer):
                answer += DISCLAIMER_NOTE
            
            # Generate follow-ups from dataset (related questions)
            followups = []
            seen = set()
            for doc in results[1:]:
                q = extract_question(doc.page_content)
                display_q = q if len(q) <= 55 else q[:52] + "..."
                if display_q not in seen and display_q != matched_q:
                    followups.append(display_q)
                    seen.add(display_q)
                if len(followups) >= 3:
                    break
                    
        else:
            answer = (
                "I'm here to help with topics related to mental well-being. "
                "Could you tell me more about how you're feeling?"
            )
            followups = ["I feel anxious", "I feel sad", "I feel stressed"]
        
        end = time.time()
        source_docs = [doc.page_content for doc in results[:2]] if results else []
        
        return {
            "reply": answer,
            "source_documents": source_docs,
            "followups": followups,
            "response_time": round(end - start, 2)
        }
    
    return ask_with_followups
