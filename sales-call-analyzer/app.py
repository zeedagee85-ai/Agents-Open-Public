import gradio as gr
from groq import Groq
import os

# Initialize Groq client
# You'll add your API key in Hugging Face settings (I'll show you)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def analyze_sales_call(transcript, focus_area):
    """Analyzes sales calls and provides coaching feedback"""
    
    if not transcript.strip():
        return "⚠️ Please paste a sales call transcript to analyze."
    
    prompt = f"""You are an expert sales coach with 20 years of experience. Analyze this sales call with special focus on: {focus_area}

CALL TRANSCRIPT:
{transcript}

Provide a detailed analysis in this format:

## 📊 Overall Score: [X/10]

## ✅ What Went Well (Top 3)
1. [Specific example from call]
2. [Specific example from call]
3. [Specific example from call]

## 🎯 Areas for Improvement (Top 3)
1. [Specific issue + how to fix it]
2. [Specific issue + how to fix it]
3. [Specific issue + how to fix it]

## 💬 Exact Phrases to Use Next Time
- "[Specific phrase for situation X]"
- "[Specific phrase for situation Y]"
- "[Specific phrase for situation Z]"

## 🎲 Deal Probability: [X]%
[Brief explanation of why]

## 🔑 Key Takeaway
[One actionable insight they can use immediately]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nMake sure your GROQ_API_KEY is set correctly in Settings."

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_sales_call,
    inputs=[
        gr.Textbox(
            lines=15, 
            placeholder="Paste your sales call transcript here...\n\nExample:\nSales Rep: Hi John, thanks for taking my call today...\nProspect: No problem, what's this about?\nSales Rep: ...",
            label="📞 Sales Call Transcript"
        ),
        gr.Dropdown(
            choices=[
                "Overall Performance", 
                "Objection Handling", 
                "Discovery Questions", 
                "Closing Techniques", 
                "Rapport Building",
                "Value Proposition",
                "Active Listening"
            ],
            value="Overall Performance",
            label="🎯 Analysis Focus"
        )
    ],
    outputs=gr.Markdown(label="📋 AI Analysis"),
    title="🎯 Sales Call Analyzer Pro",
    description="""
    ### AI-Powered Sales Coaching in 30 Seconds
    
    Paste any sales call transcript and get expert coaching feedback instantly.
    
    **Powered by Llama 3.1 70B via Groq**
    """,
    examples=[
        [
            """Sales Rep: Hi Sarah, thanks for taking my call. I wanted to follow up on the proposal I sent over last week about our CRM solution.

Prospect: Yeah, I got it. Look, I'm pretty busy right now. Can you just give me the quick version?

Sales Rep: Absolutely! So basically, our CRM can save your team about 10 hours per week on administrative work. It's $99 per user per month.

Prospect: That seems expensive. We're currently using spreadsheets and they're free.

Sales Rep: Well, sure, but think about all the time you're wasting. Our solution is much better.

Prospect: I don't know, I'll have to think about it.

Sales Rep: Okay, should I follow up next week?

Prospect: Sure, whatever.""",
            "Overall Performance"
        ]
    ],
    theme=gr.themes.Soft(),
    analytics_enabled=False
)

if __name__ == "__main__":
    demo.launch()
