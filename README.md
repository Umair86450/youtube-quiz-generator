# 🎥 YouTube Video Quiz Generator

This Streamlit-based application allows you to **generate an advanced quiz from any YouTube video** using its transcript. Designed especially for educators, researchers, and learners, it uses LLMs (via Groq) to create high-quality **multiple-choice questions (MCQs)** and **summary questions** that test deep understanding.

---

## 🚀 Features

- ✅ Fetches transcript of a YouTube video (in Hindi or English)
- ✂️ Splits the transcript into logical **chunks** (max 2500 characters) to work better with LLMs and avoid token overflow.
- 📚 Generates **20 MCQs** using LLM (LLama3 via Groq)
- 🧠 Generates **5 high-level summary questions** with model answers
- 🔁 Ensures **no duplicate questions** and **clean formatting**
- 🎨 Beautiful, responsive UI powered by Streamlit
- 🧪 Uses advanced prompt engineering for better logic-building questions

---

## 🛠️ Tech Stack / Tools Used

| Tool / Library          | Purpose |
|------------------------|---------|
| `streamlit`            | Interactive UI |
| `langchain`            | LLM chaining and prompt handling |
| `langchain-groq`       | Accessing Groq's LLama3 model |
| `youtube-transcript-api` | Fetching YouTube video transcript |
| `dotenv`               | Securely storing and loading API keys |
| `logging`, `re`        | Python built-in modules for debug and regex |

---

## ⚙️ How It Works (Flow)

1. **Input**: User enters a YouTube video ID (e.g., `dQw4w9WgXcQ`)
2. **Transcript Retrieval**: The app fetches the transcript using `youtube-transcript-api`, first in Hindi (`hi`), then falls back to English (`en`) if not found.
3. **Chunking**: The full transcript is split into **text chunks (~2500 characters)** to handle long inputs effectively for LLMs(to work better with LLMs and avoid token overflow).
4. **MCQ Generation**:
    - Each chunk is passed through a prompt to the **LLama3 model via Groq**
    - 20 **deep, logic-based MCQs** are created with 4 options each
    - Duplicates are removed
5. **Summary Question Generation**:
    - From the first chunk, 5 open-ended **summary questions** are generated
    - Each includes a model answer (max 4–5 sentences)
6. **Display**:
    - Questions are formatted and shown in two expanders (MCQs and Summaries)
    - Styled with custom CSS for dark mode and readability

---

## 📦 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/yt-quiz-generator.git
cd yt-quiz-generator
````

### Step 2: Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up `.env` file

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

Open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📝 Output Example

* 20 High-quality, PhD-level **MCQs** with correct answers
* 5 **Summary Questions** with concise, explanatory answers
* Designed for higher education, test prep, and deep comprehension

---

## 🔐 Notes

* Works best with **educational or spoken-content videos** with clear transcripts
* Groq's **LLama3-8B model** is used — fast and cost-efficient

---

