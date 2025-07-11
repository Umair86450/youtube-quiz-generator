import logging
import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_youtube_transcript(video_id, primary_language='hi', fallback_language='en'):
    """Retrieve the transcript for the specified video ID, trying primary language first, then fallback."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[primary_language])
        logger.info(f"Retrieved transcript in {primary_language}")
        return " ".join([entry['text'] for entry in transcript]), primary_language
    except Exception as e:
        logger.warning(f"Failed to retrieve transcript in {primary_language}: {e}")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[fallback_language])
            logger.info(f"Retrieved transcript in {fallback_language}")
            return " ".join([entry['text'] for entry in transcript]), fallback_language
        except Exception as e:
            logger.error(f"Error retrieving transcript in {fallback_language}: {e}")
            return None, None





def chunk_text(text, max_len=2500):
    """Split text into chunks of max_len characters without cutting words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_len:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def deduplicate_mcqs(mcq_list):
    """Remove duplicate MCQs based on question text and answer options."""
    seen = set()
    unique_mcqs = []
    question_pattern = re.compile(r'Q\d+\.\s*(.*?)\nA\).*?\nB\).*?\nC\).*?\nD\).*?\nCorrect Answer:.*$', re.DOTALL)

    for mcq in mcq_list:
        match = question_pattern.search(mcq)
        if match:
            # Create a tuple of question text and answer options for uniqueness check
            question_body = match.group(1).strip()
            full_text = mcq.strip()
            signature = (question_body, tuple(re.findall(r'[A-D]\)\s*(.*?)(?=\n|$)', full_text)))
            if signature not in seen:
                seen.add(signature)
                unique_mcqs.append(mcq)
    return unique_mcqs



import re

def clean_mcq_format(mcq_text):
    """Clean and standardize MCQ format."""
    # First, fix any cases where there are extra line breaks between options and correct answer
    mcq_text = re.sub(r'\n\n+Correct Answer:', '\nCorrect Answer:', mcq_text)
    
    # Fix any cases where there are too few line breaks between question and options
    mcq_text = re.sub(r'([.?])\nA\)', r'\1\n\nA)', mcq_text)
    
    # Standardize spacing between options
    for option in [r'A\)', r'B\)', r'C\)', r'D\)']:  # Use raw strings here
        mcq_text = re.sub(fr'\n{option}', f'\n{option}', mcq_text)
    
    # Ensure there's a single line break before "Correct Answer:"
    mcq_text = re.sub(r'\n\n+Correct Answer:', '\nCorrect Answer:', mcq_text)
    
    return mcq_text


def format_mcqs(mcqs):
    """Format MCQs with proper indentation and spacing."""
    formatted = []
    for mcq in mcqs:
        # Clean and standardize the MCQ format
        mcq = clean_mcq_format(mcq.strip())
        
        # Split MCQ into lines
        lines = mcq.split('\n')
        formatted_lines = [lines[0]]  # Question line (e.g., Q1. ...)
        
        # Indent options and correct answer
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                formatted_lines.append(f"    {line}")
        
        formatted.append('\n'.join(formatted_lines))
    
    return '\n\n'.join(formatted)

def clean_summary_question(text):
    """Clean summary question text by removing unwanted characters and formatting."""
    # Remove ** markers and other unwanted formatting
    text = re.sub(r'\*\*', '', text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def format_summary_questions_for_text_display(summary_text):
    """Format summary questions for plain text display."""
    # Clean the input text
    summary_text = clean_summary_question(summary_text)
    
    # Match summary questions and answers separately
    question_pattern = r'Summary Question (\d+):(.*?)(?=Answer:|$)'
    answer_pattern = r'Answer:(.*?)(?=Summary Question \d+:|$)'
    
    questions = re.findall(question_pattern, summary_text, re.DOTALL)
    answers = re.findall(answer_pattern, summary_text, re.DOTALL)
    
    formatted = []
    
    for i, (q_num, q_text) in enumerate(questions):
        # Format the question
        formatted.append(f"Summary Question {q_num}:{q_text.strip()}")
        
        # Format the answer if available
        if i < len(answers):
            answer = answers[i].strip()
            answer_lines = answer.split('\n')
            
            # Add the first line with "Answer:" prefix
            formatted.append(f"    Answer: {answer_lines[0]}")
            
            # Add remaining lines with indentation
            for line in answer_lines[1:]:
                if line.strip():  # Skip empty lines
                    formatted.append(f"    {line.strip()}")
    
    return '\n\n'.join(formatted)

def extract_summary_questions(summary_text):
    """Extract summary questions and answers to be displayed separately."""
    # Clean the input text
    summary_text = clean_summary_question(summary_text)
    
    # Match summary questions and answers separately
    question_pattern = r'Summary Question (\d+):(.*?)(?=Answer:|$)'
    answer_pattern = r'Answer:(.*?)(?=Summary Question \d+:|$)'
    
    questions = re.findall(question_pattern, summary_text, re.DOTALL)
    answers = re.findall(answer_pattern, summary_text, re.DOTALL)
    
    result = []
    for i, (q_num, q_text) in enumerate(questions):
        if i < len(answers):
            result.append({
                "number": q_num.strip(),
                "question": q_text.strip(),
                "answer": answers[i].strip()
            })
    
    return result

def generate_quiz_with_groq(chunks, num_mcqs=20, num_summary_questions=5):
    """Generate a quiz with unique, logic-building MCQs and summary questions."""
    try:
        # Check if API key is accessible
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in .env file")
            raise ValueError("GROQ_API_KEY is not configured in the .env file")

        groq_model = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192"
        )

        # Prompt for MCQs with strict formatting guidelines
        mcq_prompt = """
        You are an expert educator crafting a highly challenging, PhD-level quiz based on the content below.
        Generate exactly {num_mcqs} multiple-choice questions (MCQs) that test deep understanding, critical analysis, and logic-building skills.
        
        Each question must:
        - Be unique, covering distinct concepts, themes, question types, or perspectives from the content to avoid any repetition.
        - Require logical reasoning, problem-solving, or analysis of complex scenarios (e.g., edge cases, hypothetical applications, or theoretical implications).
        - Be challenging, resembling 'treasure questions' that demand critical thinking to solve.
        - Have 4 answer options, with only one correct answer.
        - Include highly plausible distractors that are technically accurate but subtly incorrect, challenging advanced readers.
        - Provide ONLY the correct answer letter (e.g., "Correct Answer: B") without any explanations.
        
        STRICT FORMATTING RULES:
        1. Format each question exactly as shown below, with no deviations:
        QX. [Question text]
        A) [Option 1]
        B) [Option 2]
        C) [Option 3]
        D) [Option 4]
        Correct Answer: [Correct option letter]
        
        2. Keep exactly one line break between each option and between the last option and "Correct Answer:"
        3. Do not add any asterisks, special characters, or additional formatting
        4. Do not add any explanations or additional text after "Correct Answer: X"

        Content:
        {text}

        MCQs:
        """

        # Prompt for summary questions with strict formatting guidelines
        summary_prompt = """
        You are an expert educator creating a PhD-level quiz based on the content below.
        Generate exactly {num_summary_questions} open-ended, reflective questions that demand critical synthesis, evaluation, and application of the content's core concepts.
        
        Each question should:
        - Require deep analysis, exploring theoretical implications, design logic, or edge cases.
        - Be challenging enough for a PhD-level audience, encouraging novel insights or connections.
        - Include a detailed answer that explains the reasoning and ties back to the content.
        - Keep the answer CONCISE (maximum 4-5 sentences) and directly relevant to the question.

        STRICT FORMATTING RULES:
        1. Format each question exactly as shown below, with no deviations:
        Summary Question 1: [Write the full question text here]
        Answer: [Write the detailed answer here in 4-5 sentences maximum]

        Summary Question 2: [Write the full question text here]
        Answer: [Write the detailed answer here in 4-5 sentences maximum]

        2. Do not use any asterisks, special characters, or markdown formatting
        3. Keep exactly one blank line between each summary question
        4. The question text should appear directly after "Summary Question X:" and the answer should appear directly after "Answer:"
        5. Do not include any additional notes, explanations, or text outside the specified format
        6. Keep answers concise and to the point - NO LONG PARAGRAPHS
        
        Content:
        {text}

        Summary Questions:
        """

        # Create prompt templates
        mcq_prompt_template = PromptTemplate(
            input_variables=["text", "num_mcqs"],
            template=mcq_prompt
        )
        summary_prompt_template = PromptTemplate(
            input_variables=["text", "num_summary_questions"],
            template=summary_prompt
        )

        # Initialize chains
        mcq_chain = LLMChain(llm=groq_model, prompt=mcq_prompt_template)
        summary_chain = LLMChain(llm=groq_model, prompt=summary_prompt_template)

        # Generate MCQs from chunks
        all_mcqs = []
        chunks_to_use = chunks[:3]  # Use up to 3 chunks for diversity
        if not chunks_to_use:
            raise ValueError("No valid chunks available for quiz generation")
        
        mcqs_per_chunk = max(1, (num_mcqs // len(chunks_to_use)) + 1)  # Distribute MCQs
        for i, chunk in enumerate(chunks_to_use):
            logger.info(f"Generating {mcqs_per_chunk} MCQs for chunk {i+1}...")
            mcq_text = mcq_chain.run(text=chunk, num_mcqs=mcqs_per_chunk)
            chunk_mcqs = re.split(r'\n(?=Q\d+\.)', mcq_text.strip())
            all_mcqs.extend([mcq.strip() for mcq in chunk_mcqs if mcq.strip()])

        # Deduplicate MCQs
        unique_mcqs = deduplicate_mcqs(all_mcqs)
        logger.info(f"After deduplication, {len(unique_mcqs)} unique MCQs generated")

        # If fewer than num_mcqs, generate more from a random chunk
        if len(unique_mcqs) < num_mcqs:
            logger.warning(f"Only {len(unique_mcqs)} unique MCQs generated, generating {num_mcqs - len(unique_mcqs)} more...")
            remaining = num_mcqs - len(unique_mcqs)
            additional_mcq_text = mcq_chain.run(text=chunks[0], num_mcqs=remaining)
            additional_mcqs = re.split(r'\n(?=Q\d+\.)', additional_mcq_text.strip())
            unique_mcqs.extend(deduplicate_mcqs([mcq.strip() for mcq in additional_mcqs if mcq.strip()]))

        # Trim to exactly num_mcqs and reindex
        unique_mcqs = unique_mcqs[:num_mcqs]
        if len(unique_mcqs) < num_mcqs:
            logger.warning(f"Generated only {len(unique_mcqs)} unique MCQs instead of {num_mcqs}")
        
        # Reindex MCQs to Q1, Q2, etc.
        for i, mcq in enumerate(unique_mcqs):
            unique_mcqs[i] = re.sub(r'Q\d+\.', f'Q{i+1}.', mcq, 1)

        # Generate summary questions from the first chunk
        logger.info("Generating summary questions...")
        summary_text = summary_chain.run(text=chunks[0], num_summary_questions=num_summary_questions)

        # Clean and format the output
        formatted_mcqs = format_mcqs(unique_mcqs)
        formatted_summary = format_summary_questions_for_text_display(summary_text.strip())

        # Combine outputs with proper formatting
        quiz_text = f"Multiple-Choice Questions:\n\n{formatted_mcqs}\n\n\nSummary Questions:\n\n{formatted_summary}"
        
        # Return the raw formatted text along with structured data for summary questions
        return quiz_text, extract_summary_questions(summary_text)
    
    except Exception as e:
        logger.error(f"Error in generate_quiz_with_groq: {str(e)}")
        raise

def main():
    # Streamlit app configuration
    st.set_page_config(page_title="YouTube Video Quiz Generator", page_icon="🎥", layout="wide")
    
    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main { 
            background-color: #f3f4f6; 
            padding: 2rem; 
        }
        .stButton>button { 
            background-color: #2563eb; 
            color: white; 
            padding: 0.75rem 1.5rem; 
            border-radius: 0.375rem; 
            border: none; 
            font-weight: 600; 
            transition: background-color 0.2s; 
        }
        .stButton>button:hover { 
            background-color: #1e40af; 
        }
        .stButton>button:disabled { 
            opacity: 0.5; 
            cursor: not-allowed; 
        }
        .stTextInput>div>input { 
            border: 1px solid #d1d5db; 
            border-radius: 0.375rem; 
            padding: 0.75rem; 
            font-size: 1rem;
        }
        .stTextInput>div>input:focus { 
            outline: none; 
            border-color: #2563eb; 
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2); 
        }
        /* MCQ container */
        .mcq-container {
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #3a3a3a;
            margin-top: 1rem;
            font-family: 'Fira Code', 'Courier New', monospace;
            font-size: 0.95rem;
            line-height: 1.6;
            white-space: pre-wrap;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            overflow-x: auto;
        }
        .quiz-section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #60a5fa;
            margin-bottom: 1rem;
        }
        .stExpander {
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            background-color: #ffffff;
            margin-bottom: 1rem;
        }
        .stExpander summary {
            font-weight: 600;
            color: #1e40af;
        }
        .stSuccess { 
            background-color: #d1fae5; 
            color: #065f46; 
        }
        .stError { 
            background-color: #fee2e2; 
            color: #991b1b; 
        }
        /* Summary question card styling */
        .summary-card {
            background-color: #1e1e1e;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            border: 1px solid #3a3a3a;
        }
        .summary-question {
            font-weight: bold;
            padding: 15px;
            background-color: #2a2a2a;
            color: #60a5fa;
        }
        .summary-answer-container {
            padding: 15px;
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description with professional wording
    st.title("🎥 YouTube Video Quiz Generator")
    st.markdown("""
        **Create a Quiz from YouTube Video Transcripts**  
        Input a YouTube video ID to generate 20 Multiple-choice questions (MCQs) with correct answers and 5 in-depth summary questions with detailed answers, designed for advanced academic analysis.  
    """)

    # Input section
    video_id = st.text_input("Enter YouTube Video ID", placeholder="e.g., dQw4w9WgXcQ", key="video_id")
    
    # Generate quiz button
    generate_button = st.button("Generate Quiz", disabled=not video_id, type="primary")

    if generate_button and video_id:
        with st.spinner("Fetching transcript and generating quiz..."):
            if len(video_id) < 11:
                st.error("Invalid video ID. Please enter a valid YouTube video ID (e.g., dQw4w9WgXcQ).")
            else:
                try:
                    transcript, language = get_youtube_transcript(video_id, primary_language='hi', fallback_language='en')
                    
                    if transcript:
                        st.success(f"✅ Transcript retrieved successfully in {language}.")
                        
                        # Chunk text
                        chunks = chunk_text(transcript, max_len=2500)
                        if not chunks:
                            st.error("Transcript is empty or could not be processed. Please try another video ID.")
                            logger.error("Empty chunks generated from transcript")
                            return

                        quiz_text, summary_questions = generate_quiz_with_groq(chunks, num_mcqs=20, num_summary_questions=5)
                        
                        # Split quiz into MCQs and summary questions for display
                        parts = quiz_text.split("Summary Questions:", 1)
                        mcq_section = parts[0].strip()

                        st.subheader("Generated Quiz")
                        
                        # Display MCQs in an expander
                        with st.expander("Multiple-Choice Questions (20)", expanded=True):
                            st.markdown(
                                f"<div class='mcq-container'><div class='quiz-section-header'>Multiple-Choice Questions</div>{mcq_section}</div>",
                                unsafe_allow_html=True
                            )

                        # Display summary questions using Streamlit components directly
                        with st.expander("Summary Questions (5)", expanded=True):
                            # Use Streamlit container for custom styling
                            summary_container = st.container()
                            
                            for question in summary_questions:
                                summary_container.markdown(f"""
                                <div class="summary-card">
                                    <div class="summary-question">Summary Question {question['number']}: {question['question']}</div>
                                    <div class="summary-answer-container">
                                        <strong style="color: #60a5fa;">Answer:</strong><br>
                                        {question['answer']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Transcript not found or could not be retrieved. Please try a video with available transcripts (e.g., educational content).")
                except Exception as e:
                    st.error(f"Failed to generate quiz: {str(e)}")
                    logger.error(f"Main function error: {str(e)}")

if __name__ == "__main__":
    main()