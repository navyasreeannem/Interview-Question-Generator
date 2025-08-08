from flask import Flask, render_template, request, jsonify
import os
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import re
import json

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Get API key and verify it exists
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
else:
    print("API key found successfully")

# Configure Gemini API with explicit API key
genai.configure(api_key=api_key)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_file):
    """Extract text content from a PDF file"""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def validate_questions_count(content, num_questions):
    """Validate that the content has exactly the requested number of questions"""
    # Count the number of questions in the content
    question_pattern = r"Question\s+\d+:"
    questions = re.findall(question_pattern, content, re.IGNORECASE)
    
    return len(questions) == num_questions

def calculate_question_distribution(total_questions):
    """Calculate how many questions should be in each category"""
    # Default distribution (adjust as needed)
    distribution = {
        "Technical": max(1, int(total_questions * 0.4)),  # 40% technical
        "Behavioral": max(1, int(total_questions * 0.2)),  # 20% behavioral
        "Situational": max(1, int(total_questions * 0.15)),  # 15% situational
        "Cultural/Personality": max(1, int(total_questions * 0.1)),  # 10% cultural
        "Problem-Solving": max(1, int(total_questions * 0.15))  # 15% problem-solving
    }
    
    # Adjust to ensure the total is correct
    current_total = sum(distribution.values())
    if current_total < total_questions:
        # Add the remaining questions to technical category
        distribution["Technical"] += (total_questions - current_total)
    elif current_total > total_questions:
        # Remove excess questions proportionally
        excess = current_total - total_questions
        for category in sorted(distribution.keys(), key=lambda k: distribution[k], reverse=True):
            if distribution[category] > 1 and excess > 0:
                reduction = min(distribution[category] - 1, excess)
                distribution[category] -= reduction
                excess -= reduction
            if excess == 0:
                break
    
    return distribution

def generate_categorized_questions(resume_text, jd_text, num_questions=10, complexity="intermediate", retry_count=0):
    """Generate categorized interview questions using Gemini API"""
    try:
        # Calculate the distribution of questions
        distribution = calculate_question_distribution(num_questions)
        
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,  # Increased for more detailed answers
        }
        
        # Define complexity guidelines based on selected level
        complexity_guidelines = {
            "basic": "Create basic-level questions suitable for entry-level candidates or those new to the field. Questions should cover fundamental concepts, basic terminology, and simple scenarios that don't require deep expertise.",
            "intermediate": "Create intermediate-level questions suitable for candidates with 2-3 years of experience. Questions should require practical knowledge, some depth of understanding, and the ability to apply concepts in typical scenarios.",
            "advanced": "Create advanced-level questions suitable for senior candidates with 5+ years of experience. Questions should be challenging, cover complex scenarios, edge cases, architectural decisions, and demonstrate deep expertise in the field."
        }
        
        # Get the appropriate complexity guideline
        complexity_guideline = complexity_guidelines.get(complexity, complexity_guidelines["intermediate"])
        
        # Create the prompt with specific categories and number of questions
        prompt = f"""
        You are an expert technical interviewer with deep knowledge in various technical domains. Your task is to create EXACTLY {num_questions} relevant interview questions
        based on a candidate's resume and the job description they're applying for, along with comprehensive, detailed answers.
        
        Resume:
        {resume_text}
        
        Job Description:
        {jd_text}
        
        COMPLEXITY LEVEL: {complexity.upper()}
        {complexity_guideline}
        
        CRITICAL INSTRUCTION: You MUST generate EXACTLY {num_questions} questions in total, distributed across the following categories:
        
        1. Technical Questions: {distribution["Technical"]} questions
        2. Behavioral Questions: {distribution["Behavioral"]} questions
        3. Situational Questions: {distribution["Situational"]} questions
        4. Cultural/Personality Questions: {distribution["Cultural/Personality"]} questions
        5. Problem-Solving Questions: {distribution["Problem-Solving"]} questions
        
        For each question, provide a comprehensive, detailed answer that:
        - Thoroughly explains the concept or approach
        - Includes specific examples where appropriate
        - Mentions best practices and industry standards
        - Provides context for why this knowledge is important for the role
        - Is at least 150-200 words in length to ensure depth and completeness
        
        IMPORTANT: The answers should be CORRECT ANSWERS that a qualified candidate would ideally provide, not expected answers or evaluation criteria.
        
        Format your response STRICTLY as follows:
        
        TECHNICAL QUESTIONS:
        
        Question 1: [The technical interview question]
        Answer 1: [Comprehensive, detailed correct answer with examples and context]
        
        Question 2: [The technical interview question]
        Answer 2: [Comprehensive, detailed correct answer with examples and context]
        
        ... and so on for all technical questions
        
        BEHAVIORAL QUESTIONS:
        
        Question [continue numbering]: [The behavioral interview question]
        Answer [continue numbering]: [Comprehensive, detailed correct answer with examples and context]
        
        ... and so on for all behavioral questions
        
        SITUATIONAL QUESTIONS:
        
        Question [continue numbering]: [The situational interview question]
        Answer [continue numbering]: [Comprehensive, detailed correct answer with examples and context]
        
        ... and so on for all situational questions
        
        CULTURAL/PERSONALITY QUESTIONS:
        
        Question [continue numbering]: [The cultural/personality interview question]
        Answer [continue numbering]: [Comprehensive, detailed correct answer with examples and context]
        
        ... and so on for all cultural/personality questions
        
        PROBLEM-SOLVING QUESTIONS:
        
        Question [continue numbering]: [The problem-solving interview question]
        Answer [continue numbering]: [Comprehensive, detailed correct answer with examples and context]
        
        ... and so on for all problem-solving questions
        
        IMPORTANT: You MUST generate EXACTLY {num_questions} questions in total, with the exact distribution specified above.
        Make sure each answer is thorough, technically accurate, and represents what a qualified candidate should ideally answer.
        
        CRITICAL: ALL questions MUST strictly adhere to the {complexity.upper()} complexity level as defined above. Do not mix complexity levels.
        
        Before submitting your response, verify that you have created EXACTLY {num_questions} questions and answers with the correct distribution across categories.
        """
        
        # Initialize the model with explicit API key
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )
        
        # Generate the response
        response = model.generate_content(prompt)
        content = response.text
        
        # Validate that we have the correct number of questions
        if validate_questions_count(content, num_questions):
            return content
        elif retry_count < 2:  # Retry up to 2 times
            print(f"Incorrect number of questions generated. Retrying... ({retry_count + 1})")
            return generate_categorized_questions(resume_text, jd_text, num_questions, complexity, retry_count + 1)
        else:
            # If we've retried and still don't have the right number, force the correct format
            return force_correct_categorized_questions(content, num_questions, distribution, complexity)
            
    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")
        raise e

def extract_questions_by_category(content):
    """Extract questions and answers by category from the content"""
    categories = {
        "TECHNICAL QUESTIONS": [],
        "BEHAVIORAL QUESTIONS": [],
        "SITUATIONAL QUESTIONS": [],
        "CULTURAL/PERSONALITY QUESTIONS": [],
        "PROBLEM-SOLVING QUESTIONS": []
    }
    
    # Extract content for each category
    current_category = None
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a category header
        for category in categories.keys():
            if category in line.upper():
                current_category = category
                break
                
        # If we have a current category and this is a question or answer, add it
        if current_category and (line.startswith("Question") or line.startswith("Answer")):
            categories[current_category].append(line)
    
    return categories

def force_correct_categorized_questions(content, num_questions, distribution, complexity="intermediate"):
    """Force the content to have exactly the requested number of questions with the correct distribution"""
    # Extract existing questions and answers by category
    categories = extract_questions_by_category(content)
    
    # Prepare the new content
    new_content = ""
    question_counter = 1
    
    # Process each category
    for category, target_count in distribution.items():
        category_upper = category.upper() + " QUESTIONS"
        
        # Get existing questions and answers for this category
        category_lines = categories.get(category_upper, [])
        
        # Extract questions and answers
        questions = []
        answers = []
        
        i = 0
        while i < len(category_lines):
            if category_lines[i].startswith("Question"):
                questions.append(category_lines[i])
                if i+1 < len(category_lines) and category_lines[i+1].startswith("Answer"):
                    answers.append(category_lines[i+1])
                    i += 2
                else:
                    answers.append(f"Answer {question_counter}: No answer available")
                    i += 1
            else:
                i += 1
        
        # Adjust the number of questions to match the target
        while len(questions) > target_count:
            questions.pop()
            if answers:
                answers.pop()
                
        while len(questions) < target_count:
            q_num = question_counter + len(questions)
            questions.append(f"Question {q_num}: Additional {complexity} {category.lower()} question {len(questions) + 1}")
            answers.append(f"Answer {q_num}: Additional {complexity} {category.lower()} answer {len(questions)}")
        
        # Add the category header and questions to the new content
        new_content += f"\n{category_upper}:\n\n"
        
        for i in range(target_count):
            # Extract question number
            q_match = re.search(r"Question\s+(\d+):", questions[i])
            old_q_num = int(q_match.group(1)) if q_match else 0
            
            # Replace with the correct sequential number
            q_text = re.sub(r"Question\s+\d+:", f"Question {question_counter}:", questions[i])
            a_text = re.sub(r"Answer\s+\d+:", f"Answer {question_counter}:", answers[i]) if i < len(answers) else f"Answer {question_counter}: No answer available"
            
            new_content += f"{q_text}\n{a_text}\n\n"
            question_counter += 1
    
    return new_content

def generate_interview_questions(resume_text, jd_text, num_questions=10, complexity="intermediate"):
    """Main function to generate categorized interview questions"""
    try:
        return generate_categorized_questions(resume_text, jd_text, num_questions, complexity)
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Check if files are present in the request
        if 'resume' not in request.files or 'jd' not in request.files:
            return jsonify({"error": "Both resume and job description files are required"}), 400
        
        resume_file = request.files['resume']
        jd_file = request.files['jd']
        
        # Check if files are selected
        if resume_file.filename == '' or jd_file.filename == '':
            return jsonify({"error": "Both files must be selected"}), 400
        
        # Get the number of questions from the form (default to 10 if not provided)
        num_questions = int(request.form.get('num_questions', 10))
        
        # Get the complexity level (default to intermediate if not provided)
        complexity = request.form.get('complexity', 'intermediate')
        
        # Validate complexity level
        if complexity not in ['basic', 'intermediate', 'advanced']:
            complexity = 'intermediate'  # Default to intermediate if invalid
        
        # Ensure number of questions is within reasonable limits
        num_questions = max(5, min(num_questions, 20))
        
        # Extract text from PDFs
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)
        
        # Calculate the distribution of questions
        distribution = calculate_question_distribution(num_questions)
        
        # Generate interview questions with comprehensive answers
        content = generate_interview_questions(resume_text, jd_text, num_questions, complexity)
        
        # Verify one last time that we have the correct number of questions
        if not validate_questions_count(content, num_questions):
            content = force_correct_categorized_questions(content, num_questions, distribution, complexity)
        
        # Return the full content with questions and answers
        return jsonify({
            "content": content, 
            "num_questions": num_questions,
            "distribution": distribution,
            "complexity": complexity
        })
    
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Return the available question categories"""
    categories = [
        "Technical",
        "Behavioral",
        "Situational",
        "Cultural/Personality",
        "Problem-Solving"
    ]
    return jsonify({"categories": categories})

@app.route('/complexity-levels', methods=['GET'])
def get_complexity_levels():
    """Return the available complexity levels"""
    levels = [
        "basic",
        "intermediate",
        "advanced"
    ]
    return jsonify({"levels": levels})

@app.route('/distribution', methods=['POST'])
def get_distribution():
    """Calculate and return the distribution of questions"""
    try:
        data = request.get_json()
        num_questions = int(data.get('num_questions', 10))
        num_questions = max(5, min(num_questions, 20))
        
        distribution = calculate_question_distribution(num_questions)
        return jsonify({
            "distribution": distribution, 
            "total": num_questions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Print environment information for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables: {os.environ.keys()}")
    print(f"GOOGLE_API_KEY set: {'GOOGLE_API_KEY' in os.environ}")
    
    # Run the Flask app
    app.run(debug=True)
