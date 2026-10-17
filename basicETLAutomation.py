import os
import re
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()

# Initialize OpenAI client (ensure OPENAI_API_KEY is in your environment variables)
client = OpenAI()

def clean_via_llm(row_data: dict) -> dict:
    """
    Passes the raw row data to GPT-4o-mini to standardize the fields as defined in the n8n Basic LLM Chain.
    """
    prompt = f"""
    Clean this student enrollment row and return ONLY a JSON object.

    Row data: {json.dumps(row_data)}

    Rules:
    - student_name: Title Case (Manoj K)
    - parent_email: if invalid set INVALID_EMAIL
    - parent_phone: if less than 10 digits set MISSING
    - fee_status: Paid/YES = true, Unpaid/NO = false, Partial=false
    - City: Title Case, empty = UNKNOWN
    - date_of_birth: convert to YYYY-MM-DD

    Return ONLY raw JSON. No explanation. No markdown.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw_output = response.choices[0].message.content.strip()
    
    # Strip markdown code fences if present
    raw_output = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.IGNORECASE)
    raw_output = re.sub(r'\s*```$', '', raw_output)
    
    return json.loads(raw_output)

def normalize_student_record(student: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replicates the 'Code in JavaScript' node's data validation and normalization.
    """
    result = student.copy()

    # Clean string fields
    string_fields = [
        "student_id", "student_name", "class", "section", "gender", 
        "parent_guardian_name", "parent_email", "parent_phone", 
        "admission_date", "student_status", "City"
    ]
    for key in string_fields:
        if result.get(key) is not None:
            result[key] = str(result[key]).strip()

    # Title Case names
    if result.get("student_name"):
        result["student_name"] = result["student_name"].title()
    if result.get("parent_guardian_name"):
        result["parent_guardian_name"] = result["parent_guardian_name"].title()

    # Email validation
    email = result.get("parent_email", "")
    if email and re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        result["parent_email"] = email.lower()
    else:
        result["parent_email"] = "INVALID_EMAIL"

    # Phone validation
    phone = re.sub(r'\D', '', str(result.get("parent_phone", "")))
    result["parent_phone"] = phone if len(phone) >= 10 else "MISSING"

    # Normalize numeric fee fields
    for key in ["annual_fee_inr", "fees_paid_inr", "fees_due_inr"]:
        val = result.get(key)
        if val not in ["", None]:
            try:
                result[key] = float(val)
            except ValueError:
                pass

    # Fee status normalization
    fee_status = result.get("fee_status")
    if isinstance(fee_status, str):
        status_lower = fee_status.strip().lower()
        if status_lower in ["paid", "partial", "unpaid"]:
            result["fee_status"] = status_lower.capitalize()
        elif status_lower in ["yes", "true"]:
            result["fee_status"] = True
        elif status_lower in ["no", "false"]:
            result["fee_status"] = False

    # Derive boolean fee status from due amount if necessary
    if isinstance(result.get("fee_status"), bool) and isinstance(result.get("fees_due_inr"), (int, float)):
        result["fee_status"] = (result["fees_due_inr"] == 0)

    # Normalize dates to YYYY-MM-DD
    for key in ["date_of_birth", "admission_date"]:
        val = result.get(key)
        if val:
            match = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', str(val))
            if match:
                year, month, day = match.groups()
                result[key] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # Format student status
    if result.get("student_status"):
        result["student_status"] = result["student_status"].capitalize()

    return result

def process_unpaid_students(input_filepath: str, output_filepath: str):
    """
    Main function replicating the full n8n workflow.
    """
    # 1. Extract from File
    if input_filepath.endswith('.csv'):
        df = pd.read_csv(input_filepath)
    else:
        df = pd.read_excel(input_filepath)

    processed_records: List[Dict[str, Any]] = []

    # 2 & 3. Process through LLM and Custom Code Logic
    for _, row in df.iterrows():
        raw_row_dict = row.dropna().to_dict()
        
        try:
            # LLM Clean
            llm_cleaned_json = clean_via_llm(raw_row_dict)
            
            # Script Normalization
            normalized_record = normalize_student_record(llm_cleaned_json)
            processed_records.append(normalized_record)
            
        except Exception as e:
            print(f"Error processing row: {raw_row_dict}\nError: {e}")

    processed_df = pd.DataFrame(processed_records)

    # 4. Switch (Filter where fee_status is false/unpaid)
    # The n8n logic looks for fee_status == "false" (or boolean False)
    if 'fee_status' in processed_df.columns:
        unpaid_df = processed_df[processed_df['fee_status'] == False]
    else:
        unpaid_df = pd.DataFrame()
        print("Warning: 'fee_status' column not found in parsed data.")

    # 5. Convert to File
    unpaid_df.to_csv(output_filepath, index=False)
    print(f"Successfully processed records. Filtered unpaid students saved to {output_filepath}")

if __name__ == "__main__":
    # Example usage:
    process_unpaid_students("student_administrative_data_50.csv", "unpaid_students.csv")
    pass