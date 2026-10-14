import pandas as pd
import google.generativeai as genai
import json
import csv
import os
import time  # <-- Added missing import

# 1. Add your Gemini API Key (set GEMINI_API_KEY env var)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Using the latest stable flash model alias to avoid 404 errors
model = genai.GenerativeModel('gemini-flash-latest')

# 2. Setup Files
input_file = 'youtube_comments_multiple.csv' 
output_file = 'IR_Benchmark_Dataset.csv'

if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'Document', 'Relevance_Score', 'Reason'])

df = pd.read_csv(input_file)

# --- SET YOUR STARTING ROW HERE ---
start_row = 10450 

all_data = [(index + 2, str(row['Comment'])) for index, row in df.iterrows()]
data_to_process = all_data[start_row:] 

def generate_query(row_num, comment):
    prompt = f"""
    You are building an IR dataset for Sri Lankan social media.
    Read this comment: "{comment}"
    
    Generate a JSON response:
    {{"Query": "A realistic Singlish user query that this comment answers", 
      "Score": 2, 
      "Reason": "Why it matches"}}
    """
    
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                print(f"⚠️ [Row {row_num}] API Rate Limit Hit. Cooling down for 35 seconds...")
                time.sleep(35) # Waits exactly as long as Google requested
                continue
            else:
                print(f"\n❌ [Error on Row {row_num}]: {e}")
                return None
                
    return None

print(f"🚀 Starting Safe Sequential Processing from Row {start_row}...\n")

# 3. Process sequentially to respect the 5 Requests Per Minute limit
for row_num, comment in data_to_process:
    
    print(f"⏳ Fetching Row {row_num}...")
    result = generate_query(row_num, comment)
    
    if result:
        print("-" * 70)
        print(f"📄 ANSWER (Row {row_num}): {comment}")
        print(f"🔎 QUERY: {result.get('Query')}")
        print(f"📊 SCORE: {result.get('Score')} | REASON: {result.get('Reason')}")
        print("-" * 70)
        
        while True:
            decision = input("Add it to data set? [y = Yes | n = No | e = Edit | q = Quit]: ").strip().lower()
            
            if decision == 'y':
                with open(output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([result['Query'], comment, result['Score'], result['Reason']])
                print("✅ Added.\n")
                break
                
            elif decision == 'n':
                print("⏭️ Skipped.\n")
                break
                
            elif decision == 'e':
                new_query = input("✏️ Type your improved query: ")
                with open(output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([new_query, comment, result['Score'], "Manually edited query"])
                print("✅ Edited and Added.\n")
                break
                
            elif decision == 'q':
                print("🛑 Exiting script. Your progress is saved safely!")
                exit()
            else:
                print("❌ Invalid input. Please type 'y', 'n', 'e', or 'q'.")
                
    # Add a mandatory 12-second delay between requests to stay exactly at 5 RPM (60s / 5 = 12s)
    time.sleep(12) 

print("🎉 Finished processing all remaining rows in the dataset!")