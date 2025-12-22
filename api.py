# =========================================
# MCP AI CHAT – ENHANCED ORCHESTRATOR
# Text + Vision + Image Edit + CSV Plots
# =========================================

import base64
import random
import urllib.parse
import requests
import os
import io
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
TEXT_API_URL = "https://text.pollinations.ai"
IMAGE_API_URL = "https://image.pollinations.ai/prompt"

HOST = "127.0.0.1"
PORT = 8001

UPLOAD_DIR = "uploads"
PLOTS_DIR = "plots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI(title="MCP AI Chat Enhanced")

# =============================
# MEMORY (IN-RAM)
# =============================
CHATS = {}
CHAT_COUNTER = 0
CHAT_FILES = {}  # Store files per chat: {chat_id: {"file_path": str, "file_type": str, "filename": str}}

# =============================
# FILE HANDLING
# =============================
def save_uploaded_file(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(file.file.read())
    file.file.seek(0)
    return path


def extract_text_from_file(file: UploadFile) -> str:
    name = file.filename.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file.file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text[:4000]

        if name.endswith(".csv"):
            file.file.seek(0)
            raw = file.file.read().decode("utf-8", errors="ignore")
            lines = raw.splitlines()
            header = lines[0] if lines else ""
            sample = "\n".join(lines[1:11])
            return f"CSV_DATA|||{header}|||{sample}"

        if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return "IMAGE_FILE"

        raw = file.file.read().decode("utf-8", errors="ignore")
        return raw[:4000]

    except:
        return ""

# =============================
# SMART INTENT DETECTION
# =============================
def detect_intent(prompt: str, has_file: bool = False, file_type: str = "") -> str:
    """
    Detects user intent from prompt
    Returns: 'image_gen', 'image_describe', 'image_edit', 'csv_plot', 'csv_edit', 'text'
    """
    q = prompt.lower()
    
    # Image editing keywords
    edit_keywords = ["edit", "modify", "change", "adjust", "filter", "enhance", 
                     "crop", "resize", "rotate", "brightness", "contrast", "blur"]
    
    # Image generation keywords
    gen_keywords = ["create", "generate", "make", "draw", "paint", "design", "produce"]
    

    
    # Visualization keywords
    viz_keywords = ["plot", "graph", "chart", "visualize", "show", "display", 
                    "trend", "distribution", "histogram", "scatter"]
    
    # CSV editing keywords
    csv_edit_keywords = ["add row", "delete row", "remove row", "add column", 
                         "delete column", "remove column", "drop", "insert", 
                         "filter", "sort", "clean", "remove duplicates"]
    
    # Check for image editing
    if has_file and file_type.endswith((".png", ".jpg", ".jpeg", ".webp")):
        if any(word in q for word in edit_keywords):
            return "image_edit"
        if any(word in q for word in desc_keywords):
            return "image_describe"
    
    # Check for CSV editing
    if has_file and file_type.endswith(".csv"):
        if any(word in q for word in csv_edit_keywords):
            return "csv_edit"
        if any(word in q for word in viz_keywords):
            return "csv_plot"
    
    # Check for image generation
    if any(word in q for word in gen_keywords) and any(word in q for word in ["image", "picture", "photo", "art"]):
        return "image_gen"
    
    # Check for general image request
    if "image" in q or "picture" in q or "photo" in q:
        return "image_gen"
    
    return "text"


def improve_image_prompt(prompt: str) -> str:
    """Enhance image generation prompts"""
    improve = f"""
Rewrite the following into a detailed, realistic image generation prompt.
Fix spelling mistakes. Make it visually rich. Only output the prompt.

User text: {prompt}
"""
    try:
        encoded = urllib.parse.quote(improve[:3000])
        r = requests.get(f"{TEXT_API_URL}/{encoded}", timeout=40)
        return r.text.strip()[:350]
    except:
        return prompt

# =============================
# AI CALLS
# =============================
def call_text_llm(prompt: str) -> str:
    """Call text generation API"""
    try:
        safe = urllib.parse.quote(prompt[:3500])
        r = requests.get(f"{TEXT_API_URL}/{safe}", timeout=60)
        return r.text.strip()
    except Exception as e:
        return f"❌ AI Error: {e}"


def call_image_llm(prompt: str) -> Optional[str]:
    """Generate image from prompt"""
    try:
        seed = random.randint(0, 999999)
        url = f"{IMAGE_API_URL}/{urllib.parse.quote(prompt)}?seed={seed}&width=1024&height=1024&nologo=true"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return base64.b64encode(r.content).decode()
    except:
        pass
    return None





def edit_image(image_path: str, edit_prompt: str) -> Optional[str]:
    """Edit image based on prompt using PIL"""
    try:
        img = Image.open(image_path)
        
        # Parse edit instructions
        prompt_lower = edit_prompt.lower()
        
        # Brightness adjustment
        if "bright" in prompt_lower or "dark" in prompt_lower:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            if "bright" in prompt_lower:
                img = enhancer.enhance(1.5)
            else:
                img = enhancer.enhance(0.7)
        
        # Blur
        if "blur" in prompt_lower:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Grayscale
        if "gray" in prompt_lower or "black and white" in prompt_lower:
            img = img.convert("L").convert("RGB")
        
        # Rotate
        if "rotate" in prompt_lower:
            if "90" in prompt_lower:
                img = img.rotate(90, expand=True)
            elif "180" in prompt_lower:
                img = img.rotate(180, expand=True)
            else:
                img = img.rotate(45, expand=True)
        
        # Resize
        if "resize" in prompt_lower or "smaller" in prompt_lower:
            width, height = img.size
            img = img.resize((width // 2, height // 2))
        
        # Save edited image
        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        
        return base64.b64encode(output.getvalue()).decode()
        
    except Exception as e:
        print(f"Image edit error: {e}")
        return None


def visualize_csv(file_path: str, user_prompt: str) -> Optional[str]:
    """Create visualization from CSV data"""
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        prompt_lower = user_prompt.lower()
        
        # Determine plot type
        plot_type = "line"
        if "bar" in prompt_lower:
            plot_type = "bar"
        elif "scatter" in prompt_lower:
            plot_type = "scatter"
        elif "hist" in prompt_lower:
            plot_type = "histogram"
        elif "pie" in prompt_lower:
            plot_type = "pie"
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None
        
        # Create plot based on type
        if plot_type == "histogram":
            plt.hist(df[numeric_cols[0]].dropna(), bins=20, color='#ec4899', edgecolor='white')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {numeric_cols[0]}')
            
        elif plot_type == "scatter" and len(numeric_cols) >= 2:
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                       color='#ec4899', alpha=0.6, edgecolors='white')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title(f'{numeric_cols[1]} vs {numeric_cols[0]}')
            
        elif plot_type == "bar":
            data_to_plot = df[numeric_cols[0]].head(10)
            plt.bar(range(len(data_to_plot)), data_to_plot, color='#ec4899', edgecolor='white')
            plt.xlabel('Index')
            plt.ylabel(numeric_cols[0])
            plt.title(f'Bar Chart: {numeric_cols[0]}')
            
        elif plot_type == "pie" and len(df) <= 10:
            if len(df.columns) >= 2:
                plt.pie(df[numeric_cols[0]].head(5), labels=df.iloc[:5, 0], 
                       autopct='%1.1f%%', colors=plt.cm.Purples(range(5)))
                plt.title('Pie Chart')
            
        else:  # Default line plot
            for col in numeric_cols[:3]:  # Plot up to 3 columns
                plt.plot(df[col].head(50), label=col, linewidth=2)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Data Visualization')
            plt.legend()
        
        plt.tight_layout()
        
        # Save to bytes
        output = io.BytesIO()
        plt.savefig(output, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='#0b0614', edgecolor='none')
        plt.close()
        output.seek(0)
        
        return base64.b64encode(output.getvalue()).decode()
        
    except Exception as e:
        print(f"CSV visualization error: {e}")
        return None


def edit_csv(file_path: str, user_prompt: str, original_filename: str) -> tuple:
    """
    Edit CSV file using LLM-generated pandas code.
    Returns: (success_message, new_filename, preview_html, new_file_path)
    """
    try:
        # Read current CSV
        df = pd.read_csv(file_path)
        cols = list(df.columns)
        head = df.head(5).to_string()
        
        # System prompt for code generation
        sys_msg = f"""
You are a Python data engineer. Generate ONLY the code to edit a pandas DataFrame named 'df'.
User request: {user_prompt}
Columns: {cols}
Sample data:
{head}

Rules:
1. Use only 'df' as the DataFrame name.
2. Do not include imports (pandas is available as 'pd').
3. No 'print' or extra text, ONLY the code.
4. If the user asks to add/delete rows or columns, be precise.
"""
        code = call_text_llm(sys_msg)
        
        # Security: Clean code block if AI wrapped it
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Execute code in a safe-ish scope
        loc = {'df': df, 'pd': pd}
        exec(code, {}, loc)
        df_new = loc['df']
        
        # Save edited CSV
        new_filename = f"edited_{original_filename}"
        new_path = os.path.join(UPLOAD_DIR, new_filename)
        df_new.to_csv(new_path, index=False)
        
        summary = f"✅ Edit applied successfully!\nCode executed:\n{code}"
        
        # Table Preview
        preview_html = df_new.head(15).to_html(
            index=False, 
            classes='csv-table',
            border=0,
            na_rep='—'
        )
        
        return summary, new_filename, preview_html, new_path
        
    except Exception as e:
        return f"❌ Error editing CSV: {e}", None, None, None

# =============================
# ROUTES
# =============================
@app.post("/new_chat")
def new_chat():
    global CHAT_COUNTER
    CHAT_COUNTER += 1
    cid = f"chat{CHAT_COUNTER}"
    CHATS[cid] = {"name": f"Chat {CHAT_COUNTER}", "messages": []}
    return {"chat_id": cid}


@app.post("/rename_chat")
def rename_chat(chat_id: str = Form(...), name: str = Form(...)):
    if chat_id not in CHATS:
        return JSONResponse({"error": "Chat not found"}, 404)
    CHATS[chat_id]["name"] = name.strip()
    return {"ok": True}


@app.get("/chats")
def get_chats():
    return [{"id": k, "name": v["name"]} for k, v in CHATS.items()]


@app.post("/clear_file")
def clear_file(chat_id: str = Form(...)):
    if chat_id in CHAT_FILES:
        del CHAT_FILES[chat_id]
    return {"ok": True}


@app.get("/history/{chat_id}")
def history(chat_id: str):
    return {
        "messages": CHATS.get(chat_id, {}).get("messages", []),
        "active_file": CHAT_FILES.get(chat_id, {}).get("filename")
    }


@app.get("/file/{filename}")
def get_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, 404)


@app.post("/chat")
async def chat(
    chat_id: str = Form(...),
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    if chat_id not in CHATS:
        return JSONResponse({"error": "Chat not found"}, 404)

    messages = CHATS[chat_id]["messages"]
    messages.append({"role": "user", "type": "text", "content": prompt})

    # Use persistent file if no new file is uploaded
    active_file = CHAT_FILES.get(chat_id)
    
    file_path = None
    file_text = ""
    file_type = ""

    if file:
        file_path = save_uploaded_file(file)
        file_text = extract_text_from_file(file)
        file_type = file.filename.lower()
        
        # Store for persistence
        CHAT_FILES[chat_id] = {
            "file_path": file_path,
            "file_text": file_text,
            "file_type": file_type,
            "filename": file.filename
        }

        messages.append({
            "role": "user",
            "type": "file",
            "filename": file.filename,
            "path": file_path
        })
    elif active_file:
        file_path = active_file["file_path"]
        file_text = active_file["file_text"]
        file_type = active_file["file_type"]
        print(f"📁 Reusing persistent file: {active_file['filename']}")

    # Detect intent
    # Note: intent detection uses has_file=bool(file_path) to account for persistent files
    intent = detect_intent(prompt, has_file=bool(file_path), file_type=file_type)
    
    print(f"🎯 Detected intent: {intent}")

    # ---- IMAGE GENERATION ----
    if intent == "image_gen":
        final_prompt = improve_image_prompt(prompt)
        img = call_image_llm(final_prompt)
        if img:
            msg = {"role": "assistant", "type": "image", "content": img}
            messages.append(msg)
            return msg
        else:
            msg = {"role": "assistant", "type": "text", "content": "❌ Failed to generate image"}
            messages.append(msg)
            return msg

    # ---- IMAGE EDITING ----
    elif intent == "image_edit":
        if file_path:
            edited = edit_image(file_path, prompt)
            if edited:
                msg = {"role": "assistant", "type": "image", "content": edited}
                messages.append(msg)
                return msg
            else:
                msg = {"role": "assistant", "type": "text", 
                       "content": "✨ Applied basic edits. Supported: brightness, blur, grayscale, rotate, resize"}
                messages.append(msg)
                return msg

    # ---- CSV VISUALIZATION ----
    elif intent == "csv_plot":
        if file_path and file_type.endswith(".csv"):
            plot_img = visualize_csv(file_path, prompt)
            if plot_img:
                msg = {"role": "assistant", "type": "image", "content": plot_img}
                messages.append(msg)
                
                # Add explanation
                explanation = f"📊 Created visualization based on your CSV data. Detected plot type from your request."
                msg2 = {"role": "assistant", "type": "text", "content": explanation}
                messages.append(msg2)
                return msg
            else:
                msg = {"role": "assistant", "type": "text", 
                       "content": "❌ Could not visualize CSV. Make sure it has numeric columns."}
                messages.append(msg)
                return msg
    
    # ---- CSV EDITING ----
    elif intent == "csv_edit":
        if file_path and file_type.endswith(".csv"):
            # Use name from either current upload or persistent store
            fname = file.filename if file else active_file["filename"]
            result, new_fname, preview_html, new_path = edit_csv(file_path, prompt, fname)
            
            msg = {"role": "assistant", "type": "text", "content": result}
            messages.append(msg)
            
            # CRITICAL: Persist the edited version so next prompt uses it!
            if new_fname and new_path:
                messages.append({
                    "role": "assistant",
                    "type": "file",
                    "filename": new_fname,
                    "path": new_path
                })
                
                # Re-extract text from edited file for the persistent registry
                with open(new_path, "rb") as f:
                    from fastapi import UploadFile as UF
                    mock_file = UF(filename=new_fname, file=f)
                    new_text = extract_text_from_file(mock_file)
                
                CHAT_FILES[chat_id] = {
                    "file_path": new_path,
                    "file_text": new_text,
                    "file_type": ".csv",
                    "filename": new_fname
                }

                # Add preview table if available
                if preview_html:
                    messages.append({
                        "role": "assistant",
                        "type": "table",
                        "content": preview_html
                    })
            
            return msg

    # ---- TEXT / FILE ANALYSIS ----
    else:
        # Extract CSV data if present
        csv_info = ""
        if "CSV_DATA|||" in file_text:
            parts = file_text.split("|||")
            if len(parts) >= 3:
                csv_info = f"\nCSV Columns: {parts[1]}\nSample Data:\n{parts[2]}"
        
        system_prompt = f"""
You are a smart AI assistant.

User request: {prompt}

File content: {file_text if file_text and "CSV_DATA" not in file_text else csv_info}

Respond helpfully and concisely.
"""
        reply = call_text_llm(system_prompt)
        msg = {"role": "assistant", "type": "text", "content": reply}
        messages.append(msg)
        return msg


# =============================
# UI
# =============================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>MCP AI Chat Enhanced</title>

<style>
body{
  margin:0;
  font-family:Segoe UI,sans-serif;
  background:#0b0614;
  color:#ec4899;
}

.container{
  display:flex;
  height:100vh;
}

.sidebar{
  width:240px;
  background:#140a25;
  padding:14px;
  overflow-y:auto;
}

.sidebar button{
  width:100%;
  margin-bottom:8px;
  padding:10px;
  border:none;
  border-radius:12px;
  background:#ec4899;
  color:white;
  font-weight:600;
  cursor:pointer;
  transition:all 0.3s;
}

.sidebar button:hover{
  background:#db2777;
  transform:scale(1.02);
}

.chat{
  flex:1;
  display:flex;
  flex-direction:column;
  background:#0b0614;
}

.messages{
  flex:1;
  overflow:auto;
  padding:20px;
}

.user,.ai{
  max-width:70%;
  padding:12px 16px;
  border-radius:14px;
  margin-bottom:12px;
  animation:fadeIn 0.3s;
}

@keyframes fadeIn{
  from{opacity:0;transform:translateY(10px);}
  to{opacity:1;transform:translateY(0);}
}

.user{
  margin-left:auto;
  background:#ec4899;
  color:white;
}

.ai{
  margin-right:auto;
  background:#6d28d9;
  color:white;
}

.file{
  background:#2a184a;
  padding:10px;
  border-radius:10px;
  margin-bottom:10px;
  animation:fadeIn 0.3s;
}

.file a{
  color:#ec4899;
  text-decoration:none;
}

.csv-table{
  width:100%;
  border-collapse:collapse;
  margin:10px 0;
  background:#1e1038;
  border-radius:8px;
  overflow:hidden;
}

.csv-table th{
  background:#6d28d9;
  color:white;
  padding:10px;
  text-align:left;
  font-weight:600;
}

.csv-table td{
  padding:8px 10px;
  border-bottom:1px solid #2a184a;
  color:#ec4899;
}

.csv-table tr:hover{
  background:#2a184a;
}

.table-container{
  max-width:100%;
  overflow-x:auto;
  background:#1e1038;
  border-radius:12px;
  padding:15px;
  margin:10px 0;
  animation:fadeIn 0.3s;
}

img{
  max-width:100%;
  border-radius:12px;
  box-shadow:0 4px 20px rgba(236,72,153,0.3);
}

.input{
  padding:16px;
  background:#12091f;
  border-top:1px solid #6d28d9;
}

textarea{
  width:calc(100% - 24px);
  height:70px;
  background:#1e1038;
  color:white;
  border:1px solid #6d28d9;
  border-radius:12px;
  padding:12px;
  font-size:14px;
  resize:none;
}

textarea:focus{
  outline:none;
  border-color:#ec4899;
}

.active-file-indicator {
  padding: 8px 12px;
  background: rgba(109, 40, 217, 0.2);
  border: 1px solid #6d28d9;
  border-radius: 8px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  display: none;
}

.active-file-indicator span {
  color: #fff;
}

.active-file-indicator button {
  background: none;
  border: none;
  color: #ef4444;
  cursor: pointer;
  font-weight: bold;
}

input[type="file"]{
  margin-top:8px;
  color:#ec4899;
}

.send-btn{
  margin-top:10px;
  width:100%;
  padding:12px;
  border:none;
  border-radius:12px;
  background:#ec4899;
  color:white;
  font-weight:600;
  cursor:pointer;
  font-size:15px;
  transition:all 0.3s;
}

.send-btn:hover{
  background:#db2777;
  transform:scale(1.02);
}

.chat-item{
  display:flex;
  margin-bottom:8px;
  gap:6px;
}

.chat-btn{
  flex:1;
  padding:10px;
  border:none;
  border-radius:12px;
  background:#2a184a;
  color:#ec4899;
  cursor:pointer;
  transition:all 0.3s;
}

.chat-btn:hover{
  background:#3d2766;
}

.edit-btn{
  width:40px;
  padding:10px;
  border:none;
  border-radius:12px;
  background:#6d28d9;
  color:white;
  cursor:pointer;
  transition:all 0.3s;
}

.edit-btn:hover{
  background:#5b21b6;
}

.hint{
  font-size:11px;
  color:#a855f7;
  margin-top:6px;
}
</style>
</head>

<body>
<div class="container">

  <div class="sidebar">
    <button onclick="newChat()">✨ New Chat</button>
    <div id="chatList"></div>
  </div>

  <div class="chat">
    <div class="messages" id="messages"></div>

    <div class="input">
      <div id="activeFile" class="active-file-indicator">
        <span>📎 Active file: <b id="activeName"></b></span>
        <button onclick="clearActiveFile()">Clear</button>
      </div>
      <textarea id="prompt" placeholder="Ask anything... Try:"></textarea>
      <input type="file" id="file" accept="image/*,.csv,.pdf,.txt">
      <div class="hint">💡 Upload images (edit) | CSV files (visualize/edit rows/columns) | PDFs/text</div>
      <button class="send-btn" onclick="send()">Send ✨</button>
    </div>
  </div>

</div>

<script>
let currentChat=null;

async function newChat(){
 const r=await fetch('/new_chat',{method:'POST'});
 const d=await r.json();
 currentChat=d.chat_id;
 loadChats();loadHistory();
}

async function loadChats(){
 const r=await fetch('/chats');
 const d=await r.json();
 const c=document.getElementById('chatList');
 c.innerHTML='';
 d.forEach(x=>{
   const div=document.createElement('div');
   div.className='chat-item';

   const b=document.createElement('button');
   b.className='chat-btn';
   b.innerText=x.name;
   b.onclick=()=>{currentChat=x.id;loadHistory();}

   const edit=document.createElement('button');
   edit.className='edit-btn';
   edit.innerText='✎';
   edit.onclick=async()=>{
     const newName=prompt("Rename Chat:", x.name);
     if(newName){
       const fd=new FormData();
       fd.append("chat_id",x.id);
       fd.append("name",newName);
       await fetch('/rename_chat',{method:'POST',body:fd});
       loadChats();
     }
   };

   div.appendChild(b);
   div.appendChild(edit);
   c.appendChild(div);
 });
}

async function clearActiveFile() {
  if(!currentChat) return;
  const fd = new FormData();
  fd.append("chat_id", currentChat);
  await fetch('/clear_file', {method:'POST', body:fd});
  loadHistory();
}

async function loadHistory(){
 if(!currentChat)return;
 const r=await fetch('/history/'+currentChat);
 const d=await r.json();
 
 // Update active file indicator
 const activeDiv = document.getElementById('activeFile');
 const activeName = document.getElementById('activeName');
 if(d.active_file) {
   activeDiv.style.display = 'flex';
   activeName.innerText = d.active_file;
 } else {
   activeDiv.style.display = 'none';
 }

 const m=document.getElementById('messages');
 m.innerHTML='';
 d.messages.forEach(x=>{
  const div=document.createElement('div');
  if(x.type==='image'){
    div.className='ai';
    div.innerHTML=`<img src="data:image/png;base64,${x.content}">`;
  }
  else if(x.type==='file'){
    div.className='file';
    div.innerHTML=`📎 <a href="/file/${x.filename}" target="_blank">${x.filename}</a>`;
  }
  else if(x.type==='table'){
    div.className='ai';
    const tableContainer=document.createElement('div');
    tableContainer.className='table-container';
    tableContainer.innerHTML='<strong>📋 Edited CSV Preview:</strong>'+x.content;
    div.appendChild(tableContainer);
  }
  else{
    div.className=x.role==='user'?'user':'ai';
    div.innerText=x.content;
  }
  m.appendChild(div);
 });
 m.scrollTop=m.scrollHeight;
}

async function send(){
 if(!currentChat) await newChat();
 const p=document.getElementById('prompt');
 if(!p.value.trim())return;
 
 const fd=new FormData();
 fd.append("chat_id",currentChat);
 fd.append("prompt",p.value);
 const f=document.getElementById('file');
 if(f.files[0])fd.append("file",f.files[0]);
 
 // Show loading
 const m=document.getElementById('messages');
 const loading=document.createElement('div');
 loading.className='ai';
 loading.innerText='✨ Processing...';
 m.appendChild(loading);
 m.scrollTop=m.scrollHeight;
 
 await fetch('/chat',{method:'POST',body:fd});
 p.value='';f.value='';
 loadHistory();
}

// Enter to send (Shift+Enter for newline)
document.getElementById('prompt').addEventListener('keydown',function(e){
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    send();
  }
});

loadChats();
</script>
</body>
</html>"""


# =============================
# RUN
# =============================
if __name__ == "__main__":
    print("🚀 MCP AI Chat Enhanced Starting...")
    print(f"📍 Running on http://{HOST}:{PORT}")
    print("\n✨ Features:")
    print("  • Image Generation")
    print("  • Image Editing (filters)")
    print("  • CSV Data Visualization")
    print("  • CSV Editing (add/delete rows/columns)")
    print("  • PDF & Text Analysis")
    uvicorn.run(app, host=HOST, port=PORT)