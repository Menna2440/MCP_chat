import base64
import random
import urllib.parse
import requests
import os
import io
from typing import Optional
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Alternative faster image generation APIs (as backup)
ALTERNATIVE_IMAGE_APIS = [
    "https://image.pollinations.ai/prompt",  # Primary
    "https://pollinations.ai/p",  # Alternative endpoint
]

HOST = "127.0.0.1"
PORT = 8001

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MCP AI Chat")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# MEMORY
# =============================
CHATS = {}
CHAT_COUNTER = 0
CHAT_FILES = {}

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
            file.file.seek(0)
            reader = PdfReader(file.file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return f"PDF_CONTENT|||{text[:6000]}"
        
        if name.endswith(".csv"):
            file.file.seek(0)
            raw = file.file.read().decode("utf-8", errors="ignore")
            lines = raw.splitlines()
            header = lines[0] if lines else ""
            sample = "\n".join(lines[1:21])
            return f"CSV|||{header}|||{sample}"
        
        if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            return "IMAGE_FILE"
        
        if name.endswith((".xls", ".xlsx")):
            file.file.seek(0)
            df = pd.read_excel(file.file)
            preview = df.head(20).to_string()
            return f"EXCEL|||{preview}"
        
        if name.endswith(".json"):
            file.file.seek(0)
            data = json.load(file.file)
            return f"JSON|||{json.dumps(data, indent=2)[:4000]}"
        
        if name.endswith((".txt", ".md", ".py", ".js", ".html", ".css")):
            file.file.seek(0)
            content = file.file.read().decode("utf-8", errors="ignore")
            return f"TEXT_FILE|||{content[:6000]}"
        
        file.file.seek(0)
        return file.file.read().decode("utf-8", errors="ignore")[:4000]
    except Exception as e:
        return f"ERROR_READING_FILE: {str(e)}"

# =============================
# INTENT DETECTION
# =============================
def detect_intent(prompt: str, has_file: bool, file_type: str) -> str:
    q = prompt.lower()

    # ✅ 1. Image EDIT has absolute priority if an image is uploaded
    if has_file and file_type.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        if any(w in q for w in [
            "edit", "modify", "change", "filter",
            "blur", "bright", "dark",
            "rotate", "gray", "grey", "grayscale",
            "black and white"
        ]):
            return "image_edit"

    # ✅ 2. Image GENERATION (only if no image file)
    if not has_file and any(w in q for w in ["create", "generate", "draw", "make", "design"]) and \
       any(w in q for w in ["image", "picture", "photo", "pic", "art"]):
        return "image_gen"

    # ✅ 3. CSV / Excel handling
    if has_file and (file_type.endswith(".csv") or file_type.endswith((".xls", ".xlsx"))):
        if any(w in q for w in ["plot", "graph", "chart", "visualize", "show"]):
            return "csv_plot"
        if any(w in q for w in ["add", "delete", "remove", "filter", "sort", "update", "edit", "modify"]):
            return "csv_edit"

    return "text"


# =============================
# AI CALLS
# =============================
def call_text_llm(prompt: str) -> str:
    try:
        safe = urllib.parse.quote(prompt[:4000])
        r = requests.get(f"{TEXT_API_URL}/{safe}", timeout=90)
        r.raise_for_status()
        return r.text.strip()
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

def call_image_llm(prompt: str) -> str:
    """Generate image URL - The actual generation happens when browser loads the URL"""
    seed = random.randint(0, 999999)
    safe_prompt = urllib.parse.quote(prompt)
    
    # Using standard parameters that work reliably
    return f"{IMAGE_API_URL}/{safe_prompt}?seed={seed}&width=1024&height=1024&nologo=true"

def edit_image(path: str, prompt: str) -> Optional[str]:
    try:
        img = Image.open(path)
        q = prompt.lower()
        
        if "bright" in q or "lighter" in q:
            from PIL import ImageEnhance
            factor = 1.8 if "very" in q else 1.4
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        if "dark" in q or "dim" in q:
            from PIL import ImageEnhance
            factor = 0.4 if "very" in q else 0.6
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        if "blur" in q:
            from PIL import ImageFilter
            radius = 10 if "very" in q else 5
            img = img.filter(ImageFilter.GaussianBlur(radius))
        
        if "sharp" in q:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SHARPEN)
        
        if "gray" in q or "grey" in q or "grayscale" in q:
            img = img.convert("L").convert("RGB")
        
        if "rotate" in q:
            angle = 90
            if "180" in q:
                angle = 180
            elif "270" in q:
                angle = 270
            elif "45" in q:
                angle = 45
            img = img.rotate(angle, expand=True)
        
        if "flip" in q:
            if "horizontal" in q:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        output = io.BytesIO()
        img.save(output, format='PNG')
        return base64.b64encode(output.getvalue()).decode()
    except Exception as e:
        print(f"Image edit error: {e}")
        return None

def visualize_csv(path: str, prompt: str) -> Optional[str]:
    try:
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        df.columns = df.columns.str.strip()
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric:
            return None
        
        plt.figure(figsize=(12, 7))
        plt.style.use('dark_background')
        q = prompt.lower()
        
        if "bar" in q:
            x_vals = range(len(df[numeric[0]].head(15)))
            plt.bar(x_vals, df[numeric[0]].head(15), color='#ec4899')
            plt.xlabel('Index')
            plt.ylabel(numeric[0])
            plt.title(f'Bar Chart: {numeric[0]}')
        
        elif "scatter" in q and len(numeric) >= 2:
            plt.scatter(df[numeric[0]], df[numeric[1]], color='#ec4899', alpha=0.6, s=50)
            plt.xlabel(numeric[0])
            plt.ylabel(numeric[1])
            plt.title(f'Scatter: {numeric[0]} vs {numeric[1]}')
        
        elif "pie" in q:
            plt.pie(df[numeric[0]].head(10), labels=df.iloc[:10, 0] if len(df.columns) > 1 else None, autopct='%1.1f%%')
            plt.title(f'Pie Chart: {numeric[0]}')
        
        else:
            for col in numeric[:3]:
                plt.plot(df[col].head(100), label=col, linewidth=2)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Line Chart')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        output = io.BytesIO()
        plt.savefig(output, format='png', dpi=120, facecolor='#0b0614')
        plt.close()
        return base64.b64encode(output.getvalue()).decode()
    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close()
        return None

def edit_csv(path: str, prompt: str, fname: str) -> tuple:
    try:
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        
        code_prompt = f"""Generate Python pandas code to edit DataFrame 'df'.
User request: {prompt}
Available columns: {list(df.columns)}
Current shape: {df.shape}

Rules:
- Only use pandas operations
- Result must be stored back in 'df'
- No explanations, only executable code
- Use proper pandas syntax

Example: df = df[df['column'] > 5]"""
        
        code = call_text_llm(code_prompt)
        code = code.replace("```python", "").replace("```", "").strip()
        
        loc = {'df': df, 'pd': pd}
        exec(code, {}, loc)
        df_new = loc['df']
        
        new_path = os.path.join(UPLOAD_DIR, f"edited_{fname}")
        
        if fname.endswith('.csv'):
            df_new.to_csv(new_path, index=False)
        else:
            df_new.to_excel(new_path, index=False)
        
        preview = df_new.head(20).to_html(index=False, classes='csv-table', border=0)
        
        return f"✅ Data edited successfully!\n\nCode executed:\n{code}\n\nNew shape: {df_new.shape}", f"edited_{fname}", preview, new_path
    except Exception as e:
        return f"❌ Error editing data: {str(e)}", None, None, None

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
    if chat_id in CHATS:
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
    return FileResponse(path) if os.path.exists(path) else JSONResponse({"error": "Not found"}, 404)

@app.post("/chat")
async def chat(chat_id: str = Form(...), prompt: str = Form(...), file: Optional[UploadFile] = File(None)):
    if chat_id not in CHATS:
        return JSONResponse({"error": "Chat not found"}, 404)
    
    messages = CHATS[chat_id]["messages"]
    messages.append({"role": "user", "type": "text", "content": prompt})
    
    active = CHAT_FILES.get(chat_id)
    file_path = file_text = file_type = None
    
    if file:
        file_path = save_uploaded_file(file)
        file_text = extract_text_from_file(file)
        file_type = file.filename.lower()
        CHAT_FILES[chat_id] = {
            "file_path": file_path, 
            "file_text": file_text, 
            "file_type": file_type, 
            "filename": file.filename
        }
        messages.append({"role": "user", "type": "file", "filename": file.filename, "path": file_path})
    elif active:
        file_path = active["file_path"]
        file_text = active["file_text"]
        file_type = active["file_type"]
    
    intent = detect_intent(prompt, bool(file_path), file_type or "")
    
    if intent == "image_gen":
        # Add loading message first
        loading_msg = {"role": "assistant", "type": "text", "content": "🎨 Generating your image... Please wait 10-20 seconds."}
        messages.append(loading_msg)
        
        # Generate image URL (instant, actual generation happens when browser loads it)
        img_url = call_image_llm(prompt)
        msg = {"role": "assistant", "type": "image", "content": img_url, "is_url": True}
        messages.append(msg)
        return msg
    
    if intent == "image_edit" and file_path:
        edited = edit_image(file_path, prompt)
        if edited:
            msg = {"role": "assistant", "type": "image", "content": edited}
            messages.append(msg)
            messages.append({"role": "assistant", "type": "text", "content": "✨ Image edited successfully!"})
            return msg
        else:
            msg = {"role": "assistant", "type": "text", "content": "❌ Could not edit image. Please try a different edit command."}
            messages.append(msg)
            return msg
    
    if intent == "csv_plot" and file_path:
        plot = visualize_csv(file_path, prompt)
        if plot:
            msg = {"role": "assistant", "type": "image", "content": plot}
            messages.append(msg)
            messages.append({"role": "assistant", "type": "text", "content": "📊 Visualization created!"})
            return msg
        else:
            msg = {"role": "assistant", "type": "text", "content": "❌ Could not create visualization. Make sure the file has numeric data."}
            messages.append(msg)
            return msg
    
    if intent == "csv_edit" and file_path:
        fname = file.filename if file else active["filename"]
        result, new_fname, preview, new_path = edit_csv(file_path, prompt, fname)
        msg = {"role": "assistant", "type": "text", "content": result}
        messages.append(msg)
        
        if new_path:
            messages.append({"role": "assistant", "type": "file", "filename": new_fname, "path": new_path})
            if preview:
                messages.append({"role": "assistant", "type": "table", "content": preview})
            
            CHAT_FILES[chat_id] = {
                "file_path": new_path,
                "file_text": file_text,
                "file_type": file_type,
                "filename": new_fname
            }
        return msg
    
    context_info = ""
    if file_text:
        if "CSV|||" in file_text:
            parts = file_text.split("|||")
            if len(parts) >= 3:
                context_info = f"\n\n[CSV File Context]\nColumns: {parts[1]}\nSample data:\n{parts[2]}"
        elif "EXCEL|||" in file_text:
            context_info = f"\n\n[Excel File Context]\n{file_text.split('|||')[1]}"
        elif "PDF_CONTENT|||" in file_text:
            context_info = f"\n\n[PDF File Context]\n{file_text.split('|||')[1]}"
        elif "JSON|||" in file_text:
            context_info = f"\n\n[JSON File Context]\n{file_text.split('|||')[1]}"
        elif "TEXT_FILE|||" in file_text:
            context_info = f"\n\n[Text File Context]\n{file_text.split('|||')[1]}"
        else:
            context_info = f"\n\n[File Context]\n{file_text[:2000]}"
    
    full_prompt = f"""You are a helpful AI assistant. Answer the user's question clearly and concisely.

User: {prompt}{context_info}

Provide a helpful, accurate response. If discussing file contents, be specific and reference the data."""
    
    reply = call_text_llm(full_prompt)
    msg = {"role": "assistant", "type": "text", "content": reply}
    messages.append(msg)
    return msg

# =============================
# UI
# =============================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCP AI Chat - Multi-functional Assistant</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#0b0614;color:#ec4899;height:100vh;overflow:hidden}
.container{display:flex;height:100vh}
.sidebar{width:260px;background:#140a25;padding:16px;overflow-y:auto;border-right:2px solid #6d28d9}
.sidebar button{width:100%;margin-bottom:10px;padding:12px;border:none;border-radius:12px;background:linear-gradient(135deg,#ec4899,#db2777);color:#fff;font-weight:700;cursor:pointer;transition:all 0.3s;font-size:14px}
.sidebar button:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(236,72,153,0.4)}
.sidebar .chat-item{background:#1e1038;margin-bottom:8px;padding:10px;border-radius:10px;cursor:pointer;transition:all 0.2s;border:2px solid transparent;display:flex;align-items:center;gap:8px}
.sidebar .chat-item:hover{border-color:#ec4899;background:#2a184a}
.sidebar .chat-item.active{border-color:#ec4899;background:#2a184a}
.chat-item span{flex:1;cursor:pointer}
.edit-chat-btn i{
    font-size: 10px;
}

.edit-chat-btn svg{
    width: 10px;
    height: 10px;
}

.chat{flex:1;display:flex;flex-direction:column}
.header{background:#140a25;padding:16px;border-bottom:2px solid #6d28d9;text-align:center}
.header h1{font-size:24px;background:linear-gradient(135deg,#ec4899,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.messages{flex:1;overflow-y:auto;padding:24px;background:#0b0614}
.message{display:flex;margin-bottom:16px;animation:slideIn 0.3s}
@keyframes slideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.user{justify-content:flex-end}
.ai{justify-content:flex-start}
.message-content{max-width:75%;padding:14px 18px;border-radius:16px;word-wrap:break-word}
.user .message-content{background:linear-gradient(135deg,#ec4899,#db2777);color:#fff;border-bottom-right-radius:4px}
.ai .message-content{background:#6d28d9;color:#fff;border-bottom-left-radius:4px}
.file-indicator{background:#2a184a;padding:12px;border-radius:12px;margin:8px 0;border:2px solid #6d28d9}
.file-indicator a{color:#ec4899;text-decoration:none;font-weight:600}
.file-indicator a:hover{text-decoration:underline}
.csv-table{width:100%;border-collapse:collapse;margin:12px 0;background:#1e1038;border-radius:8px;overflow:hidden}
.csv-table th{background:#6d28d9;color:#fff;padding:12px;text-align:left;font-weight:600}
.csv-table td{padding:10px;border-bottom:1px solid #2a184a;color:#ec4899}
.csv-table tr:last-child td{border-bottom:none}
img{max-width:100%;border-radius:14px;margin:8px 0;box-shadow:0 4px 16px rgba(0,0,0,0.3)}
.input-area{padding:20px;background:#12091f;border-top:2px solid #6d28d9}
.active-file-banner{padding:10px 14px;background:rgba(109,40,217,.25);border:2px solid #6d28d9;border-radius:10px;margin-bottom:12px;display:none;align-items:center;justify-content:space-between;font-size:13px}
.active-file-banner button{background:#ef4444;border:none;color:#fff;padding:6px 12px;border-radius:6px;cursor:pointer;font-weight:600}
.active-file-banner button:hover{background:#dc2626}
textarea{width:100%;min-height:80px;background:#1e1038;color:#fff;border:2px solid #6d28d9;border-radius:12px;padding:14px;resize:vertical;font-size:15px;font-family:inherit}
textarea:focus{outline:none;border-color:#ec4899}
.file-upload{margin:12px 0}
.file-upload input{background:#1e1038;color:#ec4899;padding:10px;border:2px dashed #6d28d9;border-radius:10px;width:100%;cursor:pointer}
.send-btn{width:100%;padding:14px;margin-top:12px;border:none;border-radius:12px;background:linear-gradient(135deg,#ec4899,#db2777);color:#fff;font-weight:700;cursor:pointer;font-size:16px;transition:all 0.3s}
.send-btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(236,72,153,0.5)}
.send-btn:active{transform:translateY(0)}
.loading{display:inline-block;width:20px;height:20px;border:3px solid rgba(255,255,255,.3);border-radius:50%;border-top-color:#fff;animation:spin 1s ease-in-out infinite}
@keyframes spin{to{transform:rotate(360deg)}}
::-webkit-scrollbar{width:10px}
::-webkit-scrollbar-track{background:#140a25}
::-webkit-scrollbar-thumb{background:#6d28d9;border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:#ec4899}
</style>
</head>
<body>
<div class="container">
<div class="sidebar">
<button onclick="newChat()">✨ New Chat</button>
<div id="chatList"></div>
</div>
<div class="chat">
<div class="header">
<h1>🤖 MCP AI Assistant</h1>
<p style="font-size:12px;color:#a855f7;margin-top:4px">Chat • Generate Images • Edit Files • Analyze Data</p>
</div>
<div class="messages" id="messages"></div>
<div class="input-area">
<div id="activeFile" class="active-file-banner">
<span>📎 <b id="activeName"></b></span>
<button onclick="clearFile()">✕ Clear</button>
</div>
<textarea id="prompt" placeholder="Ask anything... Upload files, generate images, edit data, or just chat!"></textarea>
<div class="file-upload">
<input type="file" id="file" accept="*">
</div>
<button class="send-btn" onclick="send()">Send Message ✨</button>
</div>
</div>
</div>
<script>
let currentChat=null;
let isGenerating=false;

async function newChat(){
 const r=await fetch('/new_chat',{method:'POST'});
 const d=await r.json();
 currentChat=d.chat_id;
 loadChats();
 loadHistory();
}

async function loadChats(){
 const r=await fetch('/chats');
 const d=await r.json();
 const c=document.getElementById('chatList');
 c.innerHTML='';
 d.forEach(x=>{
   const div=document.createElement('div');
   div.className='chat-item'+(x.id===currentChat?' active':'');

   const span=document.createElement('span');
   span.innerText=x.name;
   span.onclick=()=>{
     currentChat=x.id;
     loadHistory();
     loadChats();
   };

   const editBtn=document.createElement('button');
   editBtn.innerHTML='✏️';
   editBtn.className='edit-chat-btn';
   editBtn.onclick=(e)=>{
     e.stopPropagation();
     const newName=prompt("Rename chat:", x.name);
     if(newName && newName.trim()){
       renameChat(x.id, newName.trim());
     }
   };

   div.appendChild(span);
   div.appendChild(editBtn);
   c.appendChild(div);
 });
}

async function renameChat(chatId, newName){
 const fd=new FormData();
 fd.append("chat_id", chatId);
 fd.append("name", newName);
 await fetch('/rename_chat', {method: 'POST', body: fd});
 loadChats();
}

async function clearFile(){
 if(!currentChat)return;
 const fd=new FormData();
 fd.append("chat_id",currentChat);
 await fetch('/clear_file',{method:'POST',body:fd});
 loadHistory();
}

async function loadHistory(){
 if(!currentChat)return;
 const r=await fetch('/history/'+currentChat);
 const d=await r.json();
 const af=document.getElementById('activeFile');
 const an=document.getElementById('activeName');
 
 if(d.active_file){
   af.style.display='flex';
   an.innerText=d.active_file;
 }else{
   af.style.display='none';
 }
 
 const m=document.getElementById('messages');
 m.innerHTML='';
 
 d.messages.forEach(x=>{
  const msgDiv=document.createElement('div');
  msgDiv.className='message '+(x.role==='user'?'user':'ai');
  
  const content=document.createElement('div');
  content.className='message-content';
  
  if(x.type==='image'){
    if(x.is_url){
      // Create container for loading state
      const imgContainer=document.createElement('div');
      imgContainer.style.minHeight='300px';
      imgContainer.style.display='flex';
      imgContainer.style.alignItems='center';
      imgContainer.style.justifyContent='center';
      
      // Add loading text
      const loadingText=document.createElement('p');
      loadingText.innerHTML='🎨 Loading image...<br><span class="loading"></span>';
      loadingText.style.textAlign='center';
      imgContainer.appendChild(loadingText);
      content.appendChild(imgContainer);
      
      // Create image
      const img=document.createElement('img');
      img.src=x.content;
      img.alt='Generated Image';
      img.style.display='none';
      
      img.onload=function(){
        imgContainer.innerHTML='';
        img.style.display='block';
        imgContainer.appendChild(img);
      };
      
      img.onerror=function(){
        imgContainer.innerHTML='<p>❌ Image failed to load. Try again or check your internet connection.</p>';
      };
      
    }else{
      const img=document.createElement('img');
      img.src='data:image/png;base64,'+x.content;
      img.alt='Edited Image';
      content.appendChild(img);
    }
  }else if(x.type==='file'){
    content.className='file-indicator';
    content.innerHTML=`📎 <a href="/file/${x.filename}" target="_blank" download>${x.filename}</a>`;
  }else if(x.type==='table'){
    content.innerHTML=x.content;
  }else{
    content.innerText=x.content;
  }
  
  msgDiv.appendChild(content);
  m.appendChild(msgDiv);
 });
 
 m.scrollTop=m.scrollHeight;
}

async function send(){
 if(isGenerating)return;
 
 if(!currentChat)await newChat();
 
 const p=document.getElementById('prompt');
 const text=p.value.trim();
 
 if(!text)return;
 
 isGenerating=true;
 const btn=document.querySelector('.send-btn');
 const origText=btn.innerHTML;
 btn.innerHTML='<span class="loading"></span> Sending...';
 btn.disabled=true;
 
 const fd=new FormData();
 fd.append("chat_id",currentChat);
 fd.append("prompt",text);
 
 const f=document.getElementById('file');
 if(f.files[0]){
   fd.append("file",f.files[0]);
 }
 
 try{
   await fetch('/chat',{method:'POST',body:fd});
   p.value='';
   f.value='';
   await loadHistory();
 }catch(e){
   alert('Error: '+e.message);
 }finally{
   isGenerating=false;
   btn.innerHTML=origText;
   btn.disabled=false;
 }
}

document.getElementById('prompt').addEventListener('keydown',e=>{
 if(e.key==='Enter'&&!e.shiftKey){
   e.preventDefault();
   send();
 }
});

(async()=>{
 await loadChats();
 const chats=await(await fetch('/chats')).json();
 if(chats.length===0){
   await newChat();
 }
})();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MCP AI Chat - Multi-functional Assistant")
    print("=" * 60)
    print(f"📍 Server: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
