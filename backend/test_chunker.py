import re

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size: return [text]
    chunks = []
    start = 0
    text_len = len(text)
    
    count = 0
    while start < text_len:
        count += 1
        if count > 10:
            print("Infinite loop detected!")
            break
        end = min(start + chunk_size, text_len)
        if end < text_len:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        print(f"start={start}, end={end}, chunk='{chunk}'")
        
        start = end - chunk_overlap
        if start <= 0 or start >= text_len:
            print("breaking due to start check")
            break
            
        if count > 10: break
    return chunks

test_text = "x" * 100 + " " + "y" * 600
_split_text(test_text, 512, 64)
