"""
LLM-based text cleanup for garbled Whisper segments (opt-in, targeted).
"""
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def detect_garbled_segments(segments: List[Dict]) -> List[Dict]:
    """
    Detect segments that are likely garbled/hallucinated by Whisper.
    
    Detection heuristics:
    - Single word repeated 3+ times consecutively
    - Low unique-word ratio (<0.4 for segments with 5+ words)
    
    Args:
        segments: List of segments with 'text' field
        
    Returns:
        Same segments with 'likely_garbled' and 'garbled_reason' fields added
    """
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        words = text.split()
        word_count = len(words)
        
        # Heuristic 1: Check for repeated words
        if word_count >= 3:
            # Look for any word repeated 3+ times consecutively
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    seg["likely_garbled"] = True
                    seg["garbled_reason"] = f"Repeated word pattern: '{words[i]}'"
                    break
        
        # Heuristic 2: Low unique-word ratio
        if not seg.get("likely_garbled") and word_count >= 5:
            unique_words = len(set(w.lower() for w in words))
            unique_ratio = unique_words / word_count
            
            if unique_ratio < 0.4:
                seg["likely_garbled"] = True
                seg["garbled_reason"] = f"Low unique-word ratio: {unique_ratio:.2f}"
    
    # Count flagged segments
    garbled_count = sum(1 for seg in segments if seg.get("likely_garbled"))
    if garbled_count > 0:
        print(f"  ⚠️  {garbled_count} segment(s) flagged as likely garbled")
        for seg in segments:
            if seg.get("likely_garbled"):
                print(f"    [{seg['start']:.2f}s] Reason: {seg['garbled_reason']}")
                print(f"    Text: {seg['text'][:100]}")
    
    return segments


def cleanup_garbled_segments(segments: List[Dict]) -> List[Dict]:
    """
    Use Groq LLM to clean garbled segments while preserving original text.
    
    Only processes segments flagged as 'likely_garbled'.
    Original text always preserved in 'text' field.
    Cleaned version stored in 'text_cleaned' field.
    
    Args:
        segments: List of segments, some with 'likely_garbled' flag
        
    Returns:
        Segments with 'text_cleaned' and 'text_cleanup_applied' fields added
    """
    garbled_indices = [
        i for i, seg in enumerate(segments)
        if seg.get("likely_garbled")
    ]
    
    if not garbled_indices:
        return segments
    
    print(f"  Sending {len(garbled_indices)} garbled segment(s) to LLM for cleanup...")
    
    try:
        from groq import Groq
    except ImportError:
        print("  ⚠️  groq package not installed, skipping cleanup")
        return segments
    
    # Get configuration
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("  ⚠️  GROQ_API_KEY not configured, skipping cleanup")
        return segments
    
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    
    client = Groq(api_key=api_key)
    
    # Process each garbled segment with surrounding context
    cleaned_count = 0
    
    for idx in garbled_indices:
        try:
            seg = segments[idx]
            
            # Gather context: current + 1-2 surrounding segments
            context_segments = []
            
            # Previous segment
            if idx > 0:
                context_segments.append({
                    "position": "before",
                    "text": segments[idx - 1]["text"]
                })
            
            # Current (garbled) segment
            context_segments.append({
                "position": "current",
                "text": seg["text"],
                "garbled_reason": seg.get("garbled_reason", "unknown")
            })
            
            # Next segment
            if idx < len(segments) - 1:
                context_segments.append({
                    "position": "after",
                    "text": segments[idx + 1]["text"]
                })
            
            # Build prompt
            context_lines = []
            for ctx in context_segments:
                if ctx["position"] == "current":
                    context_lines.append(f"[GARBLED SEGMENT - {ctx['garbled_reason']}]: {ctx['text']}")
                else:
                    context_lines.append(f"[{ctx['position']}]: {ctx['text']}")
            
            context_str = "\n".join(context_lines)
            
            prompt = f"""You are cleaning up a garbled medical transcription segment. The segment was flagged because it contains repeated words or low information content, likely due to Whisper ASR hallucination.

Context from transcript:
{context_str}

Task: Provide a plausible cleaned version of the GARBLED SEGMENT based on conversational context. If the segment is genuinely unintelligible, return "[unintelligible]".

Return ONLY a JSON object in this format:
{{"cleaned_text": "your cleaned version here"}}"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at cleaning garbled medical transcripts. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content or content.strip() == "":
                print(f"    Segment {idx}: LLM returned empty response, keeping original")
                continue
            
            # Parse response
            try:
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                # Try stripping markdown fences
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{"):
                            try:
                                parsed = json.loads(part)
                                break
                            except:
                                continue
                    else:
                        print(f"    Segment {idx}: Failed to parse JSON, keeping original")
                        continue
                else:
                    print(f"    Segment {idx}: Failed to parse JSON, keeping original")
                    continue
            
            # Extract cleaned text
            cleaned_text = None
            if "cleaned_text" in parsed:
                cleaned_text = parsed["cleaned_text"]
            elif "text" in parsed:
                cleaned_text = parsed["text"]
            elif "result" in parsed:
                cleaned_text = parsed["result"]
            
            if cleaned_text and cleaned_text.strip():
                # NEVER overwrite original text - store in separate field
                seg["text_cleaned"] = cleaned_text.strip()
                seg["text_cleanup_applied"] = True
                cleaned_count += 1
                print(f"    Segment {idx}: Cleaned successfully")
                print(f"      Original: {seg['text'][:80]}")
                print(f"      Cleaned:  {cleaned_text[:80]}")
            else:
                print(f"    Segment {idx}: No cleaned text found, keeping original")
        
        except Exception as e:
            print(f"    Segment {idx}: Cleanup error: {e}, keeping original")
            continue
    
    if cleaned_count > 0:
        print(f"  ✓ Cleaned {cleaned_count} segment(s). Original text preserved in 'text' field.")
    
    return segments
