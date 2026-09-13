"""
LLM-based role refinement using Groq (mandatory quality assurance step).
"""
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def refine_roles_with_llm(segments: List[Dict]) -> List[Dict]:
    """
    Send all segments to Groq LLM for role refinement and error correction.
    This is a mandatory quality assurance step that runs after diarization.
    
    Args:
        segments: List of segments with 'start', 'end', 'text', 'speaker', 'role' fields
        
    Returns:
        List of segments with LLM-refined roles
    """
    if not segments:
        return segments
    
    print("  Sending transcript to Groq LLM for role refinement...")
    
    try:
        refined_roles = call_groq_for_refinement(segments)
        
        if not refined_roles or len(refined_roles) != len(segments):
            print("  ⚠️  LLM refinement failed, keeping original roles")
            for seg in segments:
                seg["role_source"] = "diarization_only"
            return segments
        
        # Apply LLM-refined roles
        changes_made = 0
        for i, seg in enumerate(segments):
            original_role = seg["role"]
            refined_role = refined_roles[i]
            
            if original_role != refined_role:
                changes_made += 1
                seg["role"] = refined_role
                seg["role_source"] = "llm_corrected"
            else:
                seg["role_source"] = "llm_confirmed"
        
        print(f"  ✓ LLM refined roles: {changes_made} change(s), {len(segments) - changes_made} confirmed")
        return segments
        
    except Exception as e:
        print(f"  ⚠️  LLM refinement error: {e}")
        # Keep original roles on error
        for seg in segments:
            seg["role_source"] = "diarization_only"
        return segments


def call_groq_for_refinement(segments: List[Dict]) -> List[str]:
    """
    Call Groq API to refine and correct role assignments.
    
    Args:
        segments: List of segments with text, speaker, and current role
        
    Returns:
        List of refined roles (one per segment, in order)
    """
    try:
        from groq import Groq
    except ImportError:
        print("  ⚠️  groq package not installed")
        return []
    
    # Get configuration
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("  ⚠️  GROQ_API_KEY not configured")
        return []
    
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    fallback_models_str = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,groq/compound")
    fallback_models = [m.strip() for m in fallback_models_str.split(",") if m.strip()]
    
    # Build detailed context with speaker labels from diarization
    context_lines = []
    for i, seg in enumerate(segments):
        speaker_label = seg.get("speaker", "UNKNOWN")
        current_role = seg["role"]
        text = seg["text"]
        context_lines.append(f"{i}: [Speaker: {speaker_label}, Role: {current_role}] {text}")
    
    context = "\n".join(context_lines)
    
    prompt = f"""You are an expert at analyzing doctor-patient clinical conversations. You will receive a transcript with automatic speaker diarization and role assignments that may contain errors.

Your task: Review the conversation and return a JSON object with a "roles" array containing {len(segments)} corrected role labels ("Doctor" or "Patient"), one for each line in order.

Rules:
1. The first speaker in a clinical conversation is typically the doctor greeting the patient
2. A patient would NOT say "Good morning, what's the problem?" - that's the doctor
3. A patient would NOT give medical instructions or prescribe medicine - that's the doctor  
4. A doctor would NOT say "Thank you, doctor" - that's the patient
5. Natural turn-taking: roles should alternate in most conversations
6. Consider the semantic meaning of each utterance, not just the diarization labels
7. If diarization is correct, keep the original role

Transcript with diarization ({len(segments)} lines):
{context}

IMPORTANT: Return ONLY a valid JSON object. Do not include any explanation or markdown. The object must have exactly {len(segments)} role labels.

Required format:
{{"roles": ["Doctor", "Patient", "Doctor", ...]}}"""
    
    client = Groq(api_key=api_key)
    
    # Try primary model first, then fallbacks
    models_to_try = [model] + fallback_models
    
    for attempt_model in models_to_try:
        try:
            # Check if prompt might be too long
            prompt_tokens_estimate = len(prompt) / 4  # Rough estimate
            if prompt_tokens_estimate > 3000:
                print(f"  ⚠️  Warning: Large prompt ({prompt_tokens_estimate:.0f} tokens est.) may exceed limits for {attempt_model}")
            
            print(f"  Trying model: {attempt_model}")
            
            response = client.chat.completions.create(
                model=attempt_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert clinical conversation analyst. You must return a valid JSON object with a 'roles' array containing role labels."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Enable strict JSON mode
            )
            
            # Log raw response for debugging
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            print(f"  Model {attempt_model} finish_reason: {finish_reason}")
            
            if not content or content.strip() == "":
                print(f"  ⚠️  Model {attempt_model} returned empty content")
                print(f"    Full response object: {response.model_dump_json()}")
                continue
            
            content = content.strip()
            print(f"  Model {attempt_model} raw response preview: {content[:150]}...")
            
            # Parse JSON object with tolerant parsing
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Model {attempt_model} JSON parse error: {e}")
                print(f"    Raw content (first 500 chars): {content[:500]}")
                # Try to extract JSON from markdown fences
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{"):
                            try:
                                parsed = json.loads(part)
                                print(f"  ✓ Extracted JSON from markdown fence")
                                break
                            except:
                                continue
                    else:
                        print(f"  ✗ Could not extract valid JSON from markdown")
                        continue
                else:
                    continue
            
            # Extract roles array from object
            if not isinstance(parsed, dict):
                print(f"  ✗ Model {attempt_model} returned non-object: {type(parsed)}")
                print(f"    Content: {parsed}")
                continue
            
            # Look for roles array
            roles = None
            if "roles" in parsed:
                roles = parsed["roles"]
            else:
                # Try common alternate keys
                alternate_keys = ["labels", "result", "data", "output", "classifications", "predictions"]
                for key in alternate_keys:
                    if key in parsed and isinstance(parsed[key], list):
                        roles = parsed[key]
                        print(f"  ℹ️  Found roles in alternate key: '{key}'")
                        break
            
            if roles is None:
                print(f"  ✗ Model {attempt_model} missing 'roles' key")
                print(f"    Available keys: {list(parsed.keys())}")
                print(f"    Full parsed object: {parsed}")
                continue
            
            if not isinstance(roles, list):
                print(f"  ✗ Model {attempt_model} 'roles' is not a list: {type(roles)}")
                continue
            
            if len(roles) != len(segments):
                print(f"  ✗ Model {attempt_model} returned {len(roles)} roles, expected {len(segments)}")
                print(f"    Roles received: {roles}")
                continue
            
            # Validate all roles
            valid_roles = {"Doctor", "Patient"}
            invalid_roles = [r for r in roles if r not in valid_roles]
            if invalid_roles:
                print(f"  ✗ Model {attempt_model} returned invalid roles: {invalid_roles}")
                print(f"    All roles: {roles}")
                continue
            
            print(f"  ✓ LLM refinement successful with model: {attempt_model}")
            return roles
            
        except Exception as e:
            print(f"  ✗ Model {attempt_model} error: {type(e).__name__}: {e}")
            
            # Show more details for certain error types
            if hasattr(e, 'response'):
                print(f"    Response status: {getattr(e.response, 'status_code', 'N/A')}")
                print(f"    Response body: {getattr(e.response, 'text', 'N/A')[:200]}")
            
            if attempt_model == models_to_try[-1]:
                print(f"  ✗ All {len(models_to_try)} model(s) failed")
                return []
            
            print(f"  → Trying next fallback model...")
            continue
    
    return []


def validate_groq_config() -> str:
    """
    Validate Groq configuration at startup.
    
    Returns:
        Warning message if validation fails, empty string if OK
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return ""  # Not configured, skip validation
    
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # List available models
        models_response = client.models.list()
        available_model_ids = [m.id for m in models_response.data]
        
        if model not in available_model_ids:
            available_str = ", ".join(available_model_ids[:5])
            return f"⚠️  Configured GROQ_MODEL '{model}' not in available models. Available: {available_str}"
        
        print(f"✓ Groq LLM configured: {model}")
        return ""
        
    except Exception as e:
        return f"⚠️  Groq validation error: {e}"
