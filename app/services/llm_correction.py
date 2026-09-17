"""
LLM-based role refinement using Groq.

Key fixes vs previous version:
- Primary model changed to llama3-70b-8192 (fast, handles large context, no JSON issues)
- Large transcripts (>30 segments) are sampled: send representative 30 segments,
  infer the rest from alternating pattern — avoids token limit failures
- Fallback chain updated to reliable models only
"""
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Fast, reliable models currently available on Groq — ordered by preference
_PRIMARY_MODEL   = "openai/gpt-oss-20b"
_FALLBACK_MODELS = ["openai/gpt-oss-120b", "groq/compound-mini", "groq/compound"]

# Max segments to send in one LLM call — beyond this we sample
# gpt-oss-20b handles ~20 segments reliably without hitting token limits
_MAX_SEGMENTS_PER_CALL = 20


def refine_roles_with_llm(segments: List[Dict]) -> List[Dict]:
    """
    Refine Doctor/Patient role assignments using Groq LLM.
    For large transcripts, samples key segments and infers the rest.
    """
    if not segments:
        return segments

    print(f"  Sending {len(segments)} segments to Groq LLM for role refinement...")

    try:
        if len(segments) <= _MAX_SEGMENTS_PER_CALL:
            # Small transcript — send everything
            refined_roles = _call_groq(segments)
        else:
            # Large transcript — sample intelligently
            refined_roles = _call_groq_sampled(segments)

        if not refined_roles or len(refined_roles) != len(segments):
            print("  ⚠️  LLM refinement failed, keeping diarization roles")
            for seg in segments:
                seg["role_source"] = "diarization_only"
            return segments

        changes = 0
        for i, seg in enumerate(segments):
            if seg["role"] != refined_roles[i]:
                seg["role"] = refined_roles[i]
                seg["role_source"] = "llm_corrected"
                changes += 1
            else:
                seg["role_source"] = "llm_confirmed"

        print(f"  ✓ LLM refined: {changes} change(s), {len(segments)-changes} confirmed")
        return segments

    except Exception as e:
        print(f"  ⚠️  LLM refinement error: {e}")
        for seg in segments:
            seg["role_source"] = "diarization_only"
        return segments


def _call_groq_sampled(segments: List[Dict]) -> List[str]:
    """
    For large transcripts: send first 15 + last 15 segments to establish
    the role pattern, then apply that pattern to all segments.
    """
    n = len(segments)
    half = _MAX_SEGMENTS_PER_CALL // 2

    # Take first half + last half as representative sample
    sample_indices = list(range(half)) + list(range(n - half, n))
    sample_segs = [segments[i] for i in sample_indices]

    print(f"  [LLM] Large transcript ({n} segs) — sampling {len(sample_segs)} representative segments")

    sample_roles = _call_groq(sample_segs)
    if not sample_roles or len(sample_roles) != len(sample_segs):
        return []

    # Map sample indices back to their roles
    sample_role_map = {sample_indices[i]: sample_roles[i] for i in range(len(sample_indices))}

    # Determine dominant pattern from sample
    # Find the role of SPEAKER_00 equivalent from first few segments
    first_role = sample_roles[0]  # role of segment 0
    second_role = "Patient" if first_role == "Doctor" else "Doctor"

    # Build full roles list using LLM results for sampled segments,
    # and alternating heuristic based on speaker changes for the rest
    full_roles = []
    prev_speaker = None
    current_role = first_role

    for i, seg in enumerate(segments):
        if i in sample_role_map:
            # Use LLM result directly
            full_roles.append(sample_role_map[i])
            current_role = sample_role_map[i]
            prev_speaker = seg.get("speaker", "UNKNOWN")
        else:
            # Infer from speaker change pattern
            this_speaker = seg.get("speaker", "UNKNOWN")
            if this_speaker != prev_speaker and this_speaker != "UNKNOWN":
                current_role = "Patient" if current_role == "Doctor" else "Doctor"
            full_roles.append(current_role)
            prev_speaker = this_speaker

    return full_roles


def _call_groq(segments: List[Dict]) -> List[str]:
    """
    Call Groq API with a list of segments (max _MAX_SEGMENTS_PER_CALL).
    Returns list of role strings or empty list on failure.
    """
    try:
        from groq import Groq
    except ImportError:
        print("  ⚠️  groq package not installed")
        return []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("  ⚠️  GROQ_API_KEY not configured")
        return []

    # Build compact context — just index, role, and text (no speaker labels)
    lines = []
    for i, seg in enumerate(segments):
        lines.append(f'{i}|{seg["role"]}|{seg["text"][:120]}')
    context = "\n".join(lines)

    prompt = f"""You are a clinical conversation analyst. Fix Doctor/Patient role assignments.

Format: index|CurrentRole|text
Return JSON: {{"roles": ["Doctor","Patient",...]}} with exactly {len(segments)} labels.

Rules:
- Doctor greets first ("Good morning", "How can I help")
- Doctor asks diagnostic questions, gives instructions, prescribes
- Patient describes symptoms, answers questions, says "thank you doctor"
- Roles should alternate naturally

Transcript ({len(segments)} lines):
{context}

Return ONLY the JSON object, no explanation."""

    models_to_try = [_PRIMARY_MODEL] + _FALLBACK_MODELS

    client = Groq(api_key=api_key)

    for model in models_to_try:
        try:
            print(f"  Trying model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "Return only valid JSON with a 'roles' array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max(512, len(segments) * 12),
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                print(f"  ✗ {model}: empty response")
                continue

            content = content.strip()

            # Strip markdown fences if present
            if "```" in content:
                for part in content.split("```"):
                    part = part.strip().lstrip("json").strip()
                    if part.startswith("{"):
                        content = part
                        break

            parsed = json.loads(content)

            # Find roles array — try multiple key names
            roles = None
            for key in ["roles", "labels", "result", "data", "output"]:
                if key in parsed and isinstance(parsed[key], list):
                    roles = parsed[key]
                    break

            if roles is None or len(roles) != len(segments):
                print(f"  ✗ {model}: expected {len(segments)} roles, got "
                      f"{len(roles) if roles else 'None'}")
                continue

            # Normalise role strings (handle "doctor", "DOCTOR", "Doctor")
            normalised = []
            for r in roles:
                r_str = str(r).strip().lower()
                if r_str in ("doctor", "dr", "physician"):
                    normalised.append("Doctor")
                elif r_str in ("patient", "pt"):
                    normalised.append("Patient")
                else:
                    normalised.append(r)  # will fail validation below

            invalid = [r for r in normalised if r not in ("Doctor", "Patient")]
            if invalid:
                print(f"  ✗ {model}: invalid roles {invalid[:3]}")
                continue

            print(f"  ✓ LLM success with {model}")
            return normalised

        except Exception as e:
            print(f"  ✗ {model}: {type(e).__name__}: {str(e)[:120]}")
            if hasattr(e, "response"):
                print(f"    HTTP {getattr(e.response, 'status_code', '?')}: "
                      f"{getattr(e.response, 'text', '')[:150]}")
            continue

    print("  ✗ All models failed")
    return []


def validate_groq_config() -> str:
    """Validate Groq config at startup. Returns warning string or empty string."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return ""

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models_response = client.models.list()
        available = [m.id for m in models_response.data]

        if _PRIMARY_MODEL in available:
            print(f"✓ Groq configured — primary model: {_PRIMARY_MODEL}")
            return ""
        else:
            return (f"⚠️  Primary model '{_PRIMARY_MODEL}' not found. "
                    f"Available: {', '.join(available[:5])}")
    except Exception as e:
        return f"⚠️  Groq validation error: {e}"
