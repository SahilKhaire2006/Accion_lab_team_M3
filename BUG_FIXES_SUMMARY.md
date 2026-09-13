# Bug Fixes and Feature Implementation Summary

## Date: 2026-09-08

## Overview
This document summarizes the fixes for Bug A (merge collapsing speakers), Bug B (Groq JSON failures), and the new garbled text cleanup feature.

---

## Bug A: Merge Step Collapsing All Segments to One Speaker

### Problem
- Diarization correctly detected 2 speakers (SPEAKER_00, SPEAKER_01, 28 turns)
- But merge assigned all 17 Whisper segments to SPEAKER_01
- LLM had to correct most roles, indicating merge logic failure

### Root Cause Analysis
- Insufficient debug logging made bug invisible
- Overlap calculation may have edge cases with timing precision
- No validation that all detected speakers appear in merge output

### Fixes Implemented

#### 1. Enhanced Debug Logging (`app/services/merge.py`)
- **Before merge**: Print all diarization turns with durations
- **During merge**: Show detailed overlap analysis for each segment
  - Lists ALL turns that overlap each segment
  - Shows overlap duration for each turn
  - Identifies which turn "won" the assignment
- **After merge**: Comprehensive sanity checks
  - Counts unique speakers in diarization vs merge output
  - Identifies missing speakers
  - Shows segment distribution per speaker
  - Lists unmatched diarization turns

#### 2. Improved Overlap Calculation
- Added explicit sorting of diarization turns by start time
- More detailed logging per-segment showing:
  - Segment text preview (first 80 chars)
  - All overlapping turns with overlap values
  - Final speaker assignment decision

#### 3. Post-Merge Validation
- Automatic detection of speaker collapse
- Warning: "⚠️ Merge collapsed N speakers into M — possible merge bug!"
- Shows which speakers went missing
- Identifies which diarization turns were never matched

### Expected Outcome
- Debug logs will now clearly show WHERE the bug occurs:
  - If segments aren't overlapping SPEAKER_00 turns → timing issue
  - If overlaps are calculated but SPEAKER_01 always wins → comparison bug
  - If certain turns are never matched → iteration/filtering bug

---

## Bug B: Groq JSON Failures with Empty Response

### Problem
- `openai/gpt-oss-20b` and `openai/gpt-oss-120b` failing with empty `failed_generation`
- Only `groq/compound` succeeds (slower agentic model, not ideal for classification)
- Failure appeared after updating to object-wrapped JSON format

### Root Cause Analysis
- Models may struggle with larger prompts (17+ segments)
- Insufficient error logging hid actual failure details
- JSON parsing was too strict

### Fixes Implemented

#### 1. Enhanced Error Logging (`app/services/llm_correction.py`)
- Log `finish_reason` for every response
- Log full response object when content is empty
- Show raw content preview (first 150 chars) for every attempt
- Display HTTP response status and body for API errors
- Better exception type identification

#### 2. Improved Prompt
- Added explicit segment count to prompt
- Emphasized required format multiple times
- Added "IMPORTANT" section reinforcing JSON-only output
- Clarified no markdown or explanations

#### 3. More Tolerant JSON Parsing
- Check for empty/null content before parsing
- Strip markdown fences (```json blocks)
- Try multiple alternate keys: `roles`, `labels`, `result`, `data`, `output`, `predictions`
- Show available keys when expected key is missing
- Show actual parsed content on validation failures

#### 4. Token Length Warning
- Estimate prompt tokens (length / 4)
- Warn if >3000 tokens (may exceed model limits)

#### 5. Better Validation Messages
- Show received roles count vs expected
- List invalid roles if found
- Show full parsed object on structure errors

### Expected Outcome
- `gpt-oss-20b` should now succeed on 15+ segment transcripts
- If it still fails, logs will show:
  - Whether response is empty or has content
  - What JSON structure was actually returned
  - Which validation check failed

---

## New Feature: Garbled Text Cleanup

### Purpose
Detect and clean Whisper hallucinations (e.g., "Don't. Don't. Don't. Don't.")

### Implementation (`app/services/text_cleanup.py`)

#### Detection Heuristics
1. **Repeated Word Pattern**: Any single word repeated 3+ times consecutively
2. **Low Unique-Word Ratio**: <0.4 unique words for segments with 5+ words

#### Cleanup Process
- **Step 2.5** in pipeline (between transcription and diarization)
- Only processes flagged segments (opt-in, minimal usage)
- Sends segment + 1-2 surrounding segments to Groq for context
- **NEVER overwrites original text**:
  - Original stays in `text` field
  - Cleaned version in `text_cleaned` field
  - Adds `text_cleanup_applied: true` flag
  - Adds `likely_garbled: true` and `garbled_reason` to flagged segments

#### Logging
- Warns when segments are flagged: "⚠️ N segment(s) flagged as garbled"
- Shows reason and text preview for each flagged segment
- Reports cleanup success/failure per segment
- Preserves original text even on cleanup failure

#### UI Integration (`static/transcript.html`)
- Shows `text_cleaned` if available
- Green "CLEANED" badge next to cleaned text
- Hover over badge or text to see tooltip with original
- Tooltip labeled "Original Whisper Output:"
- Clear visual indication that text was modified

### Safety Guarantees
1. Original text ALWAYS preserved in `text` field
2. Only runs on flagged segments (not blanket rewrite)
3. Graceful failure - keeps original if cleanup fails
4. User can always see what was actually transcribed

---

## Testing Checklist

### Bug A Testing
- [ ] Run new recording and check console output
- [ ] Verify diarization turns are printed before merge
- [ ] Verify overlap analysis shows for each segment
- [ ] Check if warning appears when speakers collapse
- [ ] Identify root cause from detailed logs

### Bug B Testing
- [ ] Test with 15+ segment transcript
- [ ] Check if `gpt-oss-20b` succeeds without falling back
- [ ] Verify error logs show response content on failure
- [ ] Confirm finish_reason is logged

### Text Cleanup Testing
- [ ] Create test audio with repeated words
- [ ] Verify detection flags the garbled segment
- [ ] Check cleanup runs and preserves original
- [ ] Test UI shows "CLEANED" badge
- [ ] Verify hover shows original text in tooltip
- [ ] Confirm normal segments are not processed

---

## Files Modified

### Core Fixes
- `app/services/merge.py` - Enhanced debug logging, better overlap calculation
- `app/services/llm_correction.py` - Improved error handling, better JSON parsing
- `app/services/text_cleanup.py` - **NEW** - Garbled text detection and cleanup
- `app/api/live_routes.py` - Integrated text cleanup as Step 2.5

### UI Updates
- `static/transcript.html` - Added cleaned text display with hover tooltip

---

## Next Steps After Testing

1. **If Bug A is fixed**: Remove or reduce debug verbosity
2. **If Bug A persists**: Analyze detailed logs to identify exact root cause
3. **If Bug B is fixed**: Keep current error logging for future debugging
4. **If Bug B persists**: Consider alternate models or prompt formats
5. **Text Cleanup**: Tune detection heuristics based on real-world performance

---

## Regression Prevention

### Bug A
- Added automated sanity check in merge code
- Warning will catch this bug automatically in future

### Bug B  
- Enhanced logging will make any JSON failures visible
- Won't silently fall through to last model anymore

### Text Cleanup
- Conservative detection heuristics minimize false positives
- Original text preservation prevents data loss
