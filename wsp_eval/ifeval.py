"""Instruction-following checks for the `if_guard` task (from google/IFEval).

Part 3 uses `if_guard` as a guard: a graft that lifts GSM8K but breaks the
host's general instruction following gets penalized. Each IFEval prompt comes
with machine-checkable instructions ("no commas", "at least 300 words", ...).
A prompt counts as passed only if every one of its instructions is followed
(IFEval's strict prompt-level accuracy).

The checks follow the rules of the official IFEval checker
(google-research/instruction_following_eval), rewritten in plain Python.
Two differences, both documented here:
  * sentence counting uses a regex split on . ! ? instead of nltk punkt;
  * language:response_language needs a language detector, so the 31 prompts
    that use it are left out (510 of 541 prompts remain).
"""
import json
import re

UNSUPPORTED = ("language:response_language",)


def _compare(count, relation, target):
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    raise ValueError("unknown relation " + str(relation))


def count_words(text):
    return len(re.findall(r"\w+", text))


def count_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    count = 0
    for part in parts:
        if re.search(r"\w", part):
            count += 1
    return count


def check_instruction(instruction_id, kwargs, response, prompt):
    """True if `response` follows one instruction."""
    args = {}
    for key, value in kwargs.items():
        if value is not None:
            args[key] = value
    text = response

    if instruction_id == "punctuation:no_comma":
        return "," not in text

    if instruction_id == "length_constraints:number_words":
        return _compare(count_words(text), args["relation"], args["num_words"])

    if instruction_id == "length_constraints:number_sentences":
        return _compare(count_sentences(text), args["relation"], args["num_sentences"])

    if instruction_id == "keywords:forbidden_words":
        for word in args["forbidden_words"]:
            if re.search(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE):
                return False
        return True

    if instruction_id == "keywords:existence":
        for word in args["keywords"]:
            if not re.search(re.escape(word), text, flags=re.IGNORECASE):
                return False
        return True

    if instruction_id == "keywords:frequency":
        count = len(re.findall(re.escape(args["keyword"]), text, flags=re.IGNORECASE))
        return _compare(count, args["relation"], args["frequency"])

    if instruction_id == "keywords:letter_frequency":
        count = text.lower().count(args["letter"].lower())
        return _compare(count, args["let_relation"], args["let_frequency"])

    if instruction_id == "detectable_format:number_highlighted_sections":
        count = 0
        for highlight in re.findall(r"\*[^\n\*]*\*", text):
            if highlight.strip("*").strip():
                count += 1
        for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
            if highlight.removeprefix("**").removesuffix("**").strip():
                count += 1
        return count >= args["num_highlights"]

    if instruction_id == "combination:repeat_prompt":
        return text.strip().lower().startswith(args["prompt_to_repeat"].strip().lower())

    if instruction_id == "startend:quotation":
        stripped = text.strip()
        return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'

    if instruction_id == "change_case:english_lowercase":
        return text == text.lower()

    if instruction_id == "change_case:english_capital":
        return text == text.upper()

    if instruction_id == "change_case:capital_word_frequency":
        count = 0
        for word in re.findall(r"\b\w+\b", text):
            if word.isupper():
                count += 1
        return _compare(count, args["capital_relation"], args["capital_frequency"])

    if instruction_id == "detectable_format:title":
        for title in re.findall(r"<<[^\n]+>>", text):
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False

    if instruction_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
        bullets += re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
        return len(bullets) == args["num_bullets"]

    if instruction_id == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.*?\]", text)) >= args["num_placeholders"]

    if instruction_id == "length_constraints:number_paragraphs":
        paragraphs = re.split(r"\s?\*\*\*\s?", text)
        count = len(paragraphs)
        for index in range(len(paragraphs)):
            if not paragraphs[index].strip():
                if index == 0 or index == len(paragraphs) - 1:
                    count -= 1
                else:
                    return False
        return count == args["num_paragraphs"]

    if instruction_id == "startend:end_checker":
        return text.strip().strip('"').lower().endswith(args["end_phrase"].strip().lower())

    if instruction_id == "detectable_content:postscript":
        marker = args["postscript_marker"].strip()
        if marker == "P.P.S":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pattern = r"\s*p\.\s?s\..*$"
        else:
            pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
        return len(re.findall(pattern, text.lower(), flags=re.MULTILINE)) > 0

    if instruction_id == "combination:two_responses":
        parts = text.split("******")
        answers = []
        for index in range(len(parts)):
            if not parts[index].strip():
                if index != 0 and index != len(parts) - 1:
                    return False
            else:
                answers.append(parts[index].strip())
        return len(answers) == 2 and answers[0] != answers[1]

    if instruction_id == "detectable_format:json_format":
        value = text.strip()
        value = value.removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
        value = value.removeprefix("```").removesuffix("```").strip()
        try:
            json.loads(value)
        except ValueError:
            return False
        return True

    if instruction_id == "detectable_format:multiple_sections":
        splitter = args["section_spliter"].strip()
        sections = re.split(r"\s?" + re.escape(splitter) + r"\s?\d+\s?", text)
        return len(sections) - 1 >= args["num_sections"]

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = re.split(r"\n\n", text)
        count = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                count -= 1
        nth = args["nth_paragraph"]
        if nth > count or count != args["num_paragraphs"]:
            return False
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
        first = paragraph.split()[0].strip()
        first = re.sub(r"[^\w]", "", first).lower()
        return first == args["first_word"].lower()

    if instruction_id == "detectable_format:constrained_response":
        for option in ("My answer is yes.", "My answer is no.", "My answer is maybe."):
            if option in text:
                return True
        return False

    raise ValueError("unsupported IFEval instruction: " + instruction_id)


def follows_all(item, response):
    for instruction_id, kwargs in zip(item["instruction_id_list"], item["kwargs"]):
        if not check_instruction(instruction_id, kwargs, response, item["prompt_text"]):
            return False
    return True
