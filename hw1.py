#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
        max_tokens=8192,
        max_retries=2,
        timeout=60,
    )

    system_text = (
        "You are an expert receipt parser for Hong Kong supermarket receipts. "
        "You must return ONLY a JSON object, no prose, no markdown fences."
    )

    human_text = """Analyze the receipt image carefully.

Extract exactly these three things:

1. final_payment:
   The amount actually charged to the customer after any ROUNDING line.
   On HK receipts this is often the line labelled "OCTOPUS", "EPS",
   "CASH", "VISA", "TOTAL", or the last monetary line of the bill.
   It already INCLUDES the ROUNDING adjustment. Do NOT add it again.

2. subtotal:
   The printed line labelled "SUBTOTAL", before ROUNDING. This amount
   may already include discounts. Do not replace it with a pre-discount
   item total. If absent, derive it only when the receipt makes it clear.

3. discounts:
   A list of every discount / promotion / coupon / "x% OFF" line on the
   receipt. Store each as a POSITIVE number (drop the minus sign).
   If there are no discounts, return an empty list [].
   Scan the entire item section from top to bottom for negative monetary
   entries, including Chinese-labelled markdowns such as 包裝變形
   (damaged packaging), clearance, member pricing, and app offers.
   A price reduction counts even without the words discount or Save.
   Preserve separate occurrences of equal-valued discounts on different
   items. Read the actual amount in the price column; promotional wording
   may describe a per-offer saving rather than the total applied saving.
   Exclude ROUNDING and do not count a repeated savings summary again.
   For cash payments, distinguish the actual charge from cash tendered
   and change. Do not use a remaining Octopus balance as payment.

Return ONLY this JSON:
{{
  "final_payment": <number>,
  "subtotal": <number>,
  "discounts": [<number>, ...]
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", [
            {"type": "text", "text": human_text},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])

    chain = prompt | llm
    return chain
    ### END YOUR CODE HERE


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from decimal import Decimal
    from langchain_core.output_parsers import JsonOutputParser

    parser = JsonOutputParser()

    def money(value):
        if value is None or isinstance(value, bool):
            raise ValueError("Missing monetary amount")
        amount = Decimal(str(value).replace(",", ""))
        if not amount.is_finite():
            raise ValueError("Non-finite monetary amount")
        if amount != amount.quantize(Decimal("0.01")):
            raise ValueError("Invalid monetary precision")
        return amount

    total_final = Decimal("0")
    total_no_discount = Decimal("0")
    failed = []

    for img_path in images:
        for attempt in range(3):
            try:
                response = chain.invoke({"image_url": image_data_url(img_path)})
                text = response_text(response)
                metadata = getattr(response, "response_metadata", {}) or {}
                print(f"  {img_path.name}: attempt={attempt + 1}, "
                      f"finish_reason={metadata.get('finish_reason', 'unknown')}, "
                      f"response_chars={len(text)}")
                if not text:
                    raise ValueError("Model returned empty answer content")
                result = parser.parse(text)
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object")
                final = money(result.get("final_payment"))
                subtotal = money(result.get("subtotal"))
                discounts = result.get("discounts")
                if not isinstance(discounts, list):
                    raise ValueError("Missing discount list")
                discount_sum = sum((abs(money(d)) for d in discounts), Decimal("0"))
                if final < 0 or subtotal < 0:
                    raise ValueError("Unexpected negative total")
                total_final += final
                total_no_discount += subtotal + discount_sum
                print(f"  {img_path.name}: final={final:.2f}, "
                      f"no_discount={subtotal + discount_sum:.2f}")
                break
            except Exception as exc:
                print(f"[warn] {img_path.name}, attempt {attempt + 1}: "
                      f"{type(exc).__name__}")
        else:
            failed.append(img_path.name)

    if failed:
        print("[error] Unresolved receipts: " + ", ".join(failed))
        error = "ERROR: receipt extraction incomplete; see terminal output"
        return {QUERY_1: error, QUERY_2: error}

    return {
        QUERY_1: f"HK${total_final:.2f}",
        QUERY_2: f"HK${total_no_discount:.2f}",
    }
    ### END YOUR CODE HERE


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
