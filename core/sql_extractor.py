"""Extract answer containers without silently dropping multiple statements."""
import re


def extract_sql(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    # Incomplete reasoning is not executable SQL.
    if re.search(r"<think>", clean, re.I):
        clean = re.split(r"<think>", clean, flags=re.I)[0]
    answers = re.findall(r"<answer>(.*?)</answer>", clean, flags=re.I | re.S)
    if answers:
        if len(answers) != 1:
            raise ValueError("Model trả nhiều đáp án SQL.")
        clean = answers[0].strip()
    fences = re.findall(r"```(?:sql|sqlite)?\s*\n?(.*?)```", clean, flags=re.I | re.S)
    if fences:
        if len(fences) != 1:
            raise ValueError("Model trả nhiều khối SQL.")
        clean = fences[0].strip()
    if not clean:
        raise ValueError("Không tìm thấy SQL hoàn chỉnh trong đầu ra model.")
    if not re.match(
        r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
        clean, re.I,
    ):
        raise ValueError("Đầu ra không bắt đầu bằng SQL; không hiển thị nội dung model dạng tự do.")
    return clean
