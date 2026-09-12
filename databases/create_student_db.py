"""Create deterministic, fictional data; existing files are verified, never overwritten."""
import argparse
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "databases/student_management.db"

DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE classes (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
 department_id INTEGER NOT NULL REFERENCES departments(id));
CREATE TABLE students (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL,
 birth_year INTEGER NOT NULL, gender TEXT NOT NULL,
 class_id INTEGER NOT NULL REFERENCES classes(id));
CREATE TABLE subjects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
 credits INTEGER NOT NULL CHECK(credits > 0));
CREATE TABLE enrollments (id INTEGER PRIMARY KEY,
 student_id INTEGER NOT NULL REFERENCES students(id),
 subject_id INTEGER NOT NULL REFERENCES subjects(id), semester TEXT NOT NULL,
 score REAL CHECK(score BETWEEN 0 AND 10), UNIQUE(student_id, subject_id, semester));
CREATE INDEX idx_students_class ON students(class_id);
CREATE INDEX idx_enrollments_student ON enrollments(student_id);
CREATE INDEX idx_enrollments_subject ON enrollments(subject_id);
"""


def verify_gold(path: Path) -> int:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    checked = 0
    try:
        for filename in ("demo_questions.json", "evaluation_questions.json"):
            items = json.loads((ROOT / "data" / filename).read_text(encoding="utf-8"))
            for item in items:
                try:
                    connection.execute(item["gold_sql"]).fetchall()
                except sqlite3.Error as error:
                    raise ValueError(f"Gold SQL không chạy được: {item['id']}: {error}") from error
                checked += 1
    finally:
        connection.close()
    return checked


def create_database(path: str | Path = DEFAULT_DB) -> Path:
    path = Path(path).resolve()
    if path.exists():
        verify_gold(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent startup from overwriting an existing database.
    with path.open("xb"):
        connection = sqlite3.connect(path)
        try:
            connection.executescript(DDL)
            connection.executemany("INSERT INTO departments VALUES (?, ?)", [
                (1, "Công nghệ thông tin"), (2, "Kinh tế"), (3, "Ngoại ngữ"), (4, "Thiết kế")
            ])
            connection.executemany("INSERT INTO classes VALUES (?, ?, ?)", [
                (1, "CNTT01", 1), (2, "CNTT02", 1), (3, "KT01", 2),
                (4, "NN01", 3), (5, "TK01", 4), (6, "CNTT03", 1)
            ])
            names = ["Nguyễn Minh An", "Trần Thu Bình", "Lê Quốc Cường", "Phạm Ngọc Dung",
                     "Hoàng Việt Dũng", "Vũ Thanh Hà", "Đặng Gia Huy", "Bùi Mỹ Linh",
                     "Đỗ Đức Nam", "Hồ Bảo Ngọc", "Ngô Anh Phúc", "Dương Mai Phương",
                     "Lý Minh Quân", "Trịnh Thu Trang", "Đinh Hải Yến", "Phan Tuấn Kiệt"]
            students = [(i, name, 2002 + i % 5, "Nữ" if i % 2 == 0 else "Nam",
                         1 + (i - 1) // 4) for i, name in enumerate(names, 1)]
            connection.executemany("INSERT INTO students VALUES (?, ?, ?, ?, ?)", students)
            connection.executemany("INSERT INTO subjects VALUES (?, ?, ?)", [
                (1, "Cơ sở dữ liệu", 3), (2, "Lập trình Python", 4), (3, "Toán rời rạc", 3),
                (4, "Kinh tế vi mô", 3), (5, "Tiếng Anh", 2), (6, "Thiết kế cơ bản", 2),
                (7, "Học máy", 4), (8, "Đồ án tốt nghiệp", 6)
            ])
            enrollments = []
            for student_id in range(1, 16):
                for subject_id in range(1, 6):
                    if (student_id + subject_id) % 4 == 0:
                        continue
                    score = None if (student_id + subject_id) % 7 == 0 else (
                        10.0 if student_id == 1 else float(3 + (student_id * 3 + subject_id) % 8)
                    )
                    enrollments.append((len(enrollments) + 1, student_id, subject_id,
                                        "2024-1" if subject_id < 4 else "2024-2", score))
            connection.executemany("INSERT INTO enrollments VALUES (?, ?, ?, ?, ?)", enrollments)
            connection.commit()
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Dữ liệu vi phạm foreign key.")
        finally:
            connection.close()
    verify_gold(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    output = create_database(args.path)
    print(f"Database ready: {output}; checked {verify_gold(output)} gold queries.")
