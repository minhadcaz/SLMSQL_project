"""Conservative lexical selection with foreign-key closure."""
import re
import unicodedata


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


class SchemaSelector:
    def select(self, question: str, schema: dict) -> dict:
        aliases = {"students": ["sinh vien"], "classes": ["lop"],
                   "departments": ["khoa"], "subjects": ["mon hoc", "tin chi"],
                   "enrollments": ["diem", "dang ky", "hoc ky"]}
        text = normalize(question)
        selected = {name for name in schema if any(
            re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
            for term in [normalize(name)] + aliases.get(name, [])
        )}
        if not selected:
            return schema
        # Connected FK closure retains bridge tables; may conservatively keep all tables.
        changed = True
        while changed:
            before = len(selected)
            for name, table in schema.items():
                for fk in table["foreign_keys"]:
                    if name in selected or fk["table"] in selected:
                        selected.update({name, fk["table"]})
            changed = len(selected) != before
        return {name: table for name, table in schema.items() if name in selected}
