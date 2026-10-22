"""Static checks only; external Groq, embedding and TTS services are not exercised."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RepositoryContractTests(unittest.TestCase):
    def test_python_entrypoints_parse(self):
        for name in ("app.py", "speaking_bot.py"):
            with self.subTest(name=name):
                ast.parse((ROOT / name).read_text(encoding="utf-8"))

    def test_knowledge_folder_and_loader_match(self):
        self.assertTrue((ROOT / "Doc").is_dir())
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("DirectoryLoader('Doc')", source)
        self.assertFalse("Peoject_2/Doc" in source)

    def test_no_literal_groq_token_in_main_app(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("gsk_", source)

if __name__ == "__main__":
    unittest.main()
