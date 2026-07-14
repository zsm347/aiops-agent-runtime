from pathlib import Path


class PromptRegistry:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir

    def load(self, version: str) -> str:
        safe_version = version.strip()
        path = self.prompt_dir / f"{safe_version}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

