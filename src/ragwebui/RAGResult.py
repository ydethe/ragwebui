from dataclasses import dataclass
import re
from typing import List, Tuple

import markdown
from openai.types.responses import Response

from .config import config


@dataclass
class RAGResult:
    source: str
    page: int
    chunk_index: int
    ocr_used: bool
    text: str


@dataclass
class RAGRequest:
    question: str
    prompt: str
    response: Response
    sources: List[RAGResult]

    def to_html(self) -> Tuple[str, str]:
        """
        Remplace [1], [2]... par des liens HTML cliquables vers les ancres correspondantes.
        """
        html_answer = markdown.markdown(self.response.output_text)

        found_num = []
        pattern = re.compile(r"\[(\d+)\]")
        for m in pattern.finditer(html_answer):
            num = int(m.group(1))
            found_num.append(num)

        answer: str = html_answer[:]
        html_sources = ""
        found_results = []
        for num in found_num:
            res = self.sources[num - 1]
            list_sources = [s.source for s in found_results]
            if res.source in list_sources:
                result_index = list_sources.index(res.source)
                res = self.sources[result_index]
            else:
                found_results.append(res)

            answer = answer.replace(
                f"[{num}]", f'<a href="#src{num}" style="text-decoration:none;">[{num}]</a>'
            )
            html_sources += f"""
                    <div id="src{num}" style='margin-top:20px; padding:10px; border:1px solid #ccc;'>
                        <b>[{num}] {res.source}</b><br>
                        <iframe src="{config.DAV_ROOT}/{res.source}" width="100%" height="300px"></iframe>
                    </div>
                    """

        return answer, html_sources
