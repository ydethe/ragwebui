import re
import time
from typing import List, Tuple

from qdrant_client import QdrantClient, models
import markdown
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai.types.responses import Response

from . import logger
from .config import config
from .RAGResult import RAGResult


class RAGRequestor(object):
    def __init__(self):
        # --- Configuration ---
        self.openai = OpenAI(
            # This is the default and can be omitted
            api_key=config.OPENAI_API_KEY,
        )

        # --- Initialisation ---
        self.encoder = SentenceTransformer(
            config.EMBEDDING_MODEL, trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE
        )
        # vector_size = self.encoder.get_sentence_embedding_dimension()
        self.qdrant = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

    def search(self, query_vector) -> List[RAGResult]:
        t0 = time.time()
        hits = self.qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=config.QDRANT_QUERY_LIMIT,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            ci_min = hit.payload["chunk_index"] - config.RAG_AUGMENTATION
            ci_max = hit.payload["chunk_index"] + config.RAG_AUGMENTATION
            if ci_min < 0:
                ci_max = ci_max - ci_min
                ci_min = 0
            recs, ids = self.qdrant.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=config.QDRANT_QUERY_LIMIT,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=hit.payload["source"]),
                        ),
                        models.FieldCondition(
                            key="chunk_index",
                            range=models.Range(gte=ci_min, lte=ci_max),
                        ),
                    ]
                ),
            )

            txt = ""
            for ci in range(ci_min, ci_max + 1):
                short_list = [r for r in recs if r.payload["chunk_index"] == ci]
                if len(short_list) == 0:
                    continue
                r = short_list[0]
                if len(txt) == 0:
                    txt = r.payload["text"]
                else:
                    txt += r.payload["text"][config.CHUNK_OVERLAP :]

            res = hit.payload.copy()
            res["text"] = txt
            res = RAGResult(
                source=hit.payload["source"],
                chunk_index=hit.payload["chunk_index"],
                ocr_used=hit.payload["ocr_used"],
                text=txt,
            )
            results.append(res)

        elapsed = time.time() - t0
        logger.info(f"Vector search completed in {elapsed:.1f} s")

        return results

    def link_citations(self, results: List[RAGResult], html_answer: str) -> Tuple[str, str]:
        """
        Remplace [1], [2]... par des liens HTML cliquables vers les ancres correspondantes.
        """
        found_num = []
        pattern = re.compile(r"\[(\d+)\]")
        for m in pattern.finditer(html_answer):
            num = int(m.group(1))
            found_num.append(num)

        answer: str = html_answer[:]
        html_sources = ""
        found_results = []
        for num in found_num:
            res = results[num - 1]
            list_sources = [s.source for s in found_results]
            if res.source in list_sources:
                result_index = list_sources.index(res.source)
                res = results[result_index]
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

    def rag_with_anchored_sources(self, message, chat_history):
        query_vector = self.encoder.encode(message).tolist()

        results = self.search(query_vector)

        context_chunks = []

        for i, hit in enumerate(results):
            num = i + 1
            context_chunks.append(f"[{num}]<{hit.source}>\n{hit.text}")

        context = "\n\n".join(context_chunks)

        prompt = f"""
        ### Contraintes
        - Utilisez uniquement les informations fournies dans la section Contexte, y compris le nom du document.
        - N'utilisez pas de connaissances extérieures ou d'hypothèses.
        - Citez la source de chaque information en utilisant le format [numero_du_fichier].
        - Si la réponse ne figure pas dans le contexte, répondez : « La réponse n'est pas disponible dans les documents fournis. »

        ### Étapes
        1. Lisez attentivement le contexte.
        2. Repérez à la fois dans le contexte et dans nom_du_fichier les éléments pertinents pour répondre à la question de l'utilisateur.
        3. Formulez une réponse claire et concise uniquement à partir de ces éléments.
        4. Intégrez les citations de sources directement dans la réponse avec le format [numero_du_fichier].

        ### Tâche
        Répondre à la question de l'utilisateur à l'aide du contenu fourni dans le contexte et du nom du fichier entre crochets.

        ### Assemblage
        Contexte :
        {context}

        Question :
        {message}

        ### Réponse
        """

        t0 = time.time()
        response: Response = self.openai.responses.create(
            input=prompt,
            model=config.OPEN_MODEL_PREF,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        logger.info(f"LLM response got in {elapsed:.1f} s")

        html_answer = markdown.markdown(response.output_text)
        answer, html_sources = self.link_citations(results, html_answer)

        chat_history = [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": message},
        ] + chat_history

        return "", chat_history, html_sources
