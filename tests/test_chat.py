import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.documents import Document
from langchain_groq import ChatGroq

import app as chat_app

PASSAGE = "The community library opens at nine on weekdays."
SOURCE = "library-hours.txt"


class ChatTests(unittest.TestCase):
    def setUp(self):
        chat_app.get_chat_model.cache_clear()
        self.addCleanup(chat_app.get_chat_model.cache_clear)
        self.enterContext(patch.dict(os.environ, {
            "GROQ_API_KEY": "mock-api-key",
            "GROQ_MODEL": "",
            "ENABLE_TTS": "0",
        }))
        self.enterContext(patch.dict(chat_app.app.config, {"TESTING": True}))
        self.client = chat_app.app.test_client()
        self.requests = []
        self.answer = {
            "answer": "It opens at nine on weekdays.",
            "citations": [{"passage_id": 1, "quote": PASSAGE}],
        }
        transport = httpx.MockTransport(self.complete)
        self.http_client = self.enterContext(httpx.Client(transport=transport))
        self.http_async_client = httpx.AsyncClient(transport=transport)
        self.addCleanup(lambda: asyncio.run(self.http_async_client.aclose()))
        self.enterContext(patch.object(chat_app, "ChatGroq", side_effect=self.create_model))
        self.retriever = self.enterContext(patch.object(chat_app, "get_retriever")).return_value
        self.retriever.invoke.return_value = [
            Document(page_content=PASSAGE, metadata={"source": SOURCE, "start_index": 6})
        ]

    def create_model(self, **options):
        return ChatGroq(
            **options, http_client=self.http_client, http_async_client=self.http_async_client
        )

    def complete(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.assertTrue(request.url.path.endswith("/chat/completions"))
        return httpx.Response(200, json={
            "id": "chat-test",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(self.answer)},
            }],
        })

    def ask(self, question="When does the community library open?", **extra):
        return self.client.post("/chat", json={"message": question, **extra})

    def test_grounded_answer_returns_verified_source_and_uses_default_model(self):
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "response": "It opens at nine on weekdays.",
            "sources": [{
                "source": SOURCE,
                "quote": PASSAGE,
                "start_index": 6,
                "end_index": 6 + len(PASSAGE),
            }],
        })
        self.assertEqual(self.requests[0]["model"], "openai/gpt-oss-20b")
        self.assertEqual(self.requests[0]["response_format"], {"type": "json_object"})
        model_input = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(model_input["passages"], [{"passage_id": 1, "text": PASSAGE}])

    def test_unsupported_question_returns_abstention_even_with_retrieved_passage(self):
        self.answer = {"answer": "", "citations": []}
        response = self.ask("What is the tallest mountain on Mars?")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"response": chat_app.NO_EVIDENCE, "sources": []})
        self.assertEqual(len(self.requests), 1)

    def test_invented_quotes_and_passage_ids_are_not_returned(self):
        for citation in (
            {"passage_id": 1, "quote": "Opens at seven."},
            {"passage_id": 2, "quote": PASSAGE},
        ):
            with self.subTest(citation=citation):
                self.answer["citations"] = [citation]
                self.assertEqual(self.ask().get_json(), {"response": chat_app.NO_EVIDENCE, "sources": []})

    def test_no_retrieved_passages_skips_provider(self):
        self.retriever.invoke.return_value = []
        self.assertEqual(self.ask().get_json(), {"response": chat_app.NO_EVIDENCE, "sources": []})
        self.assertEqual(self.requests, [])

    def test_citation_preserves_source_line_breaks(self):
        passage = "Choose a quiet place where you will not be\ninterrupted"
        self.retriever.invoke.return_value = [
            Document(page_content=passage, metadata={"source": SOURCE, "start_index": 6})
        ]
        self.answer = {
            "answer": "Choose a quiet place.",
            "citations": [{"passage_id": 1, "quote": passage.replace("\n", " ")}],
        }
        source, = self.ask().get_json()["sources"]
        self.assertEqual(source["quote"], passage)
        self.assertEqual(source["end_index"], 6 + len(passage))

    def test_uses_model_override_and_keeps_request_history_separate(self):
        os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
        history = [{"role": "user", "content": "I am asking about the library."}]
        self.assertEqual(self.ask("When does it open?", history=history).status_code, 200)
        self.assertEqual(self.ask().status_code, 200)
        self.assertEqual(self.requests[0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(json.loads(self.requests[0]["messages"][-1]["content"])["history"], history)
        self.assertEqual(json.loads(self.requests[1]["messages"][-1]["content"])["history"], [])

    def test_get_home_without_provider_credentials(self):
        os.environ.pop("GROQ_API_KEY", None)
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.ask()
        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.get_json()["error"])
        self.retriever.invoke.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_invalid_request_payloads_return_json_errors(self):
        for payload in ([], {"message": 42}, {"message": "  "},
                        {"message": "Hello", "history": [{"role": "system", "content": "rules"}]}):
            with self.subTest(payload=payload):
                response = self.client.post("/chat", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.assertEqual(self.requests, [])

    def test_provider_error_returns_json_without_provider_details(self):
        def reject(_request):
            return httpx.Response(401, json={"error": {"message": "Private provider diagnostic."}})

        with httpx.Client(transport=httpx.MockTransport(reject)) as transport:
            self.http_client = transport
            response = self.ask()
        self.assertEqual(response.status_code, 502)
        self.assertIn("model access", response.get_json()["error"])
        self.assertNotIn("Private provider diagnostic.", response.get_data(as_text=True))
