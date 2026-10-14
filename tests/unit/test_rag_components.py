"""Unit tests for M6 components: evidence selection, output parsing, provider config, Gemini adapter.
The Gemini tests use httpx.MockTransport (no network, no real model)."""
import json
import unittest

import httpx

from rag_service.api.errors import ApiError
from rag_service.core.config import Settings
from rag_service.rag.ask_service import parse_model_output
from rag_service.rag.evidence import Candidate, content_terms, matched_terms, phonetic_key, select_evidence
from rag_service.rag.providers import DisabledProvider, GeminiProvider, LLMError, build_provider


def cand(i, text):
    return Candidate(f'D{i}', 'v1', text, None, i)


class TestEvidenceSelection(unittest.TestCase):
    def test_generic_terms_are_not_content(self):
        self.assertEqual(content_terms('Me video eka gana viewers la monawada kiyanne?'), set())
        self.assertEqual(content_terms('වීඩියෝ එක ගැන මොනවද කියන්නේ'), set())

    def test_spelling_variants_match(self):
        self.assertEqual(phonetic_key('wathura'), phonetic_key('vatura'))
        self.assertEqual(matched_terms({'wetakolu'}, 'Api uyanakota vatakolu vala'), ['wetakolu'])
        self.assertEqual(matched_terms({'poththa'}, 'pothha ain karanna'), ['poththa'])

    def test_generic_only_comments_rejected(self):
        sel = select_evidence('wetakolu poththa gana video eka', None,
                              [cand(1, 'Nice video ❤'), cand(2, 'Very useful video'), cand(3, 'wetakolu pothha ain karanna')])
        self.assertEqual([c.document_id for c, _ in sel.selected], ['D3'])
        self.assertEqual({c.document_id for c in sel.rejected}, {'D1', 'D2'})

    def test_more_matched_terms_first_then_rank(self):
        sel = select_evidence('wetakolu poththa dara', None,
                              [cand(1, 'wetakolu hondai'), cand(2, 'wetakolu poththa dara ain'), cand(3, 'poththa')])
        self.assertEqual([c.document_id for c, _ in sel.selected], ['D2', 'D1', 'D3'])

    def test_normalised_query_terms_match_sinhala_comments(self):
        sel = select_evidence('bath wathura', 'බත් වතුර', [cand(1, 'බත් එක හැදෙන්න වතුර දැම්මෙ නැහැනේ')])
        self.assertEqual(len(sel.selected), 1)

    def test_max_evidence_cap(self):
        sel = select_evidence('wetakolu', None, [cand(i, 'wetakolu') for i in range(1, 11)], max_evidence=3)
        self.assertEqual(len(sel.selected), 3)


class TestParseModelOutput(unittest.TestCase):
    def test_valid_and_fenced(self):
        self.assertEqual(parse_model_output('{"status":"answered","answer":"x [E1]","citations":["E1"]}'),
                         ('answered', 'x [E1]', ['E1']))
        self.assertEqual(parse_model_output('```json\n{"status":"insufficient_evidence"}\n```')[0], 'insufficient_evidence')

    def test_invalid(self):
        for raw in ('not json', '{"answer":"x"}', '{"status":"answered","answer":""}', '[]'):
            with self.subTest(raw=raw), self.assertRaises(ApiError):
                parse_model_output(raw)


class TestProviderConfig(unittest.TestCase):
    def test_default_disabled(self):
        self.assertIsInstance(build_provider(Settings()), DisabledProvider)

    def test_gemini_without_key_disabled(self):
        self.assertIsInstance(build_provider(Settings(llm_provider='gemini', llm_api_key=None)), DisabledProvider)

    def test_unsupported_provider_disabled(self):
        p = build_provider(Settings(llm_provider='mock', llm_api_key='k'))
        self.assertIsInstance(p, DisabledProvider)  # no simulated provider is selectable by config
        self.assertIn('Unsupported', p.reason)

    def test_key_not_in_repr(self):
        self.assertNotIn('secret-key-123', repr(Settings(llm_api_key='secret-key-123')))


class TestGeminiAdapter(unittest.TestCase):
    def provider(self, handler):
        return GeminiProvider('secret-key-123', 'gemini-test', client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_success_sends_key_in_header_and_parses_text(self):
        seen = {}

        def handler(request):
            seen['url'] = str(request.url)
            seen['key'] = request.headers.get('x-goog-api-key')
            seen['body'] = json.loads(request.content)
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': '{"status":"insufficient_evidence"}'}]}}]})

        out = self.provider(handler).generate('SYS', 'USER')
        self.assertEqual(out, '{"status":"insufficient_evidence"}')
        self.assertEqual(seen['key'], 'secret-key-123')
        self.assertNotIn('secret-key-123', seen['url'])  # key never in the URL
        self.assertEqual(seen['body']['systemInstruction']['parts'][0]['text'], 'SYS')

    def test_http_error_raises_without_leaking_key(self):
        p = self.provider(lambda r: httpx.Response(429, json={'error': {'message': 'Quota exceeded'}}))
        with self.assertRaises(LLMError) as ctx:
            p.generate('s', 'u')
        self.assertIn('429', str(ctx.exception))
        self.assertNotIn('secret-key-123', str(ctx.exception))

    def test_blocked_or_empty_output_raises(self):
        p = self.provider(lambda r: httpx.Response(200, json={'promptFeedback': {'blockReason': 'SAFETY'}}))
        with self.assertRaises(LLMError):
            p.generate('s', 'u')

    def test_network_error_raises(self):
        def handler(request):
            raise httpx.ConnectError('down')
        with self.assertRaises(LLMError):
            self.provider(handler).generate('s', 'u')


if __name__ == '__main__':
    unittest.main()
