import json
import threading
import unittest
from http.client import HTTPConnection

from retrieval import LexicalRetriever
from shared import Product
from shopping_copilot import ShoppingCopilot
from shopping_copilot.server import create_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(ShoppingCopilot(LexicalRetriever([Product("A", "Black shoe")])), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_and_routes(self):
        self.assertEqual(self.request("GET", "/health")[0], 200)
        self.assertEqual(self.request("GET", "/missing")[0], 404)
        self.assertEqual(self.request("POST", "/missing", "{}")[0], 404)

    def test_search(self):
        status, response = self.request("POST", "/search", json.dumps({"state": {"query": "black shoe"}}))
        self.assertEqual(status, 200)
        self.assertEqual(response["parent_asins"], ["A"])

    def test_turn_preserves_state_and_limits(self):
        status, response = self.request("POST", "/turn", json.dumps({"query": "shoe", "updates": {"hard_constraints": {"color": "black"}}}))
        self.assertEqual(status, 200)
        status, response = self.request("POST", "/turn", json.dumps({"query": "under 100", "previous_state": response["state"], "updates": {"hard_constraints": {"budget_max": 100}}}))
        self.assertEqual(status, 200)
        self.assertEqual(response["state"]["turn"], 2)
        self.assertEqual(response["state"]["hard_constraints"]["color"], ["black"])
        status, _ = self.request("POST", "/turn", json.dumps({"query": "again", "previous_state": {"turn": 10}}))
        self.assertEqual(status, 400)

    def test_invalid_payloads(self):
        for body in ("bad-json", "[]", "{}", '{"state": {"turn": 11}}', '{"state": null}', '{"state": {}, "target_parent_asin": "A"}'):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/search", body)[0], 400)
        self.assertEqual(self.request("POST", "/search", " " * 65537)[0], 413)
