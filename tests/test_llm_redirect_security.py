import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request

from scripts.text.utils.llm import _SameOriginRedirectHandler, _origin, chat


class RedirectSecurityTests(unittest.TestCase):
    def test_cross_origin_redirects_are_blocked(self):
        handler = _SameOriginRedirectHandler()
        request = Request(
            "https://first.example.test/chat/completions",
            data=b"{}",
            headers={"Authorization": "Bearer placeholder"},
            method="POST",
        )
        destinations = (
            "https://second.example.test/collect",
            "http://first.example.test/collect",
            "https://first.example.test:444/collect",
            "https://first.example.test:0/collect",
        )
        for code in (301, 302, 303, 307, 308):
            for url in destinations:
                with self.subTest(code=code, destination=url):
                    with self.assertRaisesRegex(RuntimeError, "cross-origin redirect"):
                        handler.redirect_request(request, None, code, "", {}, url)
        self.assertEqual(_origin("https://EXAMPLE.test"), _origin("https://example.test:443"))
        self.assertEqual(_origin("http://example.test"), _origin("http://example.test:80"))

    def test_http_client_blocks_cross_port_and_allows_same_origin(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(302)
                if self.path.startswith("/cross/"):
                    destination = f"http://127.0.0.1:{target.server_port}/collect"
                else:
                    destination = "/complete"
                self.send_header("Location", destination)
                self.end_headers()

            def do_GET(self):
                received.append(self.path)
                content = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        target = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        for instance in (server, target):
            threading.Thread(target=instance.serve_forever, daemon=True).start()
        try:
            fake_key = "placeholder-secret"
            settings = {"LLM_API_KEY": fake_key, "NO_PROXY": "127.0.0.1"}
            with patch.dict(os.environ, {**settings, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/cross"}):
                with self.assertRaises(RuntimeError) as error:
                    chat("system", "user")
                self.assertNotIn(fake_key, str(error.exception))
                self.assertNotIn("127.0.0.1", str(error.exception))
            self.assertNotIn("/collect", received)
            with patch.dict(os.environ, {**settings, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/same"}):
                self.assertEqual(chat("system", "user"), "ok")
            self.assertIn("/complete", received)
        finally:
            for instance in (server, target):
                instance.shutdown()
                instance.server_close()


if __name__ == "__main__":
    unittest.main()
